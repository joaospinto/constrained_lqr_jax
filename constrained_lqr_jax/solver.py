"""Sequential and parallel solver for stagewise-constrained LQR (no dual reg).

This package solves the dynamic program

    minimize   Σ_k [ ½ xₖᵀQₖxₖ + xₖᵀMₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ ]
                 + ½ x_Nᵀ Q_N x_N + q_Nᵀ x_N
    subject to xₖ₊₁ = Aₖxₖ + Bₖuₖ + cₖ₊₁,        k = 0, ..., N-1
               x₀    = c₀
               Dₖxₖ + Eₖuₖ = dₖ,                  k = 0, ..., N-1

i.e. an LQR problem with stagewise affine equality constraints and **no dual
regularization** (the dynamics are enforced exactly).

Scope of the constraints — arbitrary ``Dx + Eu = d``
----------------------------------------------------
There is **no rank requirement** on ``D_k`` or ``E_k``.  The solver handles

  * state-only constraints (``E_k = 0``, e.g. ``D_k x_k = d_k``),
  * rank-deficient ``E_k``,
  * more constraints than controls (``p > m``),
  * the full-row-rank case, and
  * the unconstrained problem (``p = 0``, empty ``D`` / ``E`` / ``d``),

all with a single, branch-free code path.  Constraints are enforced exactly
(no ``ε·I`` regularization).  This is achieved by *carrying* the constraint
through the cost-to-go (the constraint-carrying / "mixed" interval value
function, formalized in ``MixedConstraintIVF.lean``) rather than eliminating the
constraint multiplier at the base case.

Algorithms
----------
Both a **sequential** and a **parallel** algorithm are provided and produce
identical solutions:

  - Sequential (``factor_and_solve`` / ``solve_general``):
      backward cost-to-go via ``jax.lax.scan`` and a forward primal sweep via
      ``jax.lax.scan``.  O(N · (n + m + p)³) work, O(N) depth.
  - Parallel   (``factor_and_solve_parallel`` / ``solve_general(..., parallel=True)``):
      backward cost-to-go via an ``associative_scan`` over the mixed IVF, and a
      forward primal sweep via an affine ``associative_scan``.  Same work,
      O(log N) depth.

In both cases the dual variables ``(y, λ)`` of the *returned* primal point are
recovered with a single global least-squares solve on the KKT stationarity
equations (:func:`_recover_duals`); this gives the (min-norm) multipliers and is
uniform across all constraint ranks, including the degenerate ones where the
multipliers are non-unique.

Infeasibility / degeneracy
--------------------------
The solver always returns finite numbers (never NaN).  Whether the returned
point is an actual solution is reported explicitly by :func:`solve_general`,
whose :class:`GeneralStatus` ``feasible`` flag and ``residual`` are computed from
the full KKT residual of the returned point.  An infeasible problem (e.g.
``D_0 c_0 ≠ d_0`` for a hard stage-0 state constraint, or a constraint no
control can satisfy) yields ``feasible = False`` and a large ``residual`` rather
than silent NaNs.

References:
  - Sousa-Pinto & Orban, "Dual-Regularized Riccati Recursions"
  - MixedConstraintIVF.lean / MIXED_CONSTRAINT_ANALYSIS.md in this project
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from constrained_lqr_jax.types import (
    FactorizationInputs,
    ParallelFactorizationOutputs,
    SequentialFactorizationOutputs,
    SolveInputs,
    SolveOutputs,
)
from constrained_lqr_jax.helpers import (
    mixed_ivf_base,
    mixed_ivf_combine,
    mixed_ivf_fold_terminal,
    compute_residual,
)


class GeneralStatus(NamedTuple):
    """Outcome of :func:`solve_general`.

    ``feasible`` is ``True`` iff the returned solution satisfies the full KKT
    system to within ``tol``; ``residual`` is the max-norm KKT residual of the
    returned solution (constraint + stationarity + dynamics).
    """

    feasible: jnp.ndarray  # bool scalar
    residual: jnp.ndarray  # float scalar


# ═══════════════════════════════════════════════════════════════════════════
# Backward: constraint-carrying cost-to-go  (P, p, F, Cll, g)
# ═══════════════════════════════════════════════════════════════════════════


def _cost_to_go_sequential(inputs: FactorizationInputs, solve_inputs: SolveInputs):
    """Sequential backward pass via ``jax.lax.scan``.

    Returns the carried cost-to-go ``(P, p, F, Cll, g)`` with arrays of shape
    ``(N+1, ...)`` indexed by stage ``k = 0..N``.
    """
    A, B, Q, M, R, D, E = (
        inputs.A,
        inputs.B,
        inputs.Q,
        inputs.M,
        inputs.R,
        inputs.D,
        inputs.E,
    )
    q, r, c, d = (solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d)
    N, n = A.shape[0], A.shape[1]
    p = D.shape[1]

    def step(carry, stage):
        Pp, ppv, Fp, Cllp, gp = carry
        Qk, Rk, Mk, Ak, Bk, Dk, Ek, qk, rk, ck1, dk = stage

        Hww = jnp.block([[Qk + Ak.T @ Pp @ Ak, Dk.T], [Dk, jnp.zeros((p, p))]])
        Hws = jnp.block(
            [[Mk + Ak.T @ Pp @ Bk, Ak.T @ Fp.T], [Ek, jnp.zeros((p, p))]]
        )
        Hss = jnp.block([[Rk + Bk.T @ Pp @ Bk, Bk.T @ Fp.T], [Fp @ Bk, -Cllp]])
        bw = jnp.concatenate([qk + Ak.T @ Pp @ ck1 + Ak.T @ ppv, -dk])
        bs = jnp.concatenate([rk + Bk.T @ Pp @ ck1 + Bk.T @ ppv, Fp @ ck1 - gp])

        Hss_pinv = jnp.linalg.pinv(Hss)
        Hhat = Hww - Hws @ Hss_pinv @ Hws.T
        bhat = bw - Hws @ Hss_pinv @ bs

        P_new = 0.5 * (Hhat[:n, :n] + Hhat[:n, :n].T)
        F_new = Hhat[n:, :n]
        Cll_new = -0.5 * (Hhat[n:, n:] + Hhat[n:, n:].T)
        p_new = bhat[:n]
        g_new = -bhat[n:]
        new = (P_new, p_new, F_new, Cll_new, g_new)
        return new, new

    terminal = (Q[N], q[N], jnp.zeros((p, n)), jnp.zeros((p, p)), jnp.zeros(p))
    rev = jax.tree.map(lambda x: x[::-1], (Q[:N], R, M, A, B, D, E, q[:N], r, c[1:], d))
    _, outs = jax.lax.scan(step, terminal, rev)
    outs = jax.tree.map(lambda x: x[::-1], outs)  # k = 0..N-1
    cog = jax.tree.map(lambda o, t: jnp.concatenate([o, t[None]]), outs, terminal)
    return cog  # (P, p, F, Cll, g), each (N+1, ...)


def _cost_to_go_parallel(inputs: FactorizationInputs, solve_inputs: SolveInputs):
    """Parallel backward pass via an associative scan over the mixed IVF.

    Per stage we build a base mixed IVF (control eliminated, constraint
    carried); a single ``associative_scan`` composes the suffix intervals
    ``[k, N)``; folding in the terminal cost yields the same constraint-carrying
    cost-to-go ``(P, p, F, Cll, g)`` as :func:`_cost_to_go_sequential`, in
    O(log N) depth.
    """
    A = inputs.A
    N, n = A.shape[0], A.shape[1]
    p = inputs.D.shape[1]

    base = mixed_ivf_base(inputs, solve_inputs)

    # Suffix scan: result[k] = combine(base_k, base_{k+1}, ..., base_{N-1}).
    # `associative_scan(reverse=True)` would place the *latest* stage as the
    # left argument, which is wrong for the non-commutative combine — so we
    # reverse the array, scan forward with the new element on the left, and
    # reverse back.
    rev = jax.tree.map(lambda x: x[::-1], base)
    scanned_rev = jax.lax.associative_scan(
        lambda acc, new: mixed_ivf_combine(new, acc, n, p), rev
    )
    scanned = jax.tree.map(lambda x: x[::-1], scanned_rev)  # IVF_{k->N}

    P, pv, F, Cll, g = jax.vmap(
        lambda ivf: mixed_ivf_fold_terminal(ivf, inputs.Q[N], solve_inputs.q[N], n, p)
    )(scanned)

    # append the terminal cost-to-go at k = N
    P = jnp.concatenate([P, inputs.Q[N][None]])
    pv = jnp.concatenate([pv, solve_inputs.q[N][None]])
    F = jnp.concatenate([F, jnp.zeros((1, p, n))])
    Cll = jnp.concatenate([Cll, jnp.zeros((1, p, p))])
    g = jnp.concatenate([g, jnp.zeros((1, p))])
    return P, pv, F, Cll, g


# ═══════════════════════════════════════════════════════════════════════════
# Forward: primal (x, u)
# ═══════════════════════════════════════════════════════════════════════════


def _stage_feedback(inputs: FactorizationInputs, solve_inputs: SolveInputs, cog):
    """Per-stage affine feedback ``u_k = K_k x_k + k_k`` and the closed-loop
    transition ``x_{k+1} = Φ_k x_k + ψ_k``.

    Solved from the stage KKT (control + carried right constraint + stage
    constraint) with the next-stage cost-to-go on the RHS; a pseudo-inverse
    keeps it well-defined for any constraint rank.  Returns ``(K, k, Φ, ψ)``.
    """
    A, B, M, R, D, E = (inputs.A, inputs.B, inputs.M, inputs.R, inputs.D, inputs.E)
    r, c, d = solve_inputs.r, solve_inputs.c, solve_inputs.d
    P, pv, F, Cll, g = cog
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]

    def stage(Ak, Bk, Mk, Rk, Dk, Ek, rk, ck1, dk, Pp, ppv, Fp, Cllp, gp):
        G = Rk + Bk.T @ Pp @ Bk
        KKT = jnp.block(
            [
                [G, Bk.T @ Fp.T, Ek.T],
                [Fp @ Bk, -Cllp, jnp.zeros((p, p))],
                [Ek, jnp.zeros((p, p)), jnp.zeros((p, p))],
            ]
        )
        # RHS as an affine function of x_k:  rhs = coeff_x @ x_k + const
        coeff_x = jnp.concatenate([-(Mk.T + Bk.T @ Pp @ Ak), -(Fp @ Ak), -Dk], axis=0)
        const = jnp.concatenate(
            [-(Bk.T @ Pp @ ck1 + Bk.T @ ppv + rk), gp - Fp @ ck1, dk]
        )
        rhs = jnp.concatenate([coeff_x, const[:, None]], axis=1)
        sol = jnp.linalg.lstsq(KKT, rhs, rcond=None)[0]
        K = sol[:m, :n]
        k = sol[:m, n]
        Phi = Ak + Bk @ K
        psi = Bk @ k + ck1
        return K, k, Phi, psi

    return jax.vmap(stage)(
        A, B, M, R, D, E, r, c[1:], d, P[1:], pv[1:], F[1:], Cll[1:], g[1:]
    )


def _forward_primal_sequential(inputs, solve_inputs, cog):
    """Sequential forward primal sweep via ``jax.lax.scan``."""
    K, k, Phi, psi = _stage_feedback(inputs, solve_inputs, cog)
    x0 = solve_inputs.c[0]

    def step(x_k, stage):
        K_k, k_k, Phi_k, psi_k = stage
        u_k = K_k @ x_k + k_k
        x_k1 = Phi_k @ x_k + psi_k
        return x_k1, (x_k, u_k)

    x_N, (x_rest, u) = jax.lax.scan(step, x0, (K, k, Phi, psi))
    x = jnp.concatenate([x_rest, x_N[None]])
    return x, u


def _forward_primal_parallel(inputs, solve_inputs, cog):
    """Parallel forward primal sweep via an affine associative scan."""
    K, k, Phi, psi = _stage_feedback(inputs, solve_inputs, cog)
    x0 = solve_inputs.c[0]
    N = inputs.A.shape[0]

    def compose(earlier, later):
        f1, M1 = earlier
        f2, M2 = later
        return (jnp.einsum("...ij,...j->...i", M2, f1) + f2, M2 @ M1)

    composed_psi, composed_Phi = jax.lax.associative_scan(compose, (psi, Phi))
    x_rest = jax.vmap(lambda Mk, fk: Mk @ x0 + fk)(composed_Phi, composed_psi)
    x = jnp.concatenate([x0[None], x_rest])
    u = jax.vmap(lambda K_k, x_k, k_k: K_k @ x_k + k_k)(K, x[:N], k)
    return x, u


# ═══════════════════════════════════════════════════════════════════════════
# Dual (y, λ) recovery
# ═══════════════════════════════════════════════════════════════════════════


def _recover_duals(inputs, solve_inputs, x, u):
    """Recover ``(y, λ)`` by least-squares on the KKT stationarity equations.

    Returns the (min-norm) multipliers consistent with the primal ``(x, u)``;
    these are the unique multipliers whenever the KKT system is nonsingular, and
    a valid KKT point's multipliers otherwise.  Shared by both the sequential
    and parallel solvers.
    """
    A, B, Q, M, R, D, E = (
        inputs.A,
        inputs.B,
        inputs.Q,
        inputs.M,
        inputs.R,
        inputs.D,
        inputs.E,
    )
    q, r = solve_inputs.q, solve_inputs.r
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]

    ny = (N + 1) * n
    nvar = ny + N * p
    nrow = ny + N * m
    K = jnp.zeros((nrow, nvar))
    b = jnp.zeros(nrow)
    I = jnp.eye(n)

    def yi(kk):
        return kk * n

    def li(kk):
        return ny + kk * p

    row = 0
    # x-stationarity k = 0..N-1
    for kk in range(N):
        K = K.at[row : row + n, yi(kk) : yi(kk) + n].set(-I)
        K = K.at[row : row + n, yi(kk + 1) : yi(kk + 1) + n].set(A[kk].T)
        K = K.at[row : row + n, li(kk) : li(kk) + p].set(D[kk].T)
        b = b.at[row : row + n].set(-(Q[kk] @ x[kk] + M[kk] @ u[kk] + q[kk]))
        row += n
    # terminal
    K = K.at[row : row + n, yi(N) : yi(N) + n].set(-I)
    b = b.at[row : row + n].set(-(Q[N] @ x[N] + q[N]))
    row += n
    # u-stationarity k = 0..N-1
    for kk in range(N):
        K = K.at[row : row + m, yi(kk + 1) : yi(kk + 1) + n].set(B[kk].T)
        K = K.at[row : row + m, li(kk) : li(kk) + p].set(E[kk].T)
        b = b.at[row : row + m].set(-(M[kk].T @ x[kk] + R[kk] @ u[kk] + r[kk]))
        row += m

    sol = jnp.linalg.lstsq(K, b, rcond=None)[0]
    y = sol[:ny].reshape(N + 1, n)
    lam = sol[ny:].reshape(N, p)
    return y, lam


# ═══════════════════════════════════════════════════════════════════════════
# Public factor / solve entrypoints
# ═══════════════════════════════════════════════════════════════════════════


def _zero_solve_inputs(inputs: FactorizationInputs) -> SolveInputs:
    """RHS with the right shapes, used to extract LHS-only recurrences."""
    N, n, m = inputs.B.shape
    p = inputs.D.shape[1]
    dtype = jnp.result_type(inputs.A, inputs.B, inputs.Q, inputs.M, inputs.R)
    return SolveInputs(
        q=jnp.zeros((N + 1, n), dtype=dtype),
        r=jnp.zeros((N, m), dtype=dtype),
        c=jnp.zeros((N + 1, n), dtype=dtype),
        d=jnp.zeros((N, p), dtype=dtype),
    )


def _factorization_from_cog(
    inputs: FactorizationInputs,
    cog,
    output_type,
):
    """Build reusable feedback data from a quadratic cost-to-go."""
    zero_inputs = _zero_solve_inputs(inputs)
    K, _, Phi, _ = _stage_feedback(inputs, zero_inputs, cog)
    P, _, F, Cll, _ = cog
    return output_type(P=P, F=F, Cll=Cll, K=K, Phi=Phi)


@jax.jit
def factor(inputs: FactorizationInputs) -> SequentialFactorizationOutputs:
    """Factor the constrained LQR LHS with a sequential backward pass.

    The returned data is independent of ``SolveInputs`` and can be reused for
    multiple right-hand sides ``(q, r, c, d)`` with the same dynamics, cost
    Hessians and constraint matrices.
    """
    cog = _cost_to_go_sequential(inputs, _zero_solve_inputs(inputs))
    return _factorization_from_cog(inputs, cog, SequentialFactorizationOutputs)


@jax.jit
def factor_parallel(inputs: FactorizationInputs) -> ParallelFactorizationOutputs:
    """Factor the constrained LQR LHS with the parallel backward pass."""
    cog = _cost_to_go_parallel(inputs, _zero_solve_inputs(inputs))
    return _factorization_from_cog(inputs, cog, ParallelFactorizationOutputs)



def _affine_cost_to_go_parallel(
    inputs: FactorizationInputs,
    factorization_outputs,
    solve_inputs: SolveInputs,
):
    """Recover RHS-dependent affine cost-to-go terms with an associative scan."""
    A, B, M, R, D, E = (inputs.A, inputs.B, inputs.M, inputs.R, inputs.D, inputs.E)
    q, r, c, d = (solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d)
    P, F, Cll = (
        factorization_outputs.P,
        factorization_outputs.F,
        factorization_outputs.Cll,
    )
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]

    def stage_map(Ak, Bk, Mk, Rk, Dk, Ek, qk, rk, ck1, dk, Pp, Fp, Cllp):
        Hws = jnp.block(
            [[Mk + Ak.T @ Pp @ Bk, Ak.T @ Fp.T], [Ek, jnp.zeros((p, p))]]
        )
        Hss = jnp.block([[Rk + Bk.T @ Pp @ Bk, Bk.T @ Fp.T], [Fp @ Bk, -Cllp]])
        Hws_Hss_pinv = Hws @ jnp.linalg.pinv(Hss)

        bw_coeff = jnp.block(
            [
                [Ak.T, jnp.zeros((n, p))],
                [jnp.zeros((p, n)), jnp.zeros((p, p))],
            ]
        )
        bs_coeff = jnp.block(
            [
                [Bk.T, jnp.zeros((m, p))],
                [jnp.zeros((p, n)), -jnp.eye(p)],
            ]
        )
        bhat_coeff = bw_coeff - Hws_Hss_pinv @ bs_coeff
        Z = jnp.concatenate([bhat_coeff[:n], -bhat_coeff[n:]], axis=0)

        bw_const = jnp.concatenate([qk + Ak.T @ Pp @ ck1, -dk])
        bs_const = jnp.concatenate([rk + Bk.T @ Pp @ ck1, Fp @ ck1])
        bhat_const = bw_const - Hws_Hss_pinv @ bs_const
        z = jnp.concatenate([bhat_const[:n], -bhat_const[n:]])
        return Z, z

    Z, z = jax.vmap(stage_map)(
        A, B, M, R, D, E, q[:N], r, c[1:], d, P[1:], F[1:], Cll[1:]
    )

    def compose(earlier, later):
        Z1, z1 = earlier
        Z2, z2 = later
        return Z2 @ Z1, jnp.einsum("...ij,...j->...i", Z2, z1) + z2

    Z_scan, z_scan = jax.lax.associative_scan(compose, (Z, z), reverse=True)
    terminal = jnp.concatenate([q[N], jnp.zeros(p, dtype=q.dtype)])
    affine = jax.vmap(lambda Zk, zk: Zk @ terminal + zk)(Z_scan, z_scan)
    pv = jnp.concatenate([affine[:, :n], q[N][None]])
    g = jnp.concatenate([affine[:, n:], jnp.zeros((1, p), dtype=q.dtype)])
    return P, pv, F, Cll, g


def _affine_cost_to_go_sequential(
    inputs: FactorizationInputs,
    factorization_outputs,
    solve_inputs: SolveInputs,
):
    """Recover RHS-dependent affine cost-to-go terms from a factorization."""
    A, B, M, R, D, E = (inputs.A, inputs.B, inputs.M, inputs.R, inputs.D, inputs.E)
    q, r, c, d = (solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d)
    P, F, Cll = (
        factorization_outputs.P,
        factorization_outputs.F,
        factorization_outputs.Cll,
    )
    N, n = A.shape[0], A.shape[1]
    p = D.shape[1]

    def step(carry, stage):
        ppv, gp = carry
        Ak, Bk, Mk, Rk, Dk, Ek, qk, rk, ck1, dk, Pp, Fp, Cllp = stage

        Hws = jnp.block(
            [[Mk + Ak.T @ Pp @ Bk, Ak.T @ Fp.T], [Ek, jnp.zeros((p, p))]]
        )
        Hss = jnp.block([[Rk + Bk.T @ Pp @ Bk, Bk.T @ Fp.T], [Fp @ Bk, -Cllp]])
        bw = jnp.concatenate([qk + Ak.T @ Pp @ ck1 + Ak.T @ ppv, -dk])
        bs = jnp.concatenate([rk + Bk.T @ Pp @ ck1 + Bk.T @ ppv, Fp @ ck1 - gp])

        bhat = bw - Hws @ jnp.linalg.pinv(Hss) @ bs
        new = (bhat[:n], -bhat[n:])
        return new, new

    terminal = (q[N], jnp.zeros(p, dtype=q.dtype))
    rev = jax.tree.map(
        lambda x: x[::-1],
        (A, B, M, R, D, E, q[:N], r, c[1:], d, P[1:], F[1:], Cll[1:]),
    )
    _, outs = jax.lax.scan(step, terminal, rev)
    outs = jax.tree.map(lambda x: x[::-1], outs)
    pv = jnp.concatenate([outs[0], terminal[0][None]])
    g = jnp.concatenate([outs[1], terminal[1][None]])
    return P, pv, F, Cll, g


def _stage_feedforward(
    inputs: FactorizationInputs,
    factorization_outputs,
    solve_inputs: SolveInputs,
    cog,
):
    """Per-stage affine feedforward and closed-loop offset from a factorization."""
    B, R, D, E = (inputs.B, inputs.R, inputs.D, inputs.E)
    r, c, d = solve_inputs.r, solve_inputs.c, solve_inputs.d
    P, pv, F, Cll, g = cog
    m, p = B.shape[2], D.shape[1]

    def stage(Bk, Rk, Dk, Ek, rk, ck1, dk, Pp, ppv, Fp, Cllp, gp):
        G = Rk + Bk.T @ Pp @ Bk
        KKT = jnp.block(
            [
                [G, Bk.T @ Fp.T, Ek.T],
                [Fp @ Bk, -Cllp, jnp.zeros((p, p))],
                [Ek, jnp.zeros((p, p)), jnp.zeros((p, p))],
            ]
        )
        rhs = jnp.concatenate(
            [-(Bk.T @ Pp @ ck1 + Bk.T @ ppv + rk), gp - Fp @ ck1, dk]
        )
        sol = jnp.linalg.lstsq(KKT, rhs, rcond=None)[0]
        k = sol[:m]
        psi = Bk @ k + ck1
        return k, psi

    return jax.vmap(stage)(
        B, R, D, E, r, c[1:], d, P[1:], pv[1:], F[1:], Cll[1:], g[1:]
    )


def _forward_primal_from_factorization_sequential(
    factorization_outputs,
    solve_inputs: SolveInputs,
    k,
    psi,
):
    """Sequential forward primal sweep using factored feedback matrices."""
    K, Phi = factorization_outputs.K, factorization_outputs.Phi
    x0 = solve_inputs.c[0]

    def step(x_k, stage):
        K_k, k_k, Phi_k, psi_k = stage
        u_k = K_k @ x_k + k_k
        x_k1 = Phi_k @ x_k + psi_k
        return x_k1, (x_k, u_k)

    x_N, (x_rest, u) = jax.lax.scan(step, x0, (K, k, Phi, psi))
    x = jnp.concatenate([x_rest, x_N[None]])
    return x, u


def _forward_primal_from_factorization_parallel(
    factorization_outputs,
    solve_inputs: SolveInputs,
    k,
    psi,
):
    """Parallel forward primal sweep using factored feedback matrices."""
    K, Phi = factorization_outputs.K, factorization_outputs.Phi
    x0 = solve_inputs.c[0]

    def compose(earlier, later):
        f1, M1 = earlier
        f2, M2 = later
        return (jnp.einsum("...ij,...j->...i", M2, f1) + f2, M2 @ M1)

    composed_psi, composed_Phi = jax.lax.associative_scan(compose, (psi, Phi))
    x_rest = jax.vmap(lambda Mk, fk: Mk @ x0 + fk)(composed_Phi, composed_psi)
    x = jnp.concatenate([x0[None], x_rest])
    u = jax.vmap(lambda K_k, x_k, k_k: K_k @ x_k + k_k)(K, x[:-1], k)
    return x, u


@jax.jit
def solve(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: SequentialFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using a sequential factorization."""
    cog = _affine_cost_to_go_sequential(
        factorization_inputs, factorization_outputs, solve_inputs
    )
    k, psi = _stage_feedforward(
        factorization_inputs, factorization_outputs, solve_inputs, cog
    )
    x, u = _forward_primal_from_factorization_sequential(
        factorization_outputs, solve_inputs, k, psi
    )
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def solve_parallel(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: ParallelFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using parallel backward/forward scans.

    The RHS-dependent cost-to-go is recovered by composing affine maps over
    the factored quadratic blocks, and the primal pass uses an associative scan.
    """
    cog = _affine_cost_to_go_parallel(
        factorization_inputs, factorization_outputs, solve_inputs
    )
    k, psi = _stage_feedforward(
        factorization_inputs, factorization_outputs, solve_inputs, cog
    )
    x, u = _forward_primal_from_factorization_parallel(
        factorization_outputs, solve_inputs, k, psi
    )
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def factor_and_solve(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Sequential factor + solve; return the KKT point."""
    return solve(factorization_inputs, factor(factorization_inputs), solve_inputs)


@jax.jit
def factor_and_solve_parallel(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Fully parallel factor + solve; return the KKT point."""
    cog = _cost_to_go_parallel(factorization_inputs, solve_inputs)
    x, u = _forward_primal_parallel(factorization_inputs, solve_inputs, cog)
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    _, k, _, _ = _stage_feedback(factorization_inputs, solve_inputs, cog)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


def solve_general(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
    tol: float = 1e-6,
    parallel: bool = False,
) -> tuple[SolveOutputs, GeneralStatus]:
    """Solve and report feasibility explicitly (no silent NaNs).

    Set ``parallel=True`` to use the fully parallel factor-and-solve path.

    Returns ``(solution, status)`` where ``status.feasible`` is ``True`` iff the
    returned solution satisfies the full KKT system to within ``tol``.  For an
    infeasible or degenerate problem the solution is still finite and
    ``status.feasible`` is ``False`` with a large ``status.residual``.
    """
    if parallel:
        sol = factor_and_solve_parallel(factorization_inputs, solve_inputs)
    else:
        sol = factor_and_solve(factorization_inputs, solve_inputs)
    res = compute_residual(factorization_inputs, solve_inputs, sol)
    rnorm = jnp.max(jnp.abs(res))
    feasible = jnp.isfinite(rnorm) & (rnorm <= tol)
    return sol, GeneralStatus(feasible=feasible, residual=rnorm)
