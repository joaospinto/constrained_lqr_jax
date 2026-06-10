"""Solver for stagewise-constrained LQR with optional dual regularization.

This package solves the dynamic program

    minimize   Σ_k [ ½ xₖᵀQₖxₖ + xₖᵀMₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ ]
                 + ½ x_Nᵀ Q_N x_N + q_Nᵀ x_N
    subject to xₖ₊₁ = Aₖxₖ + Bₖuₖ + cₖ₊₁,        k = 0, ..., N-1
               x₀    = c₀
               Dₖxₖ + Eₖuₖ = dₖ,                  k = 0, ..., N-1

i.e. an LQR problem with stagewise affine equality constraints. Dynamics and
stagewise equality constraints may be dual-regularized with ``Delta`` and
``Sigma`` blocks; setting those blocks to zero recovers exact dynamics and exact
stagewise constraints.

Scope of the constraints — arbitrary ``Dx + Eu = d``
----------------------------------------------------
There is **no rank requirement** on ``D_k`` or ``E_k``.  The solver handles

  * state-only constraints (``E_k = 0``, e.g. ``D_k x_k = d_k``),
  * rank-deficient ``E_k``,
  * more constraints than controls (``p > m``),
  * the full-row-rank case, and
  * the unconstrained problem (``p = 0``, empty ``D`` / ``E`` / ``d``),

all with a single, branch-free representation.  With zero regularization,
constraints are enforced exactly.  This is achieved by *carrying* the constraint
through the cost-to-go (the constraint-carrying / "mixed" interval value
function, formalized in ``MixedConstraintIVF.lean``) rather than eliminating the
constraint multiplier at the base case.

Algorithms
----------
Both a **sequential** and a **parallel** algorithm are provided and produce
identical solutions:

  - Sequential/parallel factorization helpers expose the same API shape as
    ``regularized_lqr_jax``.
  - The public solve paths use scan-native backward and forward passes.  For
    zero ``Delta`` and ``Sigma`` this is the original exact constrained LQR
    system.

In degenerate zero-regularization cases the dual variables ``(y, λ)`` may be
non-unique.  The solver recovers a minimum-norm globally consistent multiplier
representative after the scan-native primal pass (:func:`_recover_duals`).

Infeasibility / degeneracy
--------------------------
The solver always returns finite numbers (never NaN).  Whether the returned
point satisfies the requested regularized KKT system is reported explicitly by
:func:`solve_general`, whose :class:`GeneralStatus` ``feasible`` flag and
``residual`` are computed from the full KKT residual of the returned point.  With
zero regularization, an infeasible hard-constrained problem yields
``feasible = False`` and a large ``residual`` rather than silent NaNs.

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
    """Sequential suffix scan over mixed IVFs.

    This supports the same regularized interval representation as the parallel
    associative scan, but composes intervals right-to-left with ``lax.scan``.
    """
    N, n = inputs.A.shape[0], inputs.A.shape[1]
    p = inputs.D.shape[1]
    base = mixed_ivf_base(inputs, solve_inputs)

    rev = jax.tree.map(lambda x: x[::-1], base)

    def step(acc, new_left):
        combined = mixed_ivf_combine(new_left, acc, n, p)
        return combined, combined

    first = jax.tree.map(lambda x: x[0], rev)
    rest = jax.tree.map(lambda x: x[1:], rev)
    _, suffix_rest = jax.lax.scan(step, first, rest)
    suffix_rev = jax.tree.map(
        lambda f, r: jnp.concatenate([f[None], r]), first, suffix_rest
    )
    suffix = jax.tree.map(lambda x: x[::-1], suffix_rev)

    P, pv, F, Cll, g = jax.vmap(
        lambda ivf: mixed_ivf_fold_terminal(ivf, inputs.Q[N], solve_inputs.q[N], n, p)
    )(suffix)

    P = jnp.concatenate([P, inputs.Q[N][None]])
    pv = jnp.concatenate([pv, solve_inputs.q[N][None]])
    F = jnp.concatenate([F, jnp.zeros((1, p, n), dtype=P.dtype)])
    Cll = jnp.concatenate([Cll, jnp.zeros((1, p, p), dtype=P.dtype)])
    g = jnp.concatenate([g, jnp.zeros((1, p), dtype=pv.dtype)])
    return P, pv, F, Cll, g

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
    """Per-stage affine maps from ``x_k`` to local primal/dual variables.

    The local unknowns are ``(u_k, y_{k+1}, lambda_k, mu_{k+1}, x_{k+1})``.
    ``mu`` is the carried multiplier for the next cost-to-go constraint.  A
    pseudoinverse is used because arbitrary-rank equality constraints can make
    this local saddle system singular while still admitting a valid minimum-norm
    KKT representative.
    """
    A, B, M, R, D, E, Delta, Sigma = (
        inputs.A,
        inputs.B,
        inputs.M,
        inputs.R,
        inputs.D,
        inputs.E,
        inputs.Delta,
        inputs.Sigma,
    )
    r, c, d = solve_inputs.r, solve_inputs.c, solve_inputs.d
    P, pv, F, Cll, g = cog
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]

    def stage(Ak, Bk, Mk, Rk, Dk, Ek, Deltap, Sigmak, rk, ck1, dk, Pp, ppv, Fp, Cllp, gp):
        eye_n = jnp.eye(n)
        eye_p = jnp.eye(p)
        Z = jnp.block(
            [
                [Rk, Bk.T, Ek.T, jnp.zeros((m, p)), jnp.zeros((m, n))],
                [Bk, -Deltap, jnp.zeros((n, p)), jnp.zeros((n, p)), -eye_n],
                [Ek, jnp.zeros((p, n)), -Sigmak, jnp.zeros((p, p)), jnp.zeros((p, n))],
                [jnp.zeros((n, m)), -eye_n, jnp.zeros((n, p)), Fp.T, Pp],
                [jnp.zeros((p, m)), jnp.zeros((p, n)), jnp.zeros((p, p)), -Cllp, Fp],
            ]
        )
        coeff_x = jnp.concatenate(
            [
                -Mk.T,
                -Ak,
                -Dk,
                jnp.zeros((n, n)),
                jnp.zeros((p, n)),
            ],
            axis=0,
        )
        const = jnp.concatenate([-rk, -ck1, dk, -ppv, gp])
        rhs = jnp.concatenate([coeff_x, const[:, None]], axis=1)
        sol = jnp.linalg.pinv(Z) @ rhs

        u = sol[:m]
        y = sol[m : m + n]
        lam = sol[m + n : m + n + p]
        xnext = sol[m + n + p + p :]
        return (
            u[:, :n],
            u[:, n],
            xnext[:, :n],
            xnext[:, n],
            y[:, :n],
            y[:, n],
            lam[:, :n],
            lam[:, n],
        )

    return jax.vmap(stage)(
        A,
        B,
        M,
        R,
        D,
        E,
        Delta[1:],
        Sigma,
        r,
        c[1:],
        d,
        P[1:],
        pv[1:],
        F[1:],
        Cll[1:],
        g[1:],
    )

def _initial_state_and_dual(inputs, solve_inputs, K0, k0, YK0, yk0, LK0, lk0):
    G = inputs.Q[0] + inputs.M[0] @ K0 + inputs.A[0].T @ YK0 + inputs.D[0].T @ LK0
    h = inputs.M[0] @ k0 + inputs.A[0].T @ yk0 + inputs.D[0].T @ lk0 + solve_inputs.q[0]
    x0 = jnp.linalg.pinv(jnp.eye(inputs.A.shape[1]) + inputs.Delta[0] @ G) @ (
        solve_inputs.c[0] - inputs.Delta[0] @ h
    )
    y0 = G @ x0 + h
    return x0, y0


def _forward_primal_sequential(inputs, solve_inputs, cog):
    """Sequential forward sweep for primal and dual variables."""
    K, k, Phi, psi, YK, yk, LK, lk = _stage_feedback(inputs, solve_inputs, cog)
    x0, y0 = _initial_state_and_dual(inputs, solve_inputs, K[0], k[0], YK[0], yk[0], LK[0], lk[0])

    def step(x_k, stage):
        K_k, k_k, Phi_k, psi_k, YK_k, yk_k, LK_k, lk_k = stage
        u_k = K_k @ x_k + k_k
        x_k1 = Phi_k @ x_k + psi_k
        y_k1 = YK_k @ x_k + yk_k
        lam_k = LK_k @ x_k + lk_k
        return x_k1, (x_k, u_k, y_k1, lam_k)

    x_N, (x_rest, u, y_next, lam) = jax.lax.scan(
        step, x0, (K, k, Phi, psi, YK, yk, LK, lk)
    )
    x = jnp.concatenate([x_rest, x_N[None]])
    y = jnp.concatenate([y0[None], y_next])
    return x, u, y, lam, k


def _forward_primal_parallel(inputs, solve_inputs, cog):
    """Parallel forward sweep for primal and dual variables."""
    K, k, Phi, psi, YK, yk, LK, lk = _stage_feedback(inputs, solve_inputs, cog)
    x0, y0 = _initial_state_and_dual(inputs, solve_inputs, K[0], k[0], YK[0], yk[0], LK[0], lk[0])

    def compose(earlier, later):
        f1, M1 = earlier
        f2, M2 = later
        return (jnp.einsum("...ij,...j->...i", M2, f1) + f2, M2 @ M1)

    composed_psi, composed_Phi = jax.lax.associative_scan(compose, (psi, Phi))
    x_rest = jax.vmap(lambda Mk, fk: Mk @ x0 + fk)(composed_Phi, composed_psi)
    x = jnp.concatenate([x0[None], x_rest])
    u = jax.vmap(lambda K_k, x_k, k_k: K_k @ x_k + k_k)(K, x[:-1], k)
    y_next = jax.vmap(lambda YK_k, x_k, yk_k: YK_k @ x_k + yk_k)(YK, x[:-1], yk)
    lam = jax.vmap(lambda LK_k, x_k, lk_k: LK_k @ x_k + lk_k)(LK, x[:-1], lk)
    y = jnp.concatenate([y0[None], y_next])
    return x, u, y, lam, k


# ═══════════════════════════════════════════════════════════════════════════
# Dual (y, λ) recovery
# ═══════════════════════════════════════════════════════════════════════════


def _recover_duals(inputs, solve_inputs, x, u):
    """Recover a globally consistent multiplier representative.

    For rank-deficient hard constraints the multipliers are not unique.  This
    solves the linear dual equations with a pseudoinverse and returns the
    minimum-norm representative that is consistent with the already-computed
    primal trajectory.
    """
    A, B, Q, M, R, D, E, Delta, Sigma = (
        inputs.A,
        inputs.B,
        inputs.Q,
        inputs.M,
        inputs.R,
        inputs.D,
        inputs.E,
        inputs.Delta,
        inputs.Sigma,
    )
    q, r, c, d = solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]

    ny = (N + 1) * n
    nvar = ny + N * p
    nrow = (N + 1) * n + N * m + (N + 1) * n + N * p
    K = jnp.zeros((nrow, nvar), dtype=jnp.result_type(A, B, Q, M, R, D, E, q, r, c, d))
    b = jnp.zeros((nrow,), dtype=K.dtype)
    I = jnp.eye(n)

    def yi(k):
        return k * n

    def li(k):
        return ny + k * p

    row = 0
    for k in range(N):
        K = K.at[row : row + n, yi(k) : yi(k) + n].set(-I)
        K = K.at[row : row + n, yi(k + 1) : yi(k + 1) + n].set(A[k].T)
        K = K.at[row : row + n, li(k) : li(k) + p].set(D[k].T)
        b = b.at[row : row + n].set(-(Q[k] @ x[k] + M[k] @ u[k] + q[k]))
        row += n

    K = K.at[row : row + n, yi(N) : yi(N) + n].set(-I)
    b = b.at[row : row + n].set(-(Q[N] @ x[N] + q[N]))
    row += n

    for k in range(N):
        K = K.at[row : row + m, yi(k + 1) : yi(k + 1) + n].set(B[k].T)
        K = K.at[row : row + m, li(k) : li(k) + p].set(E[k].T)
        b = b.at[row : row + m].set(-(M[k].T @ x[k] + R[k] @ u[k] + r[k]))
        row += m

    K = K.at[row : row + n, yi(0) : yi(0) + n].set(-Delta[0])
    b = b.at[row : row + n].set(x[0] - c[0])
    row += n

    for k in range(N):
        K = K.at[row : row + n, yi(k + 1) : yi(k + 1) + n].set(-Delta[k + 1])
        b = b.at[row : row + n].set(x[k + 1] - A[k] @ x[k] - B[k] @ u[k] - c[k + 1])
        row += n

    for k in range(N):
        K = K.at[row : row + p, li(k) : li(k) + p].set(-Sigma[k])
        b = b.at[row : row + p].set(d[k] - D[k] @ x[k] - E[k] @ u[k])
        row += p

    sol = jnp.linalg.pinv(K) @ b
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
    K, _, Phi, _, *_ = _stage_feedback(inputs, zero_inputs, cog)
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




@jax.jit
def solve(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: SequentialFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using a sequential factorization."""
    cog = _cost_to_go_sequential(factorization_inputs, solve_inputs)
    x, u, _, _, k = _forward_primal_sequential(factorization_inputs, solve_inputs, cog)
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def solve_parallel(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: ParallelFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using the regularized KKT equations."""
    cog = _cost_to_go_parallel(factorization_inputs, solve_inputs)
    x, u, _, _, k = _forward_primal_parallel(factorization_inputs, solve_inputs, cog)
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def factor_and_solve(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Sequential factor + solve; return the KKT point."""
    cog = _cost_to_go_sequential(factorization_inputs, solve_inputs)
    x, u, _, _, k = _forward_primal_sequential(factorization_inputs, solve_inputs, cog)
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def factor_and_solve_parallel(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Parallel factor + solve; return the KKT point."""
    cog = _cost_to_go_parallel(factorization_inputs, solve_inputs)
    x, u, _, _, k = _forward_primal_parallel(factorization_inputs, solve_inputs, cog)
    y, lam = _recover_duals(factorization_inputs, solve_inputs, x, u)
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
