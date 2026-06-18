"""Solver for stagewise-constrained LQR with optional dual regularization.

This package solves the dynamic program

    minimize   Σ_k [ ½ xₖᵀQₖxₖ + xₖᵀMₖuₖ + ½ uₖᵀRₖuₖ + qₖᵀxₖ + rₖᵀuₖ ]
                 + ½ x_Nᵀ Q_N x_N + q_Nᵀ x_N
    subject to xₖ₊₁ = Aₖxₖ + Bₖuₖ + cₖ₊₁,        k = 0, ..., N-1
               x₀    = c₀
               Dₖxₖ + Eₖuₖ = dₖ,                  k = 0, ..., N-1
               D_N x_N = d_N,

i.e. an LQR problem with stagewise affine equality constraints and terminal
state constraints. Dynamics and equality constraints may be dual-regularized
with ``Delta`` and ``Sigma`` blocks; setting those blocks to zero recovers exact
hard equalities.

Scope of the constraints
------------------------
The solver handles

  * state-only constraints (``E_k = 0``, e.g. ``D_k x_k = d_k``),
  * rank-deficient ``E_k``,
  * more constraints than controls (``p > m``),
  * the full-row-rank case, and
  * the unconstrained problem (``p = 0``, empty ``D`` / ``E`` / ``d``),

with fixed-size local algebra and no horizon-sized dense solve.  Hard
state-only constraints with full-row-rank ``D_k`` use a nullspace-coordinate
path, because their exact suffix feasibility domain can have more rows than the
mixed-IVF ``F x = g`` carrier can represent.  Other cases use the
constraint-carrying / "mixed" interval value function (formalized in
``MixedConstraintIVF.lean``), which carries constraints through the cost-to-go
rather than eliminating the constraint multiplier at the base case.

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
non-unique.  The solver recovers a globally consistent multiplier
representative after the scan-native primal pass with a structured dual-only
reduction (:func:`_recover_duals`).

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

# Local cyclic-reduction pivots can be nearly singular even when the global PSD
# block-tridiagonal system is consistent.  A conservative Hermitian cutoff keeps
# those local rank decisions aligned with the singular global solve in tested
# hard-constraint cases.
_BLOCK_TRIDIAG_PINV_RTOL = 1e-6


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
        lambda ivf: mixed_ivf_fold_terminal(
            ivf,
            inputs.Q[N],
            solve_inputs.q[N],
            inputs.D[N],
            inputs.Sigma[N],
            solve_inputs.d[N],
            n,
            p,
        )
    )(suffix)

    P = jnp.concatenate([P, inputs.Q[N][None]])
    pv = jnp.concatenate([pv, solve_inputs.q[N][None]])
    F = jnp.concatenate([F, inputs.D[N][None]])
    Cll = jnp.concatenate([Cll, inputs.Sigma[N][None]])
    g = jnp.concatenate([g, solve_inputs.d[N][None]])
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
        lambda ivf: mixed_ivf_fold_terminal(
            ivf,
            inputs.Q[N],
            solve_inputs.q[N],
            inputs.D[N],
            inputs.Sigma[N],
            solve_inputs.d[N],
            n,
            p,
        )
    )(scanned)

    # append the terminal cost-to-go at k = N
    P = jnp.concatenate([P, inputs.Q[N][None]])
    pv = jnp.concatenate([pv, solve_inputs.q[N][None]])
    F = jnp.concatenate([F, inputs.D[N][None]])
    Cll = jnp.concatenate([Cll, inputs.Sigma[N][None]])
    g = jnp.concatenate([g, solve_inputs.d[N][None]])
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
        eye_n = jnp.eye(n, dtype=Ak.dtype)
        Rinv = jnp.linalg.inv(Rk)
        RiMt = Rinv @ Mk.T
        RiBt = Rinv @ Bk.T
        RiEt = Rinv @ Ek.T
        Rir = Rinv @ rk

        Ao = Ak - Bk @ RiMt
        Cdyn = Bk @ RiBt + Deltap
        co = ck1 - Bk @ Rir
        Cyl = Bk @ RiEt
        Cll_stage = Ek @ RiEt + Sigmak
        Fstage = Dk - Ek @ RiMt
        gstage = dk + Ek @ Rir

        # Unknowns are (x_{k+1}, mu_{k+1}, lambda_k).  The dynamics dual
        # y_{k+1} and control u_k are reconstructed afterwards.
        Ksmall = jnp.zeros((n + 2 * p, n + 2 * p), dtype=Ak.dtype)
        Ksmall = Ksmall.at[:n, :n].set(eye_n + Cdyn @ Pp)
        Ksmall = Ksmall.at[:n, n : n + p].set(Cdyn @ Fp.T)
        Ksmall = Ksmall.at[:n, n + p :].set(Cyl)
        Ksmall = Ksmall.at[n : n + p, :n].set(Cyl.T @ Pp)
        Ksmall = Ksmall.at[n : n + p, n : n + p].set(Cyl.T @ Fp.T)
        Ksmall = Ksmall.at[n : n + p, n + p :].set(Cll_stage)
        Ksmall = Ksmall.at[n + p :, :n].set(Fp)
        Ksmall = Ksmall.at[n + p :, n : n + p].set(-Cllp)

        coeff_x = jnp.concatenate(
            [Ao, Fstage, jnp.zeros((p, n), dtype=Ak.dtype)], axis=0
        )
        const = jnp.concatenate([co - Cdyn @ ppv, -gstage - Cyl.T @ ppv, gp])
        rhs = jnp.concatenate([coeff_x, const[:, None]], axis=1)
        if p == 0:
            sol = jnp.linalg.solve(Ksmall, rhs)
        else:
            sol = jnp.linalg.lstsq(Ksmall, rhs, rcond=None)[0]

        xnext = sol[:n]
        mu = sol[n : n + p]
        lam = sol[n + p :]
        y = Pp @ xnext + Fp.T @ mu
        y = y.at[:, n].add(ppv)
        u_rhs = jnp.concatenate([RiMt, Rir[:, None]], axis=1)
        u = -(u_rhs + RiBt @ y + RiEt @ lam)
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
        D[:N],
        E,
        Delta[1:],
        Sigma[:N],
        r,
        c[1:],
        d[:N],
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


class _BlockTridiagLevel(NamedTuple):
    Sinv: jnp.ndarray
    B_left: jnp.ndarray
    B_right: jnp.ndarray


class _BlockTridiagFactor(NamedTuple):
    single_pinv: jnp.ndarray
    endpoint_pinv: jnp.ndarray
    levels: tuple[_BlockTridiagLevel, ...]


class _DualRecoveryCache(NamedTuple):
    block_factor: _BlockTridiagFactor
    C0: jnp.ndarray
    C1: jnp.ndarray
    C_terminal: jnp.ndarray
    S_pinv: jnp.ndarray
    S_pinv_terminal: jnp.ndarray


def _block_tridiag_factor(diag, upper):
    """Factor a symmetric block-tridiagonal LHS by cyclic reduction.

    The reduction and recovery loops have static ``O(log N)`` levels under
    ``jit``; each level vmaps independent local Schur complements.  A
    prefix/suffix marginal scan can recover all nodes independently for
    nonsingular systems, but in degenerate hard-constrained cases those
    independent minimum-norm marginals need not be jointly compatible.  The
    top-down recovery here conditions each eliminated midpoint on already
    chosen parent endpoints, preserving global consistency.
    """
    n = diag.shape[-1]
    dtype = diag.dtype
    single_pinv = jnp.linalg.pinv(
        0.5 * (diag[0] + diag[0].T),
        hermitian=True,
        rtol=_BLOCK_TRIDIAG_PINV_RTOL,
    )
    if upper.shape[0] == 0:
        return _BlockTridiagFactor(
            single_pinv=single_pinv,
            endpoint_pinv=jnp.zeros((2 * n, 2 * n), dtype=dtype),
            levels=(),
        )

    def combine_lhs(A_l, C_l, B_l, A_r, C_r, B_r):
        S = 0.5 * (C_l + A_r + C_l.T + A_r.T)
        Sinv = jnp.linalg.pinv(
            S,
            hermitian=True,
            rtol=_BLOCK_TRIDIAG_PINV_RTOL,
        )
        Sinv_Blt = Sinv @ B_l.T
        Sinv_Br = Sinv @ B_r
        return (
            A_l - B_l @ Sinv_Blt,
            C_r - B_r.T @ Sinv_Br,
            -B_l @ Sinv_Br,
            Sinv,
            B_l,
            B_r,
        )

    A_level = 0.5 * diag[:-1]
    C_level = 0.5 * diag[1:]
    B_level = upper
    levels = []

    while A_level.shape[0] > 1:
        interval_count = A_level.shape[0]
        pair_count = interval_count // 2
        carry = interval_count % 2 == 1

        combined = jax.vmap(combine_lhs)(
            A_level[: 2 * pair_count : 2],
            C_level[: 2 * pair_count : 2],
            B_level[: 2 * pair_count : 2],
            A_level[1 : 2 * pair_count : 2],
            C_level[1 : 2 * pair_count : 2],
            B_level[1 : 2 * pair_count : 2],
        )
        (
            A_next,
            C_next,
            B_next,
            Sinv_level,
            B_left_level,
            B_right_level,
        ) = combined

        if carry:
            A_next = jnp.concatenate([A_next, A_level[-1:]], axis=0)
            C_next = jnp.concatenate([C_next, C_level[-1:]], axis=0)
            B_next = jnp.concatenate([B_next, B_level[-1:]], axis=0)

        levels.append(
            _BlockTridiagLevel(
                Sinv=Sinv_level,
                B_left=B_left_level,
                B_right=B_right_level,
            )
        )
        A_level, C_level, B_level = (
            A_next,
            C_next,
            B_next,
        )

    K = jnp.zeros((2 * n, 2 * n), dtype=dtype)
    K = K.at[:n, :n].set(A_level[0] + 0.5 * diag[0])
    K = K.at[:n, n:].set(B_level[0])
    K = K.at[n:, :n].set(B_level[0].T)
    K = K.at[n:, n:].set(C_level[0] + 0.5 * diag[-1])
    K = 0.5 * (K + K.T)
    endpoint_pinv = jnp.linalg.pinv(
        K,
        hermitian=True,
        rtol=_BLOCK_TRIDIAG_PINV_RTOL,
    )
    return _BlockTridiagFactor(
        single_pinv=single_pinv,
        endpoint_pinv=endpoint_pinv,
        levels=tuple(levels),
    )


def _block_tridiag_solve_from_factor(factor, rhs):
    """Apply a cyclic-reduction factorization to a block RHS."""
    if rhs.shape[0] == 1:
        return (factor.single_pinv @ rhs[0])[None]

    n = rhs.shape[-1]
    dtype = rhs.dtype
    ra_level = 0.5 * rhs[:-1]
    rc_level = 0.5 * rhs[1:]
    rhs_levels = []

    for level in factor.levels:
        interval_count = ra_level.shape[0]
        pair_count = level.Sinv.shape[0]
        carry = interval_count % 2 == 1
        rb_level = (
            rc_level[: 2 * pair_count : 2]
            + ra_level[1 : 2 * pair_count : 2]
        )
        Sinv_rb = jnp.einsum("...ij,...j->...i", level.Sinv, rb_level)
        ra_next = (
            ra_level[: 2 * pair_count : 2]
            - jnp.einsum("...ij,...j->...i", level.B_left, Sinv_rb)
        )
        rc_next = (
            rc_level[1 : 2 * pair_count : 2]
            - jnp.einsum(
                "...ji,...j->...i",
                level.B_right,
                Sinv_rb,
            )
        )
        if carry:
            ra_next = jnp.concatenate([ra_next, ra_level[-1:]], axis=0)
            rc_next = jnp.concatenate([rc_next, rc_level[-1:]], axis=0)

        rhs_levels.append(rb_level)
        ra_level, rc_level = ra_next, rc_next

    boundary_rhs = jnp.concatenate(
        [ra_level[0] + 0.5 * rhs[0], rc_level[0] + 0.5 * rhs[-1]]
    )
    endpoints = factor.endpoint_pinv @ boundary_rhs
    interval_values = endpoints.reshape(1, 2, n)

    def recover_one(Sinv, rb, B_l, B_r, y_l, y_r):
        return Sinv @ (rb - B_l.T @ y_l - B_r @ y_r)

    for level, rb_level in zip(factor.levels[::-1], rhs_levels[::-1]):
        parent_values = interval_values
        pair_count = level.Sinv.shape[0]
        carry = parent_values.shape[0] != pair_count
        interval_count = 2 * pair_count + int(carry)
        paired_values = parent_values[:pair_count]
        y_l = paired_values[:, 0]
        y_r = paired_values[:, 1]
        y_mid = jax.vmap(recover_one)(
            level.Sinv,
            rb_level,
            level.B_left,
            level.B_right,
            y_l,
            y_r,
        )

        child_values = jnp.zeros((interval_count, 2, n), dtype=dtype)
        child_values = child_values.at[: 2 * pair_count : 2, 0].set(y_l)
        child_values = child_values.at[: 2 * pair_count : 2, 1].set(y_mid)
        child_values = child_values.at[1 : 2 * pair_count : 2, 0].set(y_mid)
        child_values = child_values.at[1 : 2 * pair_count : 2, 1].set(y_r)
        if carry:
            child_values = child_values.at[-1].set(parent_values[-1])
        interval_values = child_values

    return jnp.concatenate(
        [interval_values[:1, 0], interval_values[:, 1]], axis=0
    )


def _block_tridiag_solve(diag, upper, rhs):
    """Solve a symmetric block-tridiagonal system by cyclic reduction."""
    return _block_tridiag_solve_from_factor(
        _block_tridiag_factor(diag, upper),
        rhs,
    )


def _dual_recovery_cache(inputs):
    """Build RHS-independent data for structured dual recovery."""
    A, B, D, E, Delta, Sigma = (
        inputs.A,
        inputs.B,
        inputs.D,
        inputs.E,
        inputs.Delta,
        inputs.Sigma,
    )
    N, n = A.shape[0], A.shape[1]
    m, p = B.shape[2], D.shape[1]
    dtype = jnp.result_type(A, B, D, E, Delta, Sigma)
    I_n = jnp.eye(n, dtype=dtype)

    def stage_y_normal_lhs(Ak, Bk, Dk, Ek, Sigmak):
        zeros_pn = jnp.zeros((p, n), dtype=dtype)
        S = jnp.concatenate([Dk.T, Ek.T, -Sigmak], axis=0)
        S_pinv = jnp.linalg.pinv(S)
        R0 = jnp.concatenate(
            [-I_n, jnp.zeros((m, n), dtype=dtype), zeros_pn], axis=0
        )
        R1 = jnp.concatenate([Ak.T, Bk.T, zeros_pn], axis=0)

        def project_matrix(V):
            return V - S @ (S_pinv @ V)

        C0 = project_matrix(R0)
        C1 = project_matrix(R1)
        return (
            C0.T @ C0,
            C0.T @ C1,
            C1.T @ C1,
            C0,
            C1,
            S_pinv,
        )

    H00, H01, H11, C0, C1, S_pinv = jax.vmap(stage_y_normal_lhs)(
        A, B, D[:N], E, Sigma[:N]
    )

    Hdiag = jnp.zeros((N + 1, n, n), dtype=dtype)
    Hupper = H01
    Hdiag = Hdiag.at[:N].add(H00)
    Hdiag = Hdiag.at[1:].add(H11)

    Hdyn = jax.vmap(lambda Ck: Ck.T @ Ck)(-Delta)
    Hdiag = Hdiag + Hdyn

    def terminal_y_normal_lhs(DN, SigmaN):
        S = jnp.concatenate([DN.T, -SigmaN], axis=0)
        S_pinv = jnp.linalg.pinv(S)
        R = jnp.concatenate([-I_n, jnp.zeros((p, n), dtype=dtype)], axis=0)
        C = R - S @ (S_pinv @ R)
        return C.T @ C, C, S_pinv

    HN, C_terminal, S_pinv_terminal = terminal_y_normal_lhs(D[N], Sigma[N])
    Hdiag = Hdiag.at[N].add(HN)

    return _DualRecoveryCache(
        block_factor=_block_tridiag_factor(Hdiag, Hupper),
        C0=C0,
        C1=C1,
        C_terminal=C_terminal,
        S_pinv=S_pinv,
        S_pinv_terminal=S_pinv_terminal,
    )


def _recover_duals_from_cache(inputs, solve_inputs, x, u, cache):
    """Apply cached structured dual recovery to one RHS/primal trajectory."""
    A, B, Q, M, R, D, E, Delta = (
        inputs.A,
        inputs.B,
        inputs.Q,
        inputs.M,
        inputs.R,
        inputs.D,
        inputs.E,
        inputs.Delta,
    )
    q, r, c, d = solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d
    N, n = A.shape[0], A.shape[1]

    dtype = jnp.result_type(A, B, Q, M, R, D, E, q, r, c, d)

    bx = -jax.vmap(lambda Qk, xk, Mk, uk, qk: Qk @ xk + Mk @ uk + qk)(
        Q[:N], x[:N], M, u, q[:N]
    )
    bu = -jax.vmap(lambda Mk, xk, Rk, uk, rk: Mk.T @ xk + Rk @ uk + rk)(
        M, x[:N], R, u, r
    )
    bdyn = jax.vmap(lambda xk1, Ak, xk, Bk, uk, ck1: xk1 - Ak @ xk - Bk @ uk - ck1)(
        x[1:], A, x[:N], B, u, c[1:]
    )
    bcon = jax.vmap(lambda dk, Dk, xk, Ek, uk: dk - Dk @ xk - Ek @ uk)(
        d[:N], D[:N], x[:N], E, u
    )

    stage_rhs = jnp.concatenate([bx, bu, bcon], axis=1)
    h0 = jnp.einsum("...ji,...j->...i", cache.C0, stage_rhs)
    h1 = jnp.einsum("...ji,...j->...i", cache.C1, stage_rhs)

    hrhs = jnp.zeros((N + 1, n), dtype=dtype)
    hrhs = hrhs.at[:N].add(h0)
    hrhs = hrhs.at[1:].add(h1)

    dyn_rhs = jnp.concatenate([(x[0] - c[0])[None], bdyn])
    hdyn = jnp.einsum("...ji,...j->...i", -Delta, dyn_rhs)
    hrhs = hrhs + hdyn

    bx_terminal = -(Q[N] @ x[N] + q[N])
    bcon_terminal = d[N] - D[N] @ x[N]
    terminal_rhs = jnp.concatenate([bx_terminal, bcon_terminal])
    hN = cache.C_terminal.T @ terminal_rhs
    hrhs = hrhs.at[N].add(hN)

    y = _block_tridiag_solve_from_factor(cache.block_factor, hrhs)

    def recover_stage_lambda(
        Ak, Bk, S_pinv_k, bxk, buk, bconk, yk, yk1
    ):
        rhs = jnp.concatenate(
            [bxk + yk - Ak.T @ yk1, buk - Bk.T @ yk1, bconk]
        )
        return S_pinv_k @ rhs

    lam = jax.vmap(recover_stage_lambda)(
        A, B, cache.S_pinv, bx, bu, bcon, y[:N], y[1:]
    )
    lam_terminal = cache.S_pinv_terminal @ jnp.concatenate(
        [bx_terminal + y[N], bcon_terminal]
    )
    return y, jnp.concatenate([lam, lam_terminal[None]])


def _recover_duals(inputs, solve_inputs, x, u):
    """Recover multipliers through a structured dual-only solve.

    Given the primal trajectory ``(x, u)``, the remaining KKT equations are
    linear in the dynamics multipliers ``y`` and equality multipliers ``lambda``.
    Each stage only couples ``(y_k, y_{k+1}, lambda_k)``.  We eliminate
    ``lambda_k`` locally with a pseudoinverse/projector, assemble the resulting
    block-tridiagonal normal equations in ``y_0, ..., y_N``, solve them with
    balanced endpoint condensing, and then recover each ``lambda_k`` locally.
    This avoids horizon-sized dense pseudoinverses and keeps the costate solve
    in a logarithmic-depth reduction/recovery pattern.
    """
    return _recover_duals_from_cache(
        inputs,
        solve_inputs,
        x,
        u,
        _dual_recovery_cache(inputs),
    )


def _recover_or_use_forward_duals(inputs, solve_inputs, x, u, y_forward, cache=None):
    if inputs.D.shape[1] == 0:
        lam = jnp.zeros(inputs.D.shape[:2], dtype=x.dtype)
        return y_forward, lam
    if cache is None:
        cache = _dual_recovery_cache(inputs)
    return _recover_duals_from_cache(inputs, solve_inputs, x, u, cache)


def _compress_relation(E, rhs, rows):
    """Compress ``E v = rhs`` to a fixed number of equivalent row-space rows."""
    U, S, Vh = jnp.linalg.svd(E, full_matrices=True)
    source_rows = E.shape[-2]
    cols = E.shape[-1]
    keep = min(rows, source_rows, cols)
    H = Vh[..., :keep, :]
    Ut_rhs = jnp.einsum("...ji,...j->...i", U[..., :, :keep], rhs)
    tol = 100 * jnp.finfo(E.dtype).eps * jnp.maximum(source_rows, cols)
    nonzero = S[..., :keep] > tol
    H = jnp.where(nonzero[..., None], H, 0.0)
    h = jnp.where(nonzero, Ut_rhs / S[..., :keep], 0.0)
    if keep < rows:
        pad_shape = E.shape[:-2] + (rows - keep, E.shape[-1])
        H = jnp.concatenate([H, jnp.zeros(pad_shape, dtype=E.dtype)], axis=-2)
        h = jnp.concatenate(
            [h, jnp.zeros(E.shape[:-2] + (rows - keep,), dtype=E.dtype)],
            axis=-1,
        )
    return H, h


def _state_only_domain_base(inputs, solve_inputs):
    """Base endpoint-domain relations for hard state-only constraints.

    Each interval relation has fixed shape ``2n``:

        H_left[k] x_k + H_right[k] x_{k+1} = h[k].

    The rows encode the current state constraint plus the one-step reachability
    condition after eliminating ``u_k``.
    """
    A, B, D = inputs.A, inputs.B, inputs.D
    c, d = solve_inputs.c, solve_inputs.d
    N, n = A.shape[0], A.shape[1]
    p = D.shape[1]
    dtype = jnp.result_type(A, B, D, c, d)

    def one(Ak, Bk, Dk, ck1, dk):
        PB = jnp.eye(n, dtype=dtype) - Bk @ jnp.linalg.pinv(Bk)
        pad_rows = n - p
        zero_pn = jnp.zeros((p, n), dtype=dtype)
        zero_pad_n = jnp.zeros((pad_rows, n), dtype=dtype)
        zero_pad = jnp.zeros((pad_rows,), dtype=dtype)
        H_left = jnp.concatenate([Dk, -PB @ Ak, zero_pad_n], axis=0)
        H_right = jnp.concatenate([zero_pn, PB, zero_pad_n], axis=0)
        h = jnp.concatenate([dk, PB @ ck1, zero_pad], axis=0)
        return H_left, H_right, h

    return jax.vmap(one)(A, B, D[:N], c[1:], d[:N])


def _state_only_domain_combine(left, right):
    """Associatively combine adjacent endpoint-domain relations."""
    L_i, L_z, h_l = left
    R_z, R_j, h_r = right
    n = L_i.shape[-1]
    rows = L_i.shape[-2]
    dtype = L_i.dtype
    zero = jnp.zeros_like(L_i)

    Z = jnp.concatenate([L_z, R_z], axis=-2)
    endpoint = jnp.concatenate(
        [
            jnp.concatenate([L_i, zero], axis=-1),
            jnp.concatenate([zero, R_j], axis=-1),
        ],
        axis=-2,
    )
    rhs = jnp.concatenate([h_l, h_r], axis=-1)

    projector = jnp.eye(2 * rows, dtype=dtype) - Z @ jnp.linalg.pinv(Z)
    projected_rhs = jnp.einsum("...ij,...j->...i", projector, rhs)
    compressed, h = _compress_relation(projector @ endpoint, projected_rhs, rows)
    return compressed[..., :n], compressed[..., n:], h


def _state_only_domain_fold_terminal(interval, D_N, d_N):
    """Fold terminal state constraints into an endpoint-domain relation."""
    H_left, H_terminal, h_interval = interval
    n = H_left.shape[-1]
    rows = H_left.shape[-2]
    p = D_N.shape[-2]
    dtype = H_left.dtype

    Z = jnp.concatenate([H_terminal, D_N], axis=-2)
    endpoint = jnp.concatenate(
        [H_left, jnp.zeros((p, n), dtype=dtype)],
        axis=-2,
    )
    rhs = jnp.concatenate([h_interval, d_N], axis=-1)
    projector = jnp.eye(rows + p, dtype=dtype) - Z @ jnp.linalg.pinv(Z)
    projected_rhs = jnp.einsum("...ij,...j->...i", projector, rhs)
    return _compress_relation(projector @ endpoint, projected_rhs, n)


def _state_only_suffix_domains_parallel(inputs, solve_inputs):
    """Compute hard state-only suffix feasibility domains in O(log N) depth.

    Returns ``(H, h)`` with ``H[k] x_k = h[k]`` describing the exact endpoint
    domain for suffix ``[k, N]`` up to row-space basis choices.
    """
    A = inputs.A
    N, n = A.shape[0], A.shape[1]
    base = _state_only_domain_base(inputs, solve_inputs)
    rev = jax.tree.map(lambda x: x[::-1], base)
    scanned_rev = jax.lax.associative_scan(
        lambda acc, new: _state_only_domain_combine(new, acc), rev
    )
    scanned = jax.tree.map(lambda x: x[::-1], scanned_rev)
    H, h = jax.vmap(
        lambda interval: _state_only_domain_fold_terminal(
            interval, inputs.D[N], solve_inputs.d[N]
        )
    )(scanned)
    HN, hN = _compress_relation(inputs.D[N], solve_inputs.d[N], n)
    return jnp.concatenate([H, HN[None]], axis=0), jnp.concatenate(
        [h, hN[None]], axis=0
    )


def _use_hard_state_only_nullspace_path(inputs):
    p = inputs.D.shape[1]
    n = inputs.A.shape[1]
    if not (0 < p < n):
        return False
    singular_values = jnp.linalg.svd(inputs.D[1:], compute_uv=False)
    full_row_rank = jnp.all(
        singular_values[..., -1]
        > 100 * jnp.finfo(inputs.D.dtype).eps * jnp.maximum(n, p)
    )
    return (
        jnp.all(inputs.E == 0)
        & jnp.all(inputs.Delta == 0)
        & jnp.all(inputs.Sigma == 0)
        & full_row_rank
    )


def _use_hard_state_only_projected_path(inputs):
    p = inputs.D.shape[1]
    n = inputs.A.shape[1]
    if not (0 < p < n):
        return False
    singular_values = jnp.linalg.svd(inputs.D[1:], compute_uv=False)
    full_row_rank = jnp.all(
        singular_values[..., -1]
        > 100 * jnp.finfo(inputs.D.dtype).eps * jnp.maximum(n, p)
    )
    return (
        jnp.all(inputs.E == 0)
        & jnp.all(inputs.Delta == 0)
        & jnp.all(inputs.Sigma == 0)
        & ~full_row_rank
    )


def _hard_state_only_nullspace_solve(inputs, solve_inputs, parallel: bool):
    """Solve hard state-only constraints by reducing to nullspace coordinates.

    For ``E = 0`` and unregularized hard constraints, each constrained state is
    written as ``x_k = xbar_k + Z_k z_k``.  Feasibility of the next state becomes
    a stagewise control-coupled constraint in the reduced coordinates, so the
    existing scan solver can be used without relying on singular local IVF
    pseudo-inverses to represent the hard domain.
    """
    A, B, Q, M, R, D = inputs.A, inputs.B, inputs.Q, inputs.M, inputs.R, inputs.D
    q, r, c, d = solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d
    N, n = A.shape[0], A.shape[1]
    p = D.shape[1]
    zdim = n - p
    dtype = jnp.result_type(A, B, Q, M, R, D, q, r, c, d)

    def affine_state_subspace(Dk, dk):
        _, _, vh = jnp.linalg.svd(Dk, full_matrices=True)
        Zk = jnp.swapaxes(vh[p:], -2, -1)
        Wk = jnp.swapaxes(vh[:p], -2, -1)
        xbar_k = jnp.linalg.pinv(Dk) @ dk
        return Zk, Wk, xbar_k

    Z, W, xbar = jax.vmap(affine_state_subspace)(D, d)
    Z = Z.at[0].set(jnp.zeros((n, zdim), dtype=dtype))
    W = W.at[0].set(jnp.zeros((n, p), dtype=dtype))
    xbar = xbar.at[0].set(c[0])

    Zk = Z[:N]
    Zkp1 = Z[1:]
    Wkp1 = W[1:]
    xbar_k = xbar[:N]
    xbar_kp1 = xbar[1:]

    Qz = jax.vmap(lambda Zj, Qj: Zj.T @ Qj @ Zj)(Z, Q)
    qz = jax.vmap(lambda Zj, Qj, qj, xj: Zj.T @ (Qj @ xj + qj))(
        Z, Q, q, xbar
    )
    Mz = jax.vmap(lambda Zj, Mj: Zj.T @ Mj)(Zk, M)
    rz = jax.vmap(lambda Mj, xj, rj: Mj.T @ xj + rj)(M, xbar_k, r)

    transition_offset = jax.vmap(lambda Aj, xj, cj, xj1: Aj @ xj + cj - xj1)(
        A, xbar_k, c[1:], xbar_kp1
    )
    Az = jax.vmap(lambda Zj1, Aj, Zj: Zj1.T @ Aj @ Zj)(Zkp1, A, Zk)
    Bz = jax.vmap(lambda Zj1, Bj: Zj1.T @ Bj)(Zkp1, B)
    cz = jax.vmap(lambda Zj1, off: Zj1.T @ off)(Zkp1, transition_offset)

    Dz_stage = jax.vmap(lambda Wj1, Aj, Zj: Wj1.T @ Aj @ Zj)(Wkp1, A, Zk)
    Ez = jax.vmap(lambda Wj1, Bj: Wj1.T @ Bj)(Wkp1, B)
    dz_stage = -jax.vmap(lambda Wj1, off: Wj1.T @ off)(
        Wkp1, transition_offset
    )
    Dz = jnp.concatenate(
        [Dz_stage, jnp.zeros((1, p, zdim), dtype=dtype)], axis=0
    )
    dz = jnp.concatenate([dz_stage, jnp.zeros((1, p), dtype=dtype)], axis=0)

    reduced_inputs = FactorizationInputs(
        A=Az,
        B=Bz,
        Q=Qz,
        M=Mz,
        R=R,
        D=Dz,
        E=Ez,
        Delta=jnp.zeros((N + 1, zdim, zdim), dtype=dtype),
        Sigma=jnp.zeros((N + 1, p, p), dtype=dtype),
    )
    reduced_solve_inputs = SolveInputs(
        q=qz,
        r=rz,
        c=jnp.concatenate(
            [jnp.zeros((1, zdim), dtype=dtype), cz],
            axis=0,
        ),
        d=dz,
    )

    if parallel:
        cog = _cost_to_go_parallel(reduced_inputs, reduced_solve_inputs)
        z, u, _, _, _ = _forward_primal_parallel(
            reduced_inputs, reduced_solve_inputs, cog
        )
    else:
        cog = _cost_to_go_sequential(reduced_inputs, reduced_solve_inputs)
        z, u, _, _, _ = _forward_primal_sequential(
            reduced_inputs, reduced_solve_inputs, cog
        )

    x = jax.vmap(lambda Zj, zj, xj: Zj @ zj + xj)(Z, z, xbar)
    y, lam = _recover_duals(inputs, solve_inputs, x, u)
    return SolveOutputs(
        X=x,
        U=u,
        Y=y,
        Lam=lam,
        p=jnp.zeros_like(q),
        k=jnp.zeros_like(r),
    )


def _hard_state_only_projected_solve(inputs, solve_inputs, parallel: bool):
    """Solve hard state-only constraints using scanned suffix domains.

    This path handles rank-deficient ``D`` rows.  It first computes the exact
    suffix feasibility domain ``H_k x_k = h_k`` with an associative endpoint
    relation scan, then writes ``x_k = xbar_k + P_k z_k`` where ``P_k`` projects
    onto ``null(H_k)``.  Dynamics feasibility outside the next suffix domain is
    added as a control-coupled stage constraint in the transformed problem.
    """
    A, B, Q, M, R, D = inputs.A, inputs.B, inputs.Q, inputs.M, inputs.R, inputs.D
    q, r, c, d = solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d
    N, n = A.shape[0], A.shape[1]
    dtype = jnp.result_type(A, B, Q, M, R, D, q, r, c, d)

    H, h = _state_only_suffix_domains_parallel(inputs, solve_inputs)
    Hpinv = jax.vmap(jnp.linalg.pinv)(H)
    xbar = jax.vmap(lambda Hp, hk: Hp @ hk)(Hpinv, h)
    P = jnp.eye(n, dtype=dtype)[None] - Hpinv @ H
    P = P.at[0].set(jnp.zeros((n, n), dtype=dtype))
    xbar = xbar.at[0].set(c[0])

    Pk = P[:N]
    Pnext = P[1:]
    xbar_k = xbar[:N]
    xbar_next = xbar[1:]

    Qz = jax.vmap(lambda Pj, Qj: Pj.T @ Qj @ Pj)(P, Q)
    qz = jax.vmap(lambda Pj, Qj, qj, xj: Pj.T @ (Qj @ xj + qj))(
        P, Q, q, xbar
    )
    Mz = jax.vmap(lambda Pj, Mj: Pj.T @ Mj)(Pk, M)
    rz = jax.vmap(lambda Mj, xj, rj: Mj.T @ xj + rj)(M, xbar_k, r)

    offset = jax.vmap(lambda Aj, xj, cj, xj1: Aj @ xj + cj - xj1)(
        A, xbar_k, c[1:], xbar_next
    )
    Az = jax.vmap(lambda Pj1, Aj, Pj: Pj1 @ (Aj @ Pj))(Pnext, A, Pk)
    Bz = jax.vmap(lambda Pj1, Bj: Pj1 @ Bj)(Pnext, B)
    cz = jax.vmap(lambda Pj1, off: Pj1 @ off)(Pnext, offset)

    complement = jnp.eye(n, dtype=dtype)[None] - Pnext
    Dz_stage = jax.vmap(lambda Cj, Aj, Pj: Cj @ (Aj @ Pj))(
        complement, A, Pk
    )
    Ez = jax.vmap(lambda Cj, Bj: Cj @ Bj)(complement, B)
    dz_stage = -jax.vmap(lambda Cj, off: Cj @ off)(complement, offset)
    Dz = jnp.concatenate(
        [Dz_stage, jnp.zeros((1, n, n), dtype=dtype)], axis=0
    )
    dz = jnp.concatenate([dz_stage, jnp.zeros((1, n), dtype=dtype)], axis=0)

    reduced_inputs = FactorizationInputs(
        A=Az,
        B=Bz,
        Q=Qz,
        M=Mz,
        R=R,
        D=Dz,
        E=Ez,
        Delta=jnp.zeros((N + 1, n, n), dtype=dtype),
        Sigma=jnp.zeros((N + 1, n, n), dtype=dtype),
    )
    reduced_solve_inputs = SolveInputs(
        q=qz,
        r=rz,
        c=jnp.concatenate(
            [jnp.zeros((1, n), dtype=dtype), cz],
            axis=0,
        ),
        d=dz,
    )

    if parallel:
        cog = _cost_to_go_parallel(reduced_inputs, reduced_solve_inputs)
        z, u, _, _, _ = _forward_primal_parallel(
            reduced_inputs, reduced_solve_inputs, cog
        )
    else:
        cog = _cost_to_go_sequential(reduced_inputs, reduced_solve_inputs)
        z, u, _, _, _ = _forward_primal_sequential(
            reduced_inputs, reduced_solve_inputs, cog
        )

    x = jax.vmap(lambda Pj, zj, xj: Pj @ zj + xj)(P, z, xbar)
    y, lam = _recover_duals(inputs, solve_inputs, x, u)
    return SolveOutputs(
        X=x,
        U=u,
        Y=y,
        Lam=lam,
        p=jnp.zeros_like(q),
        k=jnp.zeros_like(r),
    )


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
        d=jnp.zeros((N + 1, p), dtype=dtype),
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
    return output_type(
        P=P,
        F=F,
        Cll=Cll,
        K=K,
        Phi=Phi,
        dual_recovery=_dual_recovery_cache(inputs),
    )


@jax.jit
def factor(inputs: FactorizationInputs) -> SequentialFactorizationOutputs:
    """Factor the constrained LQR LHS with a sequential backward pass.

    The returned data is independent of ``SolveInputs``.  It is currently a
    partial LHS cache: public solve paths still recompute RHS-dependent affine
    recurrences from ``SolveInputs``.
    """
    cog = _cost_to_go_sequential(inputs, _zero_solve_inputs(inputs))
    return _factorization_from_cog(inputs, cog, SequentialFactorizationOutputs)


@jax.jit
def factor_parallel(inputs: FactorizationInputs) -> ParallelFactorizationOutputs:
    """Factor the constrained LQR LHS with the parallel backward pass.

    As with :func:`factor`, this is a partial LHS cache; solve-time affine
    recurrences are still recomputed from the RHS.
    """
    cog = _cost_to_go_parallel(inputs, _zero_solve_inputs(inputs))
    return _factorization_from_cog(inputs, cog, ParallelFactorizationOutputs)




@jax.jit
def solve(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: SequentialFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using a sequential factorization."""
    p = factorization_inputs.D.shape[1]
    n = factorization_inputs.A.shape[1]
    if not (0 < p < n):
        return _solve_generic(
            factorization_inputs,
            factorization_outputs,
            solve_inputs,
        )
    return jax.lax.cond(
        _use_hard_state_only_nullspace_path(factorization_inputs),
        lambda _: _hard_state_only_nullspace_solve(
            factorization_inputs, solve_inputs, parallel=False
        ),
        lambda _: jax.lax.cond(
            _use_hard_state_only_projected_path(factorization_inputs),
            lambda __: _hard_state_only_projected_solve(
                factorization_inputs, solve_inputs, parallel=False
            ),
            lambda __: _solve_generic(
                factorization_inputs,
                factorization_outputs,
                solve_inputs,
            ),
            operand=None,
        ),
        operand=None,
    )


def _solve_generic(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: SequentialFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    cog = _cost_to_go_sequential(factorization_inputs, solve_inputs)
    x, u, y_forward, _, k = _forward_primal_sequential(
        factorization_inputs, solve_inputs, cog
    )
    y, lam = _recover_or_use_forward_duals(
        factorization_inputs,
        solve_inputs,
        x,
        u,
        y_forward,
        factorization_outputs.dual_recovery,
    )
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def solve_parallel(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: ParallelFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a constrained LQR RHS using the regularized KKT equations."""
    p = factorization_inputs.D.shape[1]
    n = factorization_inputs.A.shape[1]
    if not (0 < p < n):
        return _solve_parallel_generic(
            factorization_inputs,
            factorization_outputs,
            solve_inputs,
        )
    return jax.lax.cond(
        _use_hard_state_only_nullspace_path(factorization_inputs),
        lambda _: _hard_state_only_nullspace_solve(
            factorization_inputs, solve_inputs, parallel=True
        ),
        lambda _: jax.lax.cond(
            _use_hard_state_only_projected_path(factorization_inputs),
            lambda __: _hard_state_only_projected_solve(
                factorization_inputs, solve_inputs, parallel=True
            ),
            lambda __: _solve_parallel_generic(
                factorization_inputs,
                factorization_outputs,
                solve_inputs,
            ),
            operand=None,
        ),
        operand=None,
    )


def _solve_parallel_generic(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: ParallelFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    cog = _cost_to_go_parallel(factorization_inputs, solve_inputs)
    x, u, y_forward, _, k = _forward_primal_parallel(
        factorization_inputs, solve_inputs, cog
    )
    y, lam = _recover_or_use_forward_duals(
        factorization_inputs,
        solve_inputs,
        x,
        u,
        y_forward,
        factorization_outputs.dual_recovery,
    )
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def factor_and_solve(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Sequential factor + solve; return the KKT point."""
    p = factorization_inputs.D.shape[1]
    n = factorization_inputs.A.shape[1]
    if not (0 < p < n):
        return _factor_and_solve_generic(factorization_inputs, solve_inputs)
    return jax.lax.cond(
        _use_hard_state_only_nullspace_path(factorization_inputs),
        lambda _: _hard_state_only_nullspace_solve(
            factorization_inputs, solve_inputs, parallel=False
        ),
        lambda _: jax.lax.cond(
            _use_hard_state_only_projected_path(factorization_inputs),
            lambda __: _hard_state_only_projected_solve(
                factorization_inputs, solve_inputs, parallel=False
            ),
            lambda __: _factor_and_solve_generic(
                factorization_inputs, solve_inputs
            ),
            operand=None,
        ),
        operand=None,
    )


def _factor_and_solve_generic(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    cog = _cost_to_go_sequential(factorization_inputs, solve_inputs)
    x, u, y_forward, _, k = _forward_primal_sequential(
        factorization_inputs, solve_inputs, cog
    )
    y, lam = _recover_or_use_forward_duals(
        factorization_inputs, solve_inputs, x, u, y_forward
    )
    return SolveOutputs(X=x, U=u, Y=y, Lam=lam, p=cog[1], k=k)


@jax.jit
def factor_and_solve_parallel(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Parallel factor + solve; return the KKT point."""
    p = factorization_inputs.D.shape[1]
    n = factorization_inputs.A.shape[1]
    if not (0 < p < n):
        return _factor_and_solve_parallel_generic(
            factorization_inputs, solve_inputs
        )
    return jax.lax.cond(
        _use_hard_state_only_nullspace_path(factorization_inputs),
        lambda _: _hard_state_only_nullspace_solve(
            factorization_inputs, solve_inputs, parallel=True
        ),
        lambda _: jax.lax.cond(
            _use_hard_state_only_projected_path(factorization_inputs),
            lambda __: _hard_state_only_projected_solve(
                factorization_inputs, solve_inputs, parallel=True
            ),
            lambda __: _factor_and_solve_parallel_generic(
                factorization_inputs, solve_inputs
            ),
            operand=None,
        ),
        operand=None,
    )


def _factor_and_solve_parallel_generic(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    cog = _cost_to_go_parallel(factorization_inputs, solve_inputs)
    x, u, y_forward, _, k = _forward_primal_parallel(
        factorization_inputs, solve_inputs, cog
    )
    y, lam = _recover_or_use_forward_duals(
        factorization_inputs, solve_inputs, x, u, y_forward
    )
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
