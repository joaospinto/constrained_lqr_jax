"""Shared building blocks for the constrained LQR solver.

All functions here are pure and reusable by both the sequential and parallel
algorithms in :mod:`constrained_lqr_jax.solver`.

The package supports optional dual regularization: dynamics use blocks
``Delta_k`` and equality constraints use blocks ``Sigma_k``. Setting these
blocks to zero recovers exact dynamics and exact equality constraints.

The parallel backward pass is built on the *mixed-constraint interval value
function* (IVF) — a uniform, fixed-dimension representation of the cost-to-go of
an interval ``[i, j)`` that **carries the constraint** instead of eliminating
its multiplier at the base case.  It supports the generic scan path for mixed
constraints, regularized constraints, and unconstrained problems.  Hard
state-only cases with more constraints than controls use an endpoint-domain scan
before the mixed-IVF solve so suffix feasibility domains are represented
explicitly.

A mixed IVF over an interval with entry state ``x`` and exit state ``x'`` is the
9-component object ``(P, p, A, C, c, Cyl, Cll, F, g)`` representing

    V(x, x') = max_{y, λ} [ ½ xᵀP x + pᵀx
                            - ½ yᵀC y + yᵀ(A x + c - x')
                            - yᵀCyl λ
                            + λᵀ(F x - g) - ½ λᵀCll λ ]

with shapes ``P,A,C ∈ ℝ^{n×n}``, ``p,c ∈ ℝ^n``, ``Cyl ∈ ℝ^{n×p}``,
``Cll ∈ ℝ^{p×p}``, ``F ∈ ℝ^{p×n}``, ``g ∈ ℝ^p``.  Here ``y`` is the (carried)
dynamics multiplier and ``λ`` the (carried) constraint multiplier of the
interval.  All combine / fold operations below are written to broadcast over an
arbitrary leading batch dimension so they can be used inside
``jax.lax.associative_scan``.
"""

import jax
import jax.numpy as jnp

from constrained_lqr_jax.types import (
    FactorizationInputs,
    SolveInputs,
    SolveOutputs,
)


@jax.jit
def symmetrize(X: jnp.ndarray) -> jnp.ndarray:
    """Force exact symmetry: (X + Xᵀ) / 2.  Works on (..., n, n) arrays."""
    return 0.5 * (X + jnp.swapaxes(X, -2, -1))


# ═══════════════════════════════════════════════════════════════════════════
# Mixed-constraint interval value function (IVF)
# ═══════════════════════════════════════════════════════════════════════════


def mixed_ivf_base(inputs: FactorizationInputs, solve_inputs: SolveInputs):
    """Per-stage base mixed IVFs (one single-stage interval ``[k, k+1)`` each).

    Built by eliminating only the control ``u_k`` (via ``R_k⁻¹``); the
    constraint multiplier ``λ_k`` is *not* eliminated — it is carried.  Returns
    a 9-tuple of arrays each stacked over ``k = 0, ..., N-1``.
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
    q, r, c, d = (solve_inputs.q, solve_inputs.r, solve_inputs.c, solve_inputs.d)
    N = A.shape[0]

    def one(Ak, Bk, Qk, Mk, Rk, Dk, Ek, Deltak1, Sigmak, qk, rk, ck1, dk):
        Rinv = jnp.linalg.inv(Rk)
        RiMt = Rinv @ Mk.T
        RiBt = Rinv @ Bk.T
        RiEt = Rinv @ Ek.T
        Rir = Rinv @ rk
        P = symmetrize(Qk - Mk @ RiMt)
        pv = qk - Mk @ Rir
        Ao = Ak - Bk @ RiMt
        Cc = symmetrize(Bk @ RiBt + Deltak1)
        co = ck1 - Bk @ Rir
        Cyl = Bk @ RiEt
        Cll = symmetrize(Ek @ RiEt + Sigmak)
        F = Dk - Ek @ RiMt
        g = dk + Ek @ Rir
        return P, pv, Ao, Cc, co, Cyl, Cll, F, g

    return jax.vmap(one)(
        A, B, Q[:N], M, R, D[:N], E, Delta[1:], Sigma[:N], q[:N], r, c[1:], d[:N]
    )


def _schur_eliminate(H, h, n_outer):
    """Schur-eliminate the trailing ``interior`` block of a quadratic form.

    Given ``½ vᵀ H v + hᵀ v`` with ``v = (o, w)`` and ``o`` the first
    ``n_outer`` coordinates, returns the reduced ``(H̃_oo, h̃_o)`` after
    optimising out ``w`` (saddle / partial optimisation).  Uses a pseudo-inverse
    so the result is well-defined even when the interior block is singular
    (e.g. carried-but-degenerate constraints).  Batched over leading axes.
    """
    tr = lambda X: jnp.swapaxes(X, -2, -1)
    o = n_outer
    Hoo = H[..., :o, :o]
    How = H[..., :o, o:]
    Hww = H[..., o:, o:]
    ho = h[..., :o]
    hw = h[..., o:]
    Hww_pinv = jnp.linalg.pinv(Hww)
    HowP = How @ Hww_pinv
    H_red = Hoo - HowP @ tr(How)
    h_red = ho - jnp.einsum("...ij,...j->...i", HowP, hw)
    return H_red, h_red


def _solve_mixed_combine_hww(C1, P2, F2, Cll2, B):
    """Solve the mixed-combine interior system without factoring y1.

    The generic combine eliminates ``w = (z, y1, lambda2)`` through the block

        [[P2, -I, F2.T],
         [-I, -C1, 0],
         [F2,  0, -Cll2]].

    Stationarity lets us eliminate ``y1`` algebraically and solve the smaller
    structured system in ``(z, lambda2)``:

        [[I + C1 P2, C1 F2.T],
         [F2,        -Cll2   ]].

    ``B`` is a stack of right-hand sides for the full ``w`` system.  The return
    value has the same shape as ``B`` and equals ``pinv(Hww) @ B`` when the
    transformed system is consistent.
    """
    n = C1.shape[-1]
    p = F2.shape[-2]
    batch = C1.shape[:-2]
    k = B.shape[-1]
    I = jnp.broadcast_to(jnp.eye(n, dtype=C1.dtype), batch + (n, n))

    bz = B[..., :n, :]
    by = B[..., n : 2 * n, :]
    bl = B[..., 2 * n :, :]

    K = jnp.zeros(batch + (n + p, n + p), dtype=C1.dtype)
    K = K.at[..., :n, :n].set(I + C1 @ P2)
    K = K.at[..., :n, n:].set(C1 @ jnp.swapaxes(F2, -2, -1))
    K = K.at[..., n:, :n].set(F2)
    K = K.at[..., n:, n:].set(-Cll2)

    rhs = jnp.concatenate([C1 @ bz - by, bl], axis=-2)
    if p == 0:
        zl = jnp.linalg.solve(K, rhs)
    else:
        zl = jnp.linalg.pinv(K) @ rhs
    z = zl[..., :n, :]
    lam = zl[..., n:, :]
    y = P2 @ z + jnp.swapaxes(F2, -2, -1) @ lam - bz
    return jnp.concatenate([z, y, lam], axis=-2)


def _solve_mixed_terminal_hww(C, QN, DN, SigmaN, B):
    """Solve the terminal-fold interior system through a smaller Schur block.

    The terminal fold eliminates ``w = (y, xN, nu)`` through

        [[-C, -I, 0],
         [-I, QN, DN.T],
         [ 0, DN, -SigmaN]].

    The first row gives ``xN = -C y - b_y``.  Substituting into the remaining
    rows leaves an ``(n+p)`` system in ``(y, nu)``, avoiding a larger pseudo-
    inverse for every folded suffix.
    """
    n = C.shape[-1]
    p = DN.shape[-2]
    batch = C.shape[:-2]
    dtype = C.dtype
    I = jnp.broadcast_to(jnp.eye(n, dtype=dtype), batch + (n, n))

    by = B[..., :n, :]
    bx = B[..., n : 2 * n, :]
    bnu = B[..., 2 * n :, :]

    def active_terminal(_):
        K = jnp.zeros(batch + (n + p, n + p), dtype=dtype)
        K = K.at[..., :n, :n].set(I + QN @ C)
        K = K.at[..., :n, n:].set(-jnp.swapaxes(DN, -2, -1))
        K = K.at[..., n:, :n].set(DN @ C)
        K = K.at[..., n:, n:].set(SigmaN)

        rhs = jnp.concatenate([-(bx + QN @ by), -(bnu + DN @ by)], axis=-2)
        ynu = jnp.linalg.pinv(K) @ rhs
        y = ynu[..., :n, :]
        nu = ynu[..., n:, :]
        xN = -(C @ y + by)
        return jnp.concatenate([y, xN, nu], axis=-2)

    def inactive_terminal(_):
        Ky = I + QN @ C
        y = jnp.linalg.solve(Ky, -(bx + QN @ by))
        xN = -(C @ y + by)
        nu = jnp.zeros(batch + (p, B.shape[-1]), dtype=dtype)
        return jnp.concatenate([y, xN, nu], axis=-2)

    return jax.lax.cond(
        jnp.all(DN == 0) & jnp.all(SigmaN == 0),
        inactive_terminal,
        active_terminal,
        operand=None,
    )


def _schur_eliminate_mixed_combine(H, h, n, p, C1, P2, F2, Cll2):
    """Schur-eliminate a mixed-IVF combination using the structured solve."""
    tr = lambda X: jnp.swapaxes(X, -2, -1)
    o = 3 * n + p
    Hoo = H[..., :o, :o]
    How = H[..., :o, o:]
    ho = h[..., :o]
    hw = h[..., o:]

    solved = _solve_mixed_combine_hww(
        C1, P2, F2, Cll2, jnp.concatenate([tr(How), hw[..., None]], axis=-1)
    )
    X = solved[..., :-1]
    xh = solved[..., -1]
    H_red = Hoo - How @ X
    h_red = ho - jnp.einsum("...ij,...j->...i", How, xh)
    return H_red, h_red


def mixed_ivf_combine(left, right, n, p):
    """Associative combination of two adjacent mixed IVFs.

    ``left`` covers ``[i, j)`` (entry ``xi``, exit ``z``) and ``right`` covers
    ``[j, k)`` (entry ``z``, exit ``xk``); the result covers ``[i, k)`` (entry
    ``xi``, exit ``xk``).  The intermediate state ``z`` and the *right* interval's
    dynamics dual / constraint multiplier are eliminated; the combined interval
    carries the right interval's dual (its ``-I`` exit coupling) and the left
    interval's constraint (acting on the persistent entry ``xi``).  Batched.
    """
    P1, p1, A1, C1, c1, Cyl1, Cll1, F1, g1 = left
    P2, p2, A2, C2, c2, Cyl2, Cll2, F2, g2 = right
    batch = P1.shape[:-2]
    dtype = P1.dtype
    tr = lambda X: jnp.swapaxes(X, -2, -1)

    # outer o = [xi(n), xk(n), y2(n), λ1(p)] ; interior w = [z(n), y1(n), λ2(p)]
    o = 3 * n + p
    xi = slice(0, n)
    y2 = slice(2 * n, 3 * n)
    l1 = slice(3 * n, 3 * n + p)
    z = slice(0, n)
    y1 = slice(n, 2 * n)
    l2 = slice(2 * n, 2 * n + p)

    zero_nn = jnp.zeros(batch + (n, n), dtype=dtype)
    zero_np = jnp.zeros(batch + (n, p), dtype=dtype)
    zero_pn = jnp.zeros(batch + (p, n), dtype=dtype)
    zero_pp = jnp.zeros(batch + (p, p), dtype=dtype)

    # Build H_wo and h_w directly.  The outer xk column is zero because xk only
    # couples to the carried y2 block, which is kept explicit in the IVF form.
    z_rhs = jnp.concatenate(
        [zero_nn, zero_nn, tr(A2), zero_np, p2[..., None]], axis=-1
    )
    y1_rhs = jnp.concatenate(
        [A1, zero_nn, zero_nn, -Cyl1, c1[..., None]], axis=-1
    )
    l2_rhs = jnp.concatenate(
        [zero_pn, zero_pn, -tr(Cyl2), zero_pp, -g2[..., None]], axis=-1
    )
    rhs = jnp.concatenate([z_rhs, y1_rhs, l2_rhs], axis=-2)
    solved = _solve_mixed_combine_hww(
        C1,
        P2,
        F2,
        Cll2,
        rhs,
    )
    X = solved[..., :-1]
    xh = solved[..., -1]
    X_z = X[..., z, :]
    X_y1 = X[..., y1, :]
    X_l2 = X[..., l2, :]
    xh_z = xh[..., z]
    xh_y1 = xh[..., y1]
    xh_l2 = xh[..., l2]

    schur_xi = tr(A1) @ X_y1
    schur_y2 = A2 @ X_z - Cyl2 @ X_l2
    schur_l1 = -tr(Cyl1) @ X_y1
    schurh_xi = jnp.einsum("...ij,...j->...i", tr(A1), xh_y1)
    schurh_y2 = jnp.einsum(
        "...ij,...j->...i", A2, xh_z
    ) - jnp.einsum("...ij,...j->...i", Cyl2, xh_l2)
    schurh_l1 = -jnp.einsum("...ij,...j->...i", tr(Cyl1), xh_y1)

    P = symmetrize(P1 - schur_xi[..., xi])
    A = -schur_y2[..., xi]
    C = symmetrize(C2 + schur_y2[..., y2])
    Cyl = schur_y2[..., l1]
    Cll = symmetrize(Cll1 + schur_l1[..., l1])
    F = F1 - schur_l1[..., xi]
    pv = p1 - schurh_xi
    cc = c2 - schurh_y2
    g = g1 + schurh_l1
    return P, pv, A, C, cc, Cyl, Cll, F, g


def mixed_ivf_fold_terminal(ivf, Q_N, q_N, D_N, Sigma_N, d_N, n, p):
    """Fold the terminal cost into an IVF ``[k, N)`` to get the cost-to-go.

    Given a mixed IVF with entry ``x_k`` and exit ``x_N``, plus the terminal
    cost ``½ x_Nᵀ Q_N x_N + q_Nᵀ x_N`` and terminal constraint
    ``D_N x_N = d_N``, optimise out the exit ``x_N``, dynamics dual ``y`` and
    terminal multiplier to obtain the constraint-carrying cost-to-go
    ``(P, p, F, Cll, g)`` of ``J_k`` as a function of ``x_k``.  Batched.
    """
    P, pv, A, C, c, Cyl, Cll, F, g = ivf
    batch = P.shape[:-2]
    dtype = P.dtype
    tr = lambda X: jnp.swapaxes(X, -2, -1)

    # outer o = [xk(n), λ(p)] ; interior w = [y(n), xN(n), ν(p)].
    # The only nonzero outer/interior coupling is in the y row, so assemble the
    # right-hand sides directly instead of scattering into Hoo/How.
    zero_no = jnp.zeros(batch + (n, n + p), dtype=dtype)
    zero_po = jnp.zeros(batch + (p, n + p), dtype=dtype)
    how = jnp.concatenate([A, -Cyl], axis=-1)
    B = jnp.concatenate(
        [
            jnp.concatenate([how, c[..., None]], axis=-1),
            jnp.concatenate([zero_no, q_N[..., None]], axis=-1),
            jnp.concatenate([zero_po, -d_N[..., None]], axis=-1),
        ],
        axis=-2,
    )
    solved = _solve_mixed_terminal_hww(
        C,
        Q_N,
        D_N,
        Sigma_N,
        B,
    )
    X = solved[..., :-1]
    xh = solved[..., -1]
    X_y = X[..., :n, :]
    xh_y = xh[..., :n]

    schur_H = tr(how) @ X_y
    schur_h = jnp.einsum("...ij,...j->...i", tr(how), xh_y)
    P_new = symmetrize(P - schur_H[..., :n, :n])
    F_new = F - schur_H[..., n:, :n]
    Cll_new = symmetrize(Cll + schur_H[..., n:, n:])
    p_new = pv - schur_h[..., :n]
    g_new = g + schur_h[..., n:]
    return P_new, p_new, F_new, Cll_new, g_new


# ═══════════════════════════════════════════════════════════════════════════
# KKT residual
# ═══════════════════════════════════════════════════════════════════════════


@jax.jit
def compute_residual(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
    solve_outputs: SolveOutputs,
) -> jnp.ndarray:
    """Flat KKT residual of the constrained LQR system (zero at the solution).

    The residual blocks (concatenated and flattened) are:
        stationarity in x:  Q x + M u + Aᵀ y₊ + Dᵀ λ + q - y,
        stationarity in u:  Mᵀ x + R u + Bᵀ y₊ + Eᵀ λ + r,
        terminal x:         Q_N x_N + q_N + D_Nᵀ λ_N - y_N,
        initial dynamics:   c₀ - x₀ - Delta₀ y₀,
        dynamics:           A x + B u + c₊ - x₊ - Delta₊ y₊,
        constraints:        D x + E u - d - Sigma λ,
        terminal constraint: D_N x_N - d_N - Sigma_N λ_N.
    """
    A = factorization_inputs.A
    B = factorization_inputs.B
    Q = factorization_inputs.Q
    M = factorization_inputs.M
    R = factorization_inputs.R
    D = factorization_inputs.D
    E = factorization_inputs.E
    Delta = factorization_inputs.Delta
    Sigma = factorization_inputs.Sigma

    q = solve_inputs.q
    r = solve_inputs.r
    c = solve_inputs.c
    d = solve_inputs.d

    X = solve_outputs.X
    U = solve_outputs.U
    Y = solve_outputs.Y
    Lam = solve_outputs.Lam

    N = A.shape[0]

    return jnp.concatenate(
        [
            jax.vmap(
                lambda Qk, Xk, Mk, Uk, Ak, Yk, Yk1, Dk, Lk, qk: Qk @ Xk
                + Mk @ Uk
                + Ak.T @ Yk1
                + Dk.T @ Lk
                + qk
                - Yk
            )(Q[:N], X[:N], M, U, A, Y[:N], Y[1:], D[:N], Lam[:N], q[:N]).flatten(),
            jax.vmap(
                lambda Mk, Xk, Rk, Uk, Bk, Yk1, Ek, Lk, rk: Mk.T @ Xk
                + Rk @ Uk
                + Bk.T @ Yk1
                + Ek.T @ Lk
                + rk
            )(M, X[:N], R, U, B, Y[1:], E, Lam[:N], r).flatten(),
            (Q[N] @ X[N] + q[N] + D[N].T @ Lam[N] - Y[N]),
            (c[0] - X[0] - Delta[0] @ Y[0]),
            jax.vmap(
                lambda Ak, Xk, Bk, Uk, ck1, Xk1, Deltak1, Yk1: Ak @ Xk
                + Bk @ Uk
                + ck1
                - Xk1
                - Deltak1 @ Yk1
            )(A, X[:N], B, U, c[1:], X[1:], Delta[1:], Y[1:]).flatten(),
            jax.vmap(lambda Dk, Xk, Ek, Uk, dk, Sigmak, Lk: Dk @ Xk + Ek @ Uk - dk - Sigmak @ Lk)(
                D[:N], X[:N], E, U, d[:N], Sigma[:N], Lam[:N]
            ).flatten(),
            (D[N] @ X[N] - d[N] - Sigma[N] @ Lam[N]),
        ]
    )
