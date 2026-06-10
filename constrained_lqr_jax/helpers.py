"""Shared building blocks for the constrained LQR solver.

All functions here are pure and reusable by both the sequential and parallel
algorithms in :mod:`constrained_lqr_jax.solver`.

The package supports optional dual regularization: dynamics use blocks
``Delta_k`` and equality constraints use blocks ``Sigma_k``. Setting these
blocks to zero recovers exact dynamics and exact equality constraints.

The parallel backward pass is built on the *mixed-constraint interval value
function* (IVF) — a uniform, fixed-dimension representation of the cost-to-go of
an interval ``[i, j)`` that **carries the constraint** instead of eliminating
its multiplier at the base case.  This is what lets a single, branch-free
associative scan handle arbitrary constraints (state-only ``E = 0``,
rank-deficient ``E``, ``p > m`` and the full-row-rank / unconstrained cases)
without any rank requirement.  The construction mirrors ``MixedConstraintIVF``
in the Lean formalization (see ``MIXED_CONSTRAINT_ANALYSIS.md``).

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
    I = jnp.broadcast_to(jnp.eye(n), batch + (n, n))
    tr = lambda X: jnp.swapaxes(X, -2, -1)

    # outer o = [xi(n), xk(n), y2(n), λ1(p)] ; interior w = [z(n), y1(n), λ2(p)]
    o = 3 * n + p
    D = 5 * n + 2 * p
    xi = slice(0, n)
    xk = slice(n, 2 * n)
    y2 = slice(2 * n, 3 * n)
    l1 = slice(3 * n, 3 * n + p)
    z = slice(3 * n + p, 4 * n + p)
    y1 = slice(4 * n + p, 5 * n + p)
    l2 = slice(5 * n + p, 5 * n + 2 * p)

    H = jnp.zeros(batch + (D, D))
    h = jnp.zeros(batch + (D,))

    def setH(a, b, val):
        nonlocal H
        H = H.at[..., a, b].add(val)

    def seth(a, val):
        nonlocal h
        h = h.at[..., a].add(val)

    # left IVF: x=xi, x'=z, y=y1, λ=λ1
    setH(xi, xi, P1)
    setH(y1, xi, A1)
    setH(xi, y1, tr(A1))
    setH(y1, z, -I)
    setH(z, y1, -I)
    setH(y1, y1, -C1)
    setH(y1, l1, -Cyl1)
    setH(l1, y1, -tr(Cyl1))
    setH(l1, xi, F1)
    setH(xi, l1, tr(F1))
    setH(l1, l1, -Cll1)
    seth(xi, p1)
    seth(y1, c1)
    seth(l1, -g1)
    # right IVF: x=z, x'=xk, y=y2, λ=λ2
    setH(z, z, P2)
    setH(y2, z, A2)
    setH(z, y2, tr(A2))
    setH(y2, xk, -I)
    setH(xk, y2, -I)
    setH(y2, y2, -C2)
    setH(y2, l2, -Cyl2)
    setH(l2, y2, -tr(Cyl2))
    setH(l2, z, F2)
    setH(z, l2, tr(F2))
    setH(l2, l2, -Cll2)
    seth(z, p2)
    seth(y2, c2)
    seth(l2, -g2)

    Ht, ht = _schur_eliminate(H, h, o)

    # read off combined IVF (outer vars x=xi, x'=xk, y=y2, λ=λ1)
    oxi = slice(0, n)
    oy = slice(2 * n, 3 * n)
    ol = slice(3 * n, 3 * n + p)
    P = symmetrize(Ht[..., oxi, oxi])
    A = Ht[..., oy, oxi]
    C = symmetrize(-Ht[..., oy, oy])
    Cyl = -Ht[..., oy, ol]
    Cll = symmetrize(-Ht[..., ol, ol])
    F = Ht[..., ol, oxi]
    pv = ht[..., oxi]
    cc = ht[..., oy]
    g = -ht[..., ol]
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
    I = jnp.broadcast_to(jnp.eye(n), batch + (n, n))
    tr = lambda X: jnp.swapaxes(X, -2, -1)

    # outer o = [xk(n), λ(p)] ; interior w = [y(n), xN(n), ν(p)]
    o = n + p
    Dd = 3 * n + 2 * p
    xk = slice(0, n)
    ll = slice(n, n + p)
    y = slice(n + p, 2 * n + p)
    xN = slice(2 * n + p, 3 * n + p)
    nu = slice(3 * n + p, 3 * n + 2 * p)

    H = jnp.zeros(batch + (Dd, Dd))
    h = jnp.zeros(batch + (Dd,))

    def setH(a, b, val):
        nonlocal H
        H = H.at[..., a, b].add(val)

    def seth(a, val):
        nonlocal h
        h = h.at[..., a].add(val)

    # IVF: x=xk, x'=xN, y=y, λ=λ
    setH(xk, xk, P)
    setH(y, xk, A)
    setH(xk, y, tr(A))
    setH(y, xN, -I)
    setH(xN, y, -I)
    setH(y, y, -C)
    setH(y, ll, -Cyl)
    setH(ll, y, -tr(Cyl))
    setH(ll, xk, F)
    setH(xk, ll, tr(F))
    setH(ll, ll, -Cll)
    seth(xk, pv)
    seth(y, c)
    seth(ll, -g)
    # terminal cost on xN
    setH(xN, xN, Q_N)
    seth(xN, q_N)
    # terminal constraint on xN
    setH(nu, xN, D_N)
    setH(xN, nu, tr(D_N))
    setH(nu, nu, -Sigma_N)
    seth(nu, -d_N)

    Ht, ht = _schur_eliminate(H, h, o)

    oxk = slice(0, n)
    ol = slice(n, n + p)
    P_new = symmetrize(Ht[..., oxk, oxk])
    F_new = Ht[..., ol, oxk]
    Cll_new = symmetrize(-Ht[..., ol, ol])
    p_new = ht[..., oxk]
    g_new = -ht[..., ol]
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
