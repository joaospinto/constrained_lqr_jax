"""Tests for the stagewise-constrained LQR solver (sequential and parallel).

Covers arbitrary constraints ``D x + E u = d``: state-only (``E = 0``),
full-row-rank ``E``, rank-deficient / mixed ``E``, more constraints than
controls (``p > m``), the unconstrained case (``p = 0``), explicit infeasibility
reporting, sequential/parallel agreement, JIT and a few edge cases.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from constrained_lqr_jax.types import FactorizationInputs, SolveInputs
from constrained_lqr_jax.helpers import compute_residual
from constrained_lqr_jax.solver import (
    factor,
    factor_and_solve,
    factor_and_solve_parallel,
    factor_parallel,
    solve,
    solve_general,
    solve_parallel,
)

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


def _spd(rng, s):
    L = rng.standard_normal((s, s))
    return L @ L.T + s * np.eye(s)


def make_problem(seed, N, n, m, p, emode):
    """Random *feasible* problem.  ``emode`` in {'zero','full','mixed'}."""
    rng = np.random.default_rng(seed)
    Q = np.stack([_spd(rng, n) for _ in range(N + 1)])
    R = np.stack([_spd(rng, m) for _ in range(N)])
    M = rng.standard_normal((N, n, m)) * 0.3
    A = rng.standard_normal((N, n, n)) * 0.5
    B = rng.standard_normal((N, n, m))
    D = rng.standard_normal((N, p, n)) * 0.5
    if p > 0:
        D[0] = 0.0  # avoid the degenerate stage-0 state-constraint vs forced x0
    if emode == "zero":
        E = np.zeros((N, p, m))
    elif emode == "full":
        E = rng.standard_normal((N, p, m))
    else:  # mixed: alternate state-only / control-coupled
        E = rng.standard_normal((N, p, m))
        E[1::2] = 0.0
    q = rng.standard_normal((N + 1, n))
    r = rng.standard_normal((N, m))
    c = rng.standard_normal((N + 1, n))
    # feasible rhs from a feasible trajectory (x0 = c0)
    xt = np.zeros((N + 1, n))
    ut = rng.standard_normal((N, m))
    xt[0] = c[0]
    for k in range(N):
        xt[k + 1] = A[k] @ xt[k] + B[k] @ ut[k] + c[k + 1]
    d = (
        np.stack([D[k] @ xt[k] + E[k] @ ut[k] for k in range(N)])
        if p > 0
        else np.zeros((N, 0))
    )
    fac = FactorizationInputs(
        A=jnp.array(A),
        B=jnp.array(B),
        Q=jnp.array(Q),
        M=jnp.array(M),
        R=jnp.array(R),
        D=jnp.array(D),
        E=jnp.array(E),
    )
    si = SolveInputs(q=jnp.array(q), r=jnp.array(r), c=jnp.array(c), d=jnp.array(d))
    return fac, si


def dense_primal(fac, si):
    """Ground-truth primal via a dense least-squares KKT solve (handles the
    degenerate state-only case where the KKT is singular)."""
    A, B, Q, M, R, D, E = (
        np.array(fac.A),
        np.array(fac.B),
        np.array(fac.Q),
        np.array(fac.M),
        np.array(fac.R),
        np.array(fac.D),
        np.array(fac.E),
    )
    q, r, c, d = (np.array(si.q), np.array(si.r), np.array(si.c), np.array(si.d))
    N, n, m, p = A.shape[0], A.shape[1], B.shape[2], D.shape[1]
    nxu = N * (n + m) + n
    ny = (N + 1) * n
    nt = nxu + ny + N * p
    K = np.zeros((nt, nt))
    rhs = np.zeros(nt)
    for k in range(N):
        xs, us = k * (n + m), k * (n + m) + n
        K[xs : xs + n, xs : xs + n] = Q[k]
        K[xs : xs + n, us : us + m] = M[k]
        K[us : us + m, xs : xs + n] = M[k].T
        K[us : us + m, us : us + m] = R[k]
    xsN = N * (n + m)
    K[xsN : xsN + n, xsN : xsN + n] = Q[N]
    K[nxu : nxu + n, 0:n] = -np.eye(n)
    K[0:n, nxu : nxu + n] = -np.eye(n)
    for k in range(N):
        ys = nxu + (k + 1) * n
        xk, uk = k * (n + m), k * (n + m) + n
        xk1 = (k + 1) * (n + m) if k + 1 < N else N * (n + m)
        K[ys : ys + n, xk : xk + n] = A[k]
        K[ys : ys + n, uk : uk + m] = B[k]
        K[ys : ys + n, xk1 : xk1 + n] = -np.eye(n)
        K[xk : xk + n, ys : ys + n] = A[k].T
        K[uk : uk + m, ys : ys + n] = B[k].T
        K[xk1 : xk1 + n, ys : ys + n] = -np.eye(n)
    for k in range(N):
        ls = nxu + ny + k * p
        xk, uk = k * (n + m), k * (n + m) + n
        K[ls : ls + p, xk : xk + n] = D[k]
        K[ls : ls + p, uk : uk + m] = E[k]
        K[xk : xk + n, ls : ls + p] = D[k].T
        K[uk : uk + m, ls : ls + p] = E[k].T
    for k in range(N):
        xs, us = k * (n + m), k * (n + m) + n
        rhs[xs : xs + n] = -q[k]
        rhs[us : us + m] = -r[k]
    rhs[xsN : xsN + n] = -q[N]
    for k in range(N + 1):
        rhs[nxu + k * n : nxu + k * n + n] = -c[k]
    for k in range(N):
        rhs[nxu + ny + k * p : nxu + ny + k * p + p] = d[k]
    sv = np.linalg.lstsq(K, rhs, rcond=None)[0]
    x = np.stack(
        [
            sv[k * (n + m) : k * (n + m) + n]
            if k < N
            else sv[N * (n + m) : N * (n + m) + n]
            for k in range(N + 1)
        ]
    )
    u = np.stack([sv[k * (n + m) + n : k * (n + m) + n + m] for k in range(N)])
    return x, u


CONFIGS = [
    ("state-only", 4, 3, 2, 1, "zero"),
    ("state-only", 6, 3, 1, 1, "zero"),
    ("full-rank", 4, 3, 2, 2, "full"),
    ("full-rank", 8, 3, 3, 2, "full"),
    ("mixed", 6, 4, 2, 2, "mixed"),
    ("p>m", 4, 3, 1, 2, "zero"),
    ("unconstrained", 8, 3, 2, 0, "zero"),
    ("single-stage", 1, 2, 2, 1, "full"),
]

ATOL = 1e-8

SOLVERS = [("sequential", factor_and_solve), ("parallel", factor_and_solve_parallel)]


# ═══════════════════════════════════════════════════════════════════════════
# Primal correctness (both solvers)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
@pytest.mark.parametrize("name,N,n,m,p,emode", CONFIGS)
def test_primal_matches_dense(solver_name, solver, name, N, n, m, p, emode):
    fac, si = make_problem(7 + N + n + m + p, N, n, m, p, emode)
    sol = solver(fac, si)
    xd, ud = dense_primal(fac, si)
    np.testing.assert_allclose(sol.X, xd, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(sol.U, ud, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
@pytest.mark.parametrize("name,N,n,m,p,emode", CONFIGS)
def test_valid_kkt_point(solver_name, solver, name, N, n, m, p, emode):
    """The returned (x,u,y,lam) is an exact KKT point (small residual)."""
    fac, si = make_problem(7 + N + n + m + p, N, n, m, p, emode)
    sol = solver(fac, si)
    res = compute_residual(fac, si, sol)
    np.testing.assert_allclose(res, 0, atol=ATOL)


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
@pytest.mark.parametrize("name,N,n,m,p,emode", CONFIGS)
def test_constraint_satisfaction(solver_name, solver, name, N, n, m, p, emode):
    fac, si = make_problem(7 + N + n + m + p, N, n, m, p, emode)
    sol = solver(fac, si)
    if p > 0:
        res = jax.vmap(lambda Dk, Ek, xk, uk, dk: Dk @ xk + Ek @ uk - dk)(
            fac.D, fac.E, sol.X[:N], sol.U, si.d
        )
        np.testing.assert_allclose(res, 0, atol=ATOL)


# ═══════════════════════════════════════════════════════════════════════════
# Sequential / parallel agreement
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("name,N,n,m,p,emode", CONFIGS)
def test_sequential_parallel_agree(name, N, n, m, p, emode):
    fac, si = make_problem(7 + N + n + m + p, N, n, m, p, emode)
    s = factor_and_solve(fac, si)
    pll = factor_and_solve_parallel(fac, si)
    np.testing.assert_allclose(s.X, pll.X, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(s.U, pll.U, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(s.Y, pll.Y, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(s.Lam, pll.Lam, atol=1e-7, rtol=1e-6)


def test_separate_factor_and_solve_matches_factor_and_solve():
    fac, si = make_problem(17, 5, 3, 2, 1, "mixed")

    seq_fac = factor(fac)
    seq_sol = solve(fac, seq_fac, si)
    seq_ref = factor_and_solve(fac, si)

    par_fac = factor_parallel(fac)
    par_sol = solve_parallel(fac, par_fac, si)
    par_ref = factor_and_solve_parallel(fac, si)

    np.testing.assert_allclose(seq_sol.X, seq_ref.X, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(seq_sol.U, seq_ref.U, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(par_sol.X, par_ref.X, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(par_sol.U, par_ref.U, atol=1e-8, rtol=1e-7)


def test_factorization_can_be_reused_for_multiple_rhs():
    fac, si0 = make_problem(23, 6, 3, 2, 2, "full")
    _, si1 = make_problem(29, 6, 3, 2, 2, "full")

    seq_fac = factor(fac)
    sol0 = solve(fac, seq_fac, si0)
    sol1 = solve(fac, seq_fac, si1)

    ref0 = factor_and_solve(fac, si0)
    ref1 = factor_and_solve(fac, si1)

    np.testing.assert_allclose(sol0.X, ref0.X, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(sol0.U, ref0.U, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(sol1.X, ref1.X, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(sol1.U, ref1.U, atol=1e-8, rtol=1e-7)


# ═══════════════════════════════════════════════════════════════════════════
# Degeneracy / infeasibility handling
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
def test_state_only_no_nan(solver_name, solver):
    """State-only (E = 0) returns finite values, not NaN."""
    fac, si = make_problem(1, 5, 3, 2, 1, "zero")
    sol = solver(fac, si)
    assert bool(jnp.all(jnp.isfinite(sol.X)))
    assert bool(jnp.all(jnp.isfinite(sol.U)))


@pytest.mark.parametrize("parallel", [False, True])
def test_infeasible_is_reported(parallel):
    """An infeasible problem is flagged (feasible=False), not silent NaN."""
    fac, si = make_problem(2, 4, 3, 2, 1, "zero")
    rng = np.random.default_rng(99)
    D = np.array(fac.D)
    D[0] = rng.standard_normal((1, 3))  # stage-0 state constraint vs forced x0=c0
    fac2 = FactorizationInputs(
        A=fac.A, B=fac.B, Q=fac.Q, M=fac.M, R=fac.R, D=jnp.array(D), E=fac.E
    )
    sol, st = solve_general(fac2, si, parallel=parallel)
    assert not bool(st.feasible)
    assert float(st.residual) > 1e-3
    assert bool(jnp.all(jnp.isfinite(sol.X)))  # finite, not NaN


@pytest.mark.parametrize("parallel", [False, True])
def test_solve_general_feasible(parallel):
    fac, si = make_problem(5, 6, 3, 2, 2, "mixed")
    sol, st = solve_general(fac, si, parallel=parallel)
    assert bool(st.feasible)
    assert float(st.residual) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
def test_nontrivial_constraints_change_solution(solver_name, solver):
    fac, si = make_problem(42, 5, 3, 2, 1, "full")
    sol_c = solver(fac, si)
    fac0 = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=jnp.zeros((5, 0, 3)),
        E=jnp.zeros((5, 0, 2)),
    )
    si0 = SolveInputs(q=si.q, r=si.r, c=si.c, d=jnp.zeros((5, 0)))
    sol_u = solver(fac0, si0)
    assert not np.allclose(sol_c.X, sol_u.X, atol=1e-2)


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
def test_identity_dynamics(solver_name, solver):
    fac, si = make_problem(333, 5, 3, 2, 0, "zero")
    fac = FactorizationInputs(
        A=jnp.tile(jnp.eye(3), (5, 1, 1)),
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=fac.D,
        E=fac.E,
    )
    sol = solver(fac, si)
    xd, ud = dense_primal(fac, si)
    np.testing.assert_allclose(sol.X, xd, atol=1e-6, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# JIT
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("solver_name,solver", SOLVERS)
def test_jit(solver_name, solver):
    fac, si = make_problem(3, 5, 3, 2, 1, "mixed")
    sol = jax.jit(solver)(fac, si)
    xd, ud = dense_primal(fac, si)
    np.testing.assert_allclose(sol.X, xd, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
