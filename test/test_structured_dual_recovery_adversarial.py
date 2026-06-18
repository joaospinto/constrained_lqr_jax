from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from constrained_lqr_jax.helpers import compute_residual
from constrained_lqr_jax.solver import (
    factor,
    factor_and_solve,
    factor_and_solve_parallel,
    factor_parallel,
    solve,
    solve_general,
    solve_parallel,
    _block_tridiag_factor,
    _block_tridiag_solve,
    _block_tridiag_solve_from_factor,
    _state_only_suffix_domains_parallel,
)
from constrained_lqr_jax.types import FactorizationInputs, SolveInputs
from test.test_solver import dense_primal, make_problem

jax.config.update("jax_enable_x64", True)


def _max_abs(x) -> float:
    return float(jnp.max(jnp.abs(x)))


def _scale(*xs) -> float:
    return max(1.0, *(float(jnp.max(jnp.abs(x))) for x in xs if x.size))


def _seed(N: int, n: int, m: int, p: int, emode: str, regularized: bool) -> int:
    return (
        10_000
        + 97 * N
        + 13 * n
        + 7 * m
        + 3 * p
        + len(emode)
        + (503 if regularized else 0)
    )


def _dense_block_tridiag(diag, upper):
    Np1, n = diag.shape[:2]
    dense = jnp.zeros((Np1 * n, Np1 * n), dtype=diag.dtype)
    idx = jnp.arange(n)
    for k in range(Np1):
        rows = k * n + idx[:, None]
        cols = k * n + idx[None, :]
        dense = dense.at[rows, cols].set(diag[k])
    for k in range(Np1 - 1):
        rows = k * n + idx[:, None]
        cols = (k + 1) * n + idx[None, :]
        dense = dense.at[rows, cols].set(upper[k])
        dense = dense.at[cols.T, rows.T].set(upper[k].T)
    return dense


def test_block_tridiag_solve_matches_dense_spd_reference():
    rng = np.random.default_rng(1234)
    N, n = 7, 3
    upper = jnp.array(rng.standard_normal((N, n, n)) * 0.05)
    diag = []
    for _ in range(N + 1):
        M = rng.standard_normal((n, n))
        diag.append(M.T @ M + 4.0 * np.eye(n))
    diag = jnp.array(diag)
    rhs = jnp.array(rng.standard_normal((N + 1, n)))

    got = _block_tridiag_solve(diag, upper, rhs)
    factored = _block_tridiag_solve_from_factor(
        _block_tridiag_factor(diag, upper),
        rhs,
    )
    dense = _dense_block_tridiag(diag, upper)
    expected = jnp.linalg.solve(dense, rhs.reshape(-1)).reshape(N + 1, n)

    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(factored, expected, atol=1e-10, rtol=1e-10)


def test_block_tridiag_solve_singular_case_is_globally_compatible():
    diag = jnp.array([[[1.0]], [[2.0]], [[1.0]]])
    upper = jnp.array([[[-1.0]], [[-1.0]]])
    rhs = jnp.array([[1.0], [0.0], [-1.0]])

    got = _block_tridiag_solve(diag, upper, rhs)
    residual = _dense_block_tridiag(diag, upper) @ got.reshape(-1) - rhs.reshape(-1)

    assert _max_abs(residual) < 1e-10


def test_block_tridiag_solve_exact_singular_psd_pivot_is_compatible():
    diag = jnp.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [0.0, 2.0]],
        ]
    )
    upper = jnp.array(
        [
            [[0.25, 0.0], [0.0, 0.0]],
            [[-0.5, 0.0], [0.0, 0.0]],
        ]
    )
    y_ref = jnp.array([[0.7, -1.0], [1.3, 4.0], [-0.2, 0.5]])
    dense = _dense_block_tridiag(diag, upper)
    rhs = (dense @ y_ref.reshape(-1)).reshape(3, 2)

    got = _block_tridiag_solve(diag, upper, rhs)
    residual = dense @ got.reshape(-1) - rhs.reshape(-1)

    assert _max_abs(residual) < 1e-10


@pytest.mark.parametrize(
    "N,n,seed",
    [
        pytest.param(2, 2, 23, id="local-near-singular-pivot"),
        pytest.param(7, 3, 244, id="stress-sweep-worst-seed"),
    ],
)
def test_block_tridiag_solve_psd_singular_bidiag_normal_is_compatible(
    N, n, seed
):
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((N + 1, n, n))
    right = rng.standard_normal((N, n, n))
    if n > 1:
        left[:, -1, :] = 0.0
        right[:, -1, :] = 0.0
    else:
        left[::2, :, :] = 0.0

    diag = []
    upper = []
    for k in range(N + 1):
        block = left[k].T @ left[k]
        if k > 0:
            block = block + right[k - 1].T @ right[k - 1]
        diag.append(block)
    for k in range(N):
        upper.append(left[k].T @ right[k])
    diag = jnp.array(diag)
    upper = jnp.array(upper)
    dense = _dense_block_tridiag(diag, upper)
    y_ref = jnp.array(rng.standard_normal((N + 1, n)))
    rhs = (dense @ y_ref.reshape(-1)).reshape(N + 1, n)

    got = _block_tridiag_solve(diag, upper, rhs)
    residual = dense @ got.reshape(-1) - rhs.reshape(-1)

    assert _max_abs(residual) / _scale(rhs) < 1e-10


@pytest.mark.parametrize(
    "N,n,m,p,emode,regularized",
    [
        pytest.param(32, 8, 6, 2, "mixed", False, id="mixed-hard-p-le-m"),
        pytest.param(32, 8, 8, 4, "full", False, id="full-hard-p-le-m"),
        pytest.param(32, 8, 6, 2, "zero", False, id="state-only-hard"),
        pytest.param(
            32,
            8,
            3,
            5,
            "zero",
            False,
            id="state-only-hard-p-gt-m",
        ),
        pytest.param(32, 8, 6, 2, "mixed", True, id="mixed-regularized"),
        pytest.param(32, 8, 8, 4, "full", True, id="full-regularized"),
    ],
)
def test_parallel_solver_primal_matches_dense_on_larger_constrained_cases(
    N, n, m, p, emode, regularized
):
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )

    sol = factor_and_solve_parallel(fac, si)
    xd, ud = dense_primal(fac, si)

    primal_error = max(
        _max_abs(sol.X - jnp.array(xd)),
        _max_abs(sol.U - jnp.array(ud)),
    )
    primal_scale = _scale(sol.X, sol.U, jnp.array(xd), jnp.array(ud))
    residual = compute_residual(fac, si, sol)
    residual_scale = _scale(
        fac.A,
        fac.B,
        fac.Q,
        fac.M,
        fac.R,
        fac.D,
        fac.E,
        si.q,
        si.r,
        si.c,
        si.d,
        sol.X,
        sol.U,
        sol.Y,
        sol.Lam,
    )
    if p > m and emode == "zero" and not regularized:
        assert primal_error / primal_scale < 1e-6
        assert _max_abs(residual) / residual_scale < 1e-6
    else:
        np.testing.assert_allclose(sol.X, xd, atol=5e-6, rtol=5e-6)
        np.testing.assert_allclose(sol.U, ud, atol=5e-6, rtol=5e-6)
        assert _max_abs(residual) < 1e-6


@pytest.mark.parametrize(
    "name,N,n,m,p,emode,regularized",
    [
        pytest.param(
            "mixed-hard-p-le-m",
            64,
            8,
            6,
            2,
            "mixed",
            False,
            id="mixed-hard-p-le-m",
        ),
        pytest.param(
            "full-hard-p-le-m",
            64,
            8,
            8,
            4,
            "full",
            False,
            id="full-hard-p-le-m",
        ),
        pytest.param(
            "state-only-hard",
            64,
            8,
            6,
            2,
            "zero",
            False,
            id="state-only-hard",
        ),
        pytest.param(
            "state-only-hard-p-gt-m",
            64,
            8,
            3,
            5,
            "zero",
            False,
            id="state-only-hard-p-gt-m",
        ),
        pytest.param(
            "mixed-regularized",
            64,
            8,
            6,
            2,
            "mixed",
            True,
            id="mixed-regularized",
        ),
        pytest.param(
            "full-regularized",
            64,
            8,
            8,
            4,
            "full",
            True,
            id="full-regularized",
        ),
    ],
)
def test_parallel_solver_large_structured_dual_recovery_has_small_kkt_residual(
    name, N, n, m, p, emode, regularized
):
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )

    sol = factor_and_solve_parallel(fac, si)
    xd, ud = dense_primal(fac, si)
    residual = compute_residual(fac, si, sol)
    residual_scale = _scale(
        fac.A,
        fac.B,
        fac.Q,
        fac.M,
        fac.R,
        fac.D,
        fac.E,
        si.q,
        si.r,
        si.c,
        si.d,
        sol.X,
        sol.U,
        sol.Y,
        sol.Lam,
    )

    assert bool(jnp.all(jnp.isfinite(sol.X)))
    assert bool(jnp.all(jnp.isfinite(sol.U)))
    assert bool(jnp.all(jnp.isfinite(sol.Y)))
    assert bool(jnp.all(jnp.isfinite(sol.Lam)))
    primal_error = max(
        _max_abs(sol.X - jnp.array(xd)),
        _max_abs(sol.U - jnp.array(ud)),
    )
    primal_scale = _scale(sol.X, sol.U, jnp.array(xd), jnp.array(ud))
    primal_tol = 1e-6 if p > m and emode == "zero" and not regularized else 1e-12
    residual_tol = (
        1e-6 if p > m and emode == "zero" and not regularized else 1e-12
    )
    assert primal_error / primal_scale < primal_tol, (
        f"{name} scaled primal error was {primal_error / primal_scale:.3e}; "
        f"absolute error was {primal_error:.3e}"
    )
    scaled_residual = _max_abs(residual) / residual_scale
    assert scaled_residual < residual_tol, (
        f"{name} scaled KKT residual was {scaled_residual:.3e}; "
        f"absolute residual was {_max_abs(residual):.3e}"
    )


def test_separate_factor_and_solve_matches_hard_state_only_nullspace_path():
    N, n, m, p, emode, regularized = 32, 8, 3, 5, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )

    seq_sol = solve(fac, factor(fac), si)
    seq_ref = factor_and_solve(fac, si)
    par_sol = solve_parallel(fac, factor_parallel(fac), si)
    par_ref = factor_and_solve_parallel(fac, si)

    for sol, ref in [(seq_sol, seq_ref), (par_sol, par_ref)]:
        error = max(
            _max_abs(sol.X - ref.X),
            _max_abs(sol.U - ref.U),
        )
        scale = _scale(sol.X, sol.U, ref.X, ref.U)
        assert error / scale < 1e-12


def test_state_only_suffix_domain_scan_captures_full_viability_rank():
    N, n, m, p, emode, regularized = 32, 8, 3, 5, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    sol = factor_and_solve_parallel(fac, si)

    H, h = _state_only_suffix_domains_parallel(fac, si)
    residual = jax.vmap(lambda Hk, hk, xk: Hk @ xk - hk)(H, h, sol.X)

    assert np.linalg.matrix_rank(np.array(H[1]), tol=1e-8) == n
    assert _max_abs(residual) / _scale(H, h, sol.X) < 1e-10


def test_state_only_suffix_domain_scan_handles_rank_deficient_rows():
    N, n, m, p, emode, regularized = 16, 6, 2, 4, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    D = fac.D.at[:, 3].set(fac.D[:, 2])
    d = si.d.at[:, 3].set(si.d[:, 2])
    rank_def_fac = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=D,
        E=fac.E,
        Delta=fac.Delta,
        Sigma=fac.Sigma,
    )
    rank_def_si = SolveInputs(q=si.q, r=si.r, c=si.c, d=d)
    xd, _ = dense_primal(rank_def_fac, rank_def_si)

    H, h = _state_only_suffix_domains_parallel(rank_def_fac, rank_def_si)
    residual = jax.vmap(lambda Hk, hk, xk: Hk @ xk - hk)(
        H, h, jnp.array(xd)
    )

    assert _max_abs(residual) / _scale(H, h, jnp.array(xd)) < 1e-10


def test_parallel_solver_matches_dense_with_inactive_terminal_constraints():
    N, n, m, p, emode, regularized = 16, 6, 3, 2, "mixed", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    terminal_zero_fac = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=fac.D.at[N].set(jnp.zeros_like(fac.D[N])),
        E=fac.E,
        Delta=fac.Delta,
        Sigma=fac.Sigma.at[N].set(jnp.zeros_like(fac.Sigma[N])),
    )
    terminal_zero_si = SolveInputs(
        q=si.q,
        r=si.r,
        c=si.c,
        d=si.d.at[N].set(jnp.zeros_like(si.d[N])),
    )

    sol = factor_and_solve_parallel(terminal_zero_fac, terminal_zero_si)
    xd, ud = dense_primal(terminal_zero_fac, terminal_zero_si)
    residual = compute_residual(terminal_zero_fac, terminal_zero_si, sol)

    np.testing.assert_allclose(sol.X, xd, atol=5e-8, rtol=5e-8)
    np.testing.assert_allclose(sol.U, ud, atol=5e-8, rtol=5e-8)
    assert _max_abs(residual) < 1e-8


@pytest.mark.parametrize("N", [8, 16, 32, 64])
def test_rank_deficient_hard_state_only_projected_path_matches_dense(N):
    n, m, p, emode, regularized = 8, 3, 5, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    D = fac.D.at[:, 4].set(fac.D[:, 3])
    d = si.d.at[:, 4].set(si.d[:, 3])
    rank_def_fac = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=D,
        E=fac.E,
        Delta=fac.Delta,
        Sigma=fac.Sigma,
    )
    rank_def_si = SolveInputs(q=si.q, r=si.r, c=si.c, d=d)

    sol = factor_and_solve_parallel(rank_def_fac, rank_def_si)
    xd, ud = dense_primal(rank_def_fac, rank_def_si)
    primal_error = max(
        _max_abs(sol.X - jnp.array(xd)),
        _max_abs(sol.U - jnp.array(ud)),
    )
    primal_scale = _scale(sol.X, sol.U, jnp.array(xd), jnp.array(ud))
    residual = compute_residual(rank_def_fac, rank_def_si, sol)
    residual_scale = _scale(
        rank_def_fac.A,
        rank_def_fac.B,
        rank_def_fac.Q,
        rank_def_fac.M,
        rank_def_fac.R,
        rank_def_fac.D,
        rank_def_fac.E,
        rank_def_si.q,
        rank_def_si.r,
        rank_def_si.c,
        rank_def_si.d,
        sol.X,
        sol.U,
        sol.Y,
        sol.Lam,
    )

    assert primal_error / primal_scale < 1e-8
    assert _max_abs(residual) / residual_scale < 1e-5


def test_separate_factor_and_solve_matches_rank_deficient_projected_path():
    N, n, m, p, emode, regularized = 16, 8, 3, 5, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    D = fac.D.at[:, 4].set(fac.D[:, 3])
    d = si.d.at[:, 4].set(si.d[:, 3])
    rank_def_fac = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=D,
        E=fac.E,
        Delta=fac.Delta,
        Sigma=fac.Sigma,
    )
    rank_def_si = SolveInputs(q=si.q, r=si.r, c=si.c, d=d)

    seq_sol = solve(rank_def_fac, factor(rank_def_fac), rank_def_si)
    seq_ref = factor_and_solve(rank_def_fac, rank_def_si)
    par_sol = solve_parallel(rank_def_fac, factor_parallel(rank_def_fac), rank_def_si)
    par_ref = factor_and_solve_parallel(rank_def_fac, rank_def_si)

    for sol, ref in [(seq_sol, seq_ref), (par_sol, par_ref)]:
        error = max(
            _max_abs(sol.X - ref.X),
            _max_abs(sol.U - ref.U),
        )
        scale = _scale(sol.X, sol.U, ref.X, ref.U)
        assert error / scale < 1e-12


def test_inconsistent_duplicate_state_only_row_reports_infeasible():
    N, n, m, p, emode, regularized = 16, 8, 3, 5, "zero", False
    fac, si = make_problem(
        _seed(N, n, m, p, emode, regularized),
        N,
        n,
        m,
        p,
        emode,
        regularized=regularized,
    )
    D = fac.D.at[:, 4].set(fac.D[:, 3])
    d = si.d.at[:, 4].set(si.d[:, 3] + 0.25)
    bad_fac = FactorizationInputs(
        A=fac.A,
        B=fac.B,
        Q=fac.Q,
        M=fac.M,
        R=fac.R,
        D=D,
        E=fac.E,
        Delta=fac.Delta,
        Sigma=fac.Sigma,
    )
    bad_si = SolveInputs(q=si.q, r=si.r, c=si.c, d=d)

    sol, status = solve_general(bad_fac, bad_si, parallel=True)
    residual = compute_residual(bad_fac, bad_si, sol)

    assert not bool(status.feasible)
    assert _max_abs(residual) > 1e-3
