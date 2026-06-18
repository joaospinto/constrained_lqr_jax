from __future__ import annotations

import argparse
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from constrained_lqr_jax.helpers import compute_residual
from constrained_lqr_jax.solver import factor_and_solve, factor_and_solve_parallel
from constrained_lqr_jax.types import FactorizationInputs, SolveInputs

jax.config.update("jax_enable_x64", True)


def deterministic_value(seed: int, i: int, j: int = 0) -> float:
    x = float(seed + 17 * i + 31 * j)
    return 0.5 * math.sin(0.37 * x) + 0.25 * math.cos(0.19 * x)


def generated_vector(size: int, seed: int, scale: float = 1.0) -> np.ndarray:
    return np.array([scale * deterministic_value(seed, i) for i in range(size)])


def generated_matrix(
    rows: int, cols: int, seed: int, scale: float = 1.0
) -> np.ndarray:
    return np.array(
        [
            [scale * deterministic_value(seed, i, j) for j in range(cols)]
            for i in range(rows)
        ]
    )


def positive_definite(size: int, seed: int, diagonal: float) -> np.ndarray:
    g = generated_matrix(size, size, seed, 0.15)
    out = g.T @ g
    out[np.diag_indices(size)] += diagonal
    return out


def make_reference_problem(
    seed: int, horizon: int, states: int, controls: int, constraints: int
) -> tuple[FactorizationInputs, SolveInputs]:
    x = np.stack(
        [generated_vector(states, seed + 100 + i, 0.5) for i in range(horizon + 1)]
    )
    u = np.stack(
        [generated_vector(controls, seed + 200 + i, 0.4) for i in range(horizon)]
    )

    A = []
    B = []
    c = []
    Q = []
    R = []
    M = []
    q = []
    r = []
    D = []
    E = []
    d = []
    for i in range(horizon):
        Ai = generated_matrix(states, states, seed + 10 * i, 0.1)
        Ai[np.diag_indices(states)] += 0.9
        Bi = generated_matrix(states, controls, seed + 300 + 10 * i, 0.2)
        A.append(Ai)
        B.append(Bi)
        c.append(x[i + 1] - Ai @ x[i] - Bi @ u[i])
        Q.append(positive_definite(states, seed + 400 + i, 1.0))
        R.append(positive_definite(controls, seed + 500 + i, 1.5))
        M.append(generated_matrix(states, controls, seed + 600 + i, 0.03))
        q.append(generated_vector(states, seed + 700 + i, 0.2))
        r.append(generated_vector(controls, seed + 800 + i, 0.2))

        Dk = np.zeros((constraints, states))
        Ek = np.zeros((constraints, controls))
        dk = np.zeros((constraints,))
        if constraints > 0 and i % 3 == 1:
            Dk = generated_matrix(constraints, states, seed + 900 + i, 0.25)
            Ek = generated_matrix(constraints, controls, seed + 1000 + i, 0.25)
            # Reference repo uses C x + D u + d_ref = 0.
            dk = Dk @ x[i] + Ek @ u[i]
        D.append(Dk)
        E.append(Ek)
        d.append(dk)

    D.append(np.zeros((constraints, states)))
    d.append(np.zeros((constraints,)))

    return (
        FactorizationInputs(
            A=jnp.array(A),
            B=jnp.array(B),
            Q=jnp.array(Q + [positive_definite(states, seed + 1200, 1.5)]),
            M=jnp.array(M),
            R=jnp.array(R),
            D=jnp.array(D),
            E=jnp.array(E),
            Delta=jnp.zeros((horizon + 1, states, states)),
            Sigma=jnp.zeros((horizon + 1, constraints, constraints)),
        ),
        SolveInputs(
            q=jnp.array(q + [generated_vector(states, seed + 1300, 0.2)]),
            r=jnp.array(r),
            c=jnp.array([x[0]] + c),
            d=jnp.array(d),
        ),
    )


def block_until_ready(tree):
    return jax.tree.map(lambda x: x.block_until_ready(), tree)


def bench(fn, args, iterations: int, repeats: int):
    out = block_until_ready(fn(*args))
    times_us = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            out = block_until_ready(fn(*args))
        times_us.append(1e6 * (time.perf_counter() - start) / iterations)
    return statistics.median(times_us), min(times_us), out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    dimensions = [
        (16, 4, 2, 200),
        (16, 6, 3, 100),
        (32, 6, 3, 50),
        (64, 6, 3, 20),
        (128, 8, 4, 10),
    ]
    print(
        "case,iterations,repeats,sequential_us,parallel_us,"
        "sequential_best_us,parallel_best_us,"
        "sequential_residual,parallel_residual"
    )
    seed = 1
    for horizon, states, controls, iterations in dimensions:
        for constraints in range(3):
            fac, si = make_reference_problem(
                seed, horizon, states, controls, constraints
            )
            seed += 1
            seq = jax.jit(factor_and_solve)
            par = jax.jit(factor_and_solve_parallel)
            seq_us, seq_best_us, seq_sol = bench(
                seq, (fac, si), iterations, args.repeats
            )
            par_us, par_best_us, par_sol = bench(
                par, (fac, si), iterations, args.repeats
            )
            seq_res = jnp.max(jnp.abs(compute_residual(fac, si, seq_sol)))
            par_res = jnp.max(jnp.abs(compute_residual(fac, si, par_sol)))
            print(
                f"N={horizon} n={states} m={controls} p={constraints} mixed,"
                f"{iterations},{args.repeats},"
                f"{seq_us:.1f},{par_us:.1f},"
                f"{seq_best_us:.1f},{par_best_us:.1f},"
                f"{float(seq_res):.2e},{float(par_res):.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
