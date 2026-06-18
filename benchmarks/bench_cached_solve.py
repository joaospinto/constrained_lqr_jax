from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from benchmarks.bench_clqr_reference import make_reference_problem
from constrained_lqr_jax.helpers import compute_residual
from constrained_lqr_jax.solver import (
    factor_and_solve_parallel,
    factor_parallel,
    solve_parallel,
)

jax.config.update("jax_enable_x64", True)


def block_until_ready(tree):
    def block(x):
        return x.block_until_ready() if hasattr(x, "block_until_ready") else x

    return jax.tree.map(block, tree)


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
        (32, 6, 3, 1, 50),
        (32, 6, 3, 2, 50),
        (64, 6, 3, 1, 20),
        (64, 6, 3, 2, 20),
        (128, 8, 4, 1, 10),
        (128, 8, 4, 2, 10),
    ]

    print(
        "case,iterations,repeats,one_shot_us,cached_solve_us,"
        "one_shot_best_us,cached_solve_best_us,"
        "one_shot_residual,cached_residual"
    )
    seed = 100
    jit_factor = jax.jit(factor_parallel)
    jit_solve = jax.jit(solve_parallel)
    jit_one_shot = jax.jit(factor_and_solve_parallel)
    for horizon, states, controls, constraints, iterations in dimensions:
        fac, si = make_reference_problem(
            seed, horizon, states, controls, constraints
        )
        seed += 1
        par_factor = block_until_ready(jit_factor(fac))
        one_us, one_best_us, one_sol = bench(
            jit_one_shot, (fac, si), iterations, args.repeats
        )
        cached_us, cached_best_us, cached_sol = bench(
            jit_solve, (fac, par_factor, si), iterations, args.repeats
        )
        one_res = jnp.max(jnp.abs(compute_residual(fac, si, one_sol)))
        cached_res = jnp.max(jnp.abs(compute_residual(fac, si, cached_sol)))
        print(
            f"N={horizon} n={states} m={controls} p={constraints} mixed,"
            f"{iterations},{args.repeats},"
            f"{one_us:.1f},{cached_us:.1f},"
            f"{one_best_us:.1f},{cached_best_us:.1f},"
            f"{float(one_res):.2e},{float(cached_res):.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
