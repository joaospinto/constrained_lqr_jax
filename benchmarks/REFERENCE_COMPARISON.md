# Reference Benchmark Comparison

These are local CPU wall-clock timings collected while comparing the structured
dual recovery implementation against
`https://github.com/joaospinto/constrained_lqr_elimination`.

Reference command:

```sh
bazel run -c opt //:clqr_benchmark
```

Reference checkout:

```text
constrained_lqr_elimination main zip, commit 552d02bcc17af88d027c626eeccc7c903fe0be3e
```

Structured JAX one-shot command:

```sh
uv run python benchmarks/bench_clqr_reference.py --repeats 3
```

Structured JAX cached-solve command:

```sh
uv run python benchmarks/bench_cached_solve.py --repeats 3
```

Representative constrained rows on this CPU:

| Case | C++ sequential mean | JAX sequential median | JAX parallel one-shot median | JAX cached parallel solve median |
|---|---:|---:|---:|---:|
| `N=32 n=6 m=3 p=1` | `300 us` | `1132 us` | `1359 us` | `1135 us` |
| `N=32 n=6 m=3 p=2` | `297 us` | `1279 us` | `1460 us` | `1216 us` |
| `N=64 n=6 m=3 p=1` | `590 us` | `1985 us` | `2347 us` | `1910 us` |
| `N=64 n=6 m=3 p=2` | `591 us` | `2458 us` | `2543 us` | `2137 us` |
| `N=128 n=8 m=4 p=1` | `1339 us` | `4493 us` | `4907 us` | `3870 us` |
| `N=128 n=8 m=4 p=2` | `1306 us` | `5258 us` | `5467 us` | `4369 us` |

The C++ reference remains substantially faster on this CPU.  The JAX parallel
path is intended to expose logarithmic-depth structure, not to beat optimized
sequential C++ without GPU/accelerator parallelism.  The cached JAX solve path
does reduce repeated-RHS solve time by avoiding repeated RHS-independent dual
recovery setup.

## Phase Timings

Collected with a standalone phase script.  Times are median microseconds per
JIT-warmed call.

| Case | Backward | Forward | Dual total | Dual setup | Dual apply | One-shot |
|---|---:|---:|---:|---:|---:|---:|
| `N=32 n=6 m=3 p=0` | `256` | `214` | `14` | n/a | n/a | `536` |
| `N=32 n=6 m=3 p=1` | `747` | `413` | `356` | `341` | `128` | `1558` |
| `N=32 n=6 m=3 p=2` | `1834` | `1118` | `381` | `834` | `69` | `1650` |
| `N=64 n=6 m=3 p=1` | `1454` | `580` | `608` | `621` | `74` | `2686` |
| `N=64 n=6 m=3 p=2` | `1566` | `553` | `693` | `704` | `84` | `3228` |
| `N=128 n=8 m=4 p=1` | `5480` | `2228` | `1421` | `1542` | `111` | `6246` |
| `N=128 n=8 m=4 p=2` | `3889` | `2336` | `2035` | `1580` | `130` | `6197` |

Cached dual application is small once the RHS-independent dual recovery cache
has been built.
