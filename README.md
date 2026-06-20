# constrained_lqr_jax

JAX implementation of equality-constrained finite-horizon LQR with sequential
and parallel solve paths.  The parallel path uses scan-based primal passes and
structured block-tridiagonal dual recovery.

## Verification

Verification for the structured dual recovery changes:

```sh
uv run pytest -q
# 100 passed
```

## Benchmarks

Detailed benchmark commands and additional rows are in
`benchmarks/REFERENCE_COMPARISON.md`.

## Algorithm

The solver handles finite-horizon equality-constrained LQR problems with
states `x_k in R^n`, controls `u_k in R^m`, and stage constraint dimension
`p`:

```text
minimize  sum_{k=0}^{N-1} 1/2 [x_k; u_k]^T [Q_k M_k; M_k^T R_k] [x_k; u_k]
                         + q_k^T x_k + r_k^T u_k
        + 1/2 x_N^T Q_N x_N + q_N^T x_N

subject to
        x_0                     = c_0
        x_{k+1} - A_k x_k - B_k u_k = c_{k+1},  k = 0, ..., N-1
        D_k x_k + E_k u_k      = d_k,           k = 0, ..., N-1
        D_N x_N                = d_N.
```

`Delta_k` and `Sigma_k` are optional dual regularization blocks for the
dynamics and equality multipliers.  Setting `p = 0` gives the unconstrained
LQR problem.  The returned solution contains primal variables `(X, U)`,
dynamics multipliers `Y`, equality multipliers `Lam`, and the affine
cost-to-go/feedforward terms used internally.

The parallel solve has three phases.

1. **Backward pass.**  Each stage is converted into a mixed
   initial-value-function (mixed IVF) block: controls are eliminated locally,
   while equality constraints are carried symbolically.  Mixed IVF composition
   is associative, so `jax.lax.associative_scan` composes all suffix intervals
   in logarithmic scan depth.  Folding each suffix into the terminal block gives
   the constrained cost-to-go data `(P_k, p_k, F_k, Cll_k, g_k)` for every
   stage.

2. **Forward primal pass.**  From the cost-to-go data, each stage forms an
   affine local KKT map from `x_k` to `(u_k, x_{k+1})` and auxiliary local
   multipliers.  The closed-loop transitions have the form
   `x_{k+1} = Phi_k x_k + psi_k`.  Affine transition composition is also
   associative, so a second `jax.lax.associative_scan` computes all states in
   logarithmic scan depth; controls are then evaluated stagewise by `vmap`.

3. **Dual recovery.**  Given `(X, U)`, the remaining KKT stationarity equations
   are linear in `(Y, Lam)`.  Each stage couples only `(Y_k, Y_{k+1}, Lam_k)`.
   The solver eliminates `Lam_k` locally with a pseudoinverse/projector and
   assembles symmetric block-tridiagonal normal equations for
   `Y_0, ..., Y_N`.  Those equations are solved by cyclic reduction: each level
   eliminates independent midpoint blocks with `vmap`, reducing the horizon by
   about half, then a top-down recovery reconstructs all eliminated costates.
   Equality multipliers are recovered independently per stage afterward.

Hard state-only constraints with more constraints than controls use an
endpoint-domain projection before the three phases above.  Each stage is first
converted to a linear endpoint relation
`H_left,k x_k + H_right,k x_{k+1} = h_k` describing the states for which some
control satisfies both dynamics and stage constraints.  An associative scan
combines these relations into suffix feasibility domains `H_k x_k = h_k`; the
solver then writes `x_k = xbar_k + P_k z_k`, where `P_k` projects onto the
domain nullspace.  The transformed problem is solved by the same scan-based LQR
machinery and the original multipliers are recovered by the structured dual
pass.  Other hard and regularized cases use the mixed-IVF scan path directly.

`factor_parallel(inputs)` builds RHS-independent data: zero-RHS cost-to-go
feedback matrices and the dual-recovery block-tridiagonal factorization.
`solve_parallel(inputs, factor_outputs, solve_inputs)` reuses that cache but
still recomputes RHS-dependent affine recurrences.  `factor_and_solve_parallel`
is the end-to-end compiled solve for one RHS and does not reuse cached
factorization outputs across calls.

For fixed `(n, m, p)`, the dominant arithmetic work is linear in `N`, while the
parallel depth of the scan and cyclic-reduction phases is logarithmic in `N`
under JIT compilation.  Local dense algebra is cubic in the stage dimensions.
Pseudoinverses are used only for local saddle/projector blocks and local
cyclic-reduction pivots, not for a dense horizon-sized KKT system.
