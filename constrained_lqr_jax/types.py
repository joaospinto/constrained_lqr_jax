"""Data containers for the constrained LQR solver.

``constrained_lqr_jax`` solves the *stagewise-constrained* LQR problem

    D_k x_k + E_k u_k = d_k,    k = 0, ..., N-1

with optional dual regularization.  There is
no rank requirement on ``D_k`` or ``E_k``: state-only constraints (``E_k = 0``),
rank-deficient ``E_k`` and ``p > m`` are all supported.  Setting the constraint
dimension to ``p = 0`` (empty ``D`` / ``E`` / ``d``) recovers the ordinary
unconstrained LQR problem with no separate code path.
"""

import jax

from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class FactorizationInputs:
    """LHS data for a stagewise-constrained LQR problem.

    Stagewise affine equality constraints are supported for ``k = 0, ..., N-1``
    with **arbitrary** ``D_k``, ``E_k`` (no rank requirement); ``p = 0`` gives
    the unconstrained problem.  ``Delta`` regularizes dynamics multipliers and
    ``Sigma`` regularizes stagewise equality multipliers.  Zero blocks recover
    exact hard equalities.

    Shapes (n, m state/control dims; p constraint dim):
        A: [N, n, n],
        B: [N, n, m],
        Q: [N+1, n, n],
        M: [N, n, m],
        R: [N, m, m],
        D: [N, p, n],
        E: [N, p, m],
        Delta: [N+1, n, n] dynamics dual regularization,
        Sigma: [N, p, p] constraint dual regularization.
    """

    A: jax.Array
    B: jax.Array
    Q: jax.Array
    M: jax.Array
    R: jax.Array
    D: jax.Array
    E: jax.Array
    Delta: jax.Array
    Sigma: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SequentialFactorizationOutputs:
    """Reusable LHS factorization for the sequential constrained LQR solve.

    Shapes (n, m state/control dims; p constraint dim):
        P:    [N+1, n, n] carried quadratic value matrices,
        F:    [N+1, p, n] carried constraint Jacobians,
        Cll:  [N+1, p, p] carried constraint curvature blocks,
        K:    [N, m, n]   affine feedback matrices,
        Phi:  [N, n, n]   closed-loop transition matrices.
    """

    P: jax.Array
    F: jax.Array
    Cll: jax.Array
    K: jax.Array
    Phi: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ParallelFactorizationOutputs:
    """Reusable LHS factorization for the parallel constrained LQR solve.

    The fields have the same meaning and shapes as
    :class:`SequentialFactorizationOutputs`; only the factorization algorithm
    differs.
    """

    P: jax.Array
    F: jax.Array
    Cll: jax.Array
    K: jax.Array
    Phi: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SolveInputs:
    """RHS data for a stagewise-constrained LQR problem.

    Shapes (n, m state/control dims; p constraint dim):
        q: [N+1, n],
        r: [N, m],
        c: [N+1, n],
        d: [N, p].
    """

    q: jax.Array
    r: jax.Array
    c: jax.Array
    d: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SolveOutputs:
    """Solution of the stagewise-constrained LQR problem.

    Shapes (n, m state/control dims; p constraint dim):
        X:   [N+1, n]   states,
        U:   [N, m]     controls,
        Y:   [N+1, n]   dynamics multipliers,
        Lam: [N, p]     constraint multipliers,
        p:   [N+1, n]   affine cost-to-go vectors,
        k:   [N, m]     affine feedforward terms (unused; kept for layout).
    """

    X: jax.Array
    U: jax.Array
    Y: jax.Array
    Lam: jax.Array
    p: jax.Array
    k: jax.Array
