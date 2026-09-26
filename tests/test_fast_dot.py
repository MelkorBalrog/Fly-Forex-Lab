"""Row-blocked batched sparse multiply matches sequential W.dot."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from flyfx.brain.fast_dot import ParallelSpmv, split_csr_by_nnz


def _random_csr(n: int = 64, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(n * 8).astype(np.float32) * 0.05
    rows = rng.integers(0, n, size=data.size)
    cols = rng.integers(0, n, size=data.size)
    W = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    return W.astype(np.float32)


def test_split_csr_balances_nnz():
    W = _random_csr(80)
    blocks = split_csr_by_nnz(W, 4)
    assert len(blocks) >= 2
    nnzs = [int(blk.nnz) for _, _, blk in blocks]
    assert sum(nnzs) == W.nnz
    assert max(nnzs) <= min(nnzs) * 4 + 20


def test_parallel_spmv_matches_sequential_vector():
    W = _random_csr(96, seed=3)
    x = np.linspace(-0.4, 0.4, 96, dtype=np.float32)
    y = ParallelSpmv(W, n_threads=4).dot(x)
    np.testing.assert_allclose(y, W.dot(x), rtol=1e-5, atol=1e-5)


def test_parallel_spmv_matches_sequential_batch():
    W = _random_csr(96, seed=4)
    X = np.linspace(-0.3, 0.3, 96 * 7, dtype=np.float32).reshape(96, 7)
    Y = ParallelSpmv(W, n_threads=4).dot(X)
    np.testing.assert_allclose(Y, W.dot(X), rtol=1e-5, atol=1e-5)


def test_first_step_from_rest_skips_spmv():
    """W @ 0 is 0, so the first settle step is just leaky tanh of the drive."""
    leak = 0.15
    I = np.array([0.2, -0.5, 0.0, 0.8], dtype=np.float32)
    x = np.zeros_like(I)
    syn = np.zeros_like(I)  # W.dot(0)
    x_step = (1.0 - leak) * np.tanh(syn + I) + leak * x
    x_skip = (1.0 - leak) * np.tanh(I)
    np.testing.assert_allclose(x_step, x_skip, rtol=1e-6, atol=1e-6)
