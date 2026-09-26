"""Parallel CSR × dense for the MaleCNS rate net.

``W.dot(x)`` on 166k neurons / ~25M synapses is memory-bound. Category replay
used to do that once per head per substep (12–21 heads × 6). Streaming ``W``
once per substep into ``k`` columns, and splitting rows across threads, cuts
that from ~1.9s/bar to ~0.2s/bar on an 8-core machine. Full-speed replay also
skips unchanged bars so a 2000-bar file is tens of seconds, not minutes.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import csr_matrix


def split_csr_by_nnz(W: csr_matrix, n_parts: int) -> list[tuple[int, int, csr_matrix]]:
    """Row blocks with roughly equal nonzero counts (load-balanced SPMV)."""
    n = int(W.shape[0])
    n_parts = max(1, min(int(n_parts), n))
    if n_parts == 1:
        return [(0, n, W)]
    indptr = W.indptr
    nnz = int(indptr[-1])
    if nnz <= 0:
        return [(0, n, W)]
    cuts = [0]
    target = 1
    quota = nnz / n_parts
    for i in range(1, n):
        if target < n_parts and int(indptr[i]) >= quota * target:
            cuts.append(i)
            target += 1
    cuts.append(n)
    blocks: list[tuple[int, int, csr_matrix]] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            blocks.append((int(a), int(b), W[a:b, :]))
    return blocks or [(0, n, W)]


def _dot_vec_block(blk: csr_matrix, x: np.ndarray, y: np.ndarray, r0: int, r1: int) -> None:
    y[r0:r1] = blk.dot(x)


def _dot_mat_block(blk: csr_matrix, x: np.ndarray, y: np.ndarray, r0: int, r1: int) -> None:
    y[r0:r1, :] = blk.dot(x)


class ParallelSpmv:
    """Thread-pooled row-blocked ``W @ x`` (vector or (n, k) batch)."""

    def __init__(self, W: csr_matrix, n_threads: int | None = None) -> None:
        self.W = W
        self.n = int(W.shape[0])
        cpu = os.cpu_count() or 2
        want = int(n_threads) if n_threads else min(8, max(1, cpu))
        self.blocks = split_csr_by_nnz(W, want)
        self.n_threads = len(self.blocks)
        self.pool: ThreadPoolExecutor | None = None
        if self.n_threads > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.n_threads, thread_name_prefix="flyw")
            atexit.register(self.pool.shutdown, wait=False)
        self._y1 = np.empty(self.n, dtype=np.float32)
        self._y2 = np.empty((self.n, 21), dtype=np.float32)

    def dot(self, x: np.ndarray) -> np.ndarray:
        """Sparse multiply. Returned buffer is reused — consume before the next call."""
        if x.ndim == 1:
            if self.pool is None:
                return self.W.dot(x)
            y = self._y1
            futs = [
                self.pool.submit(_dot_vec_block, blk, x, y, r0, r1)
                for r0, r1, blk in self.blocks
            ]
            for fut in futs:
                fut.result()
            return y
        k = int(x.shape[1])
        if k <= 0:
            return np.zeros((self.n, 0), dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        if self.pool is None:
            return self.W.dot(xc)
        if k > self._y2.shape[1]:
            self._y2 = np.empty((self.n, k), dtype=np.float32)
        y = self._y2[:, :k]
        futs = [
            self.pool.submit(_dot_mat_block, blk, xc, y, r0, r1)
            for r0, r1, blk in self.blocks
        ]
        for fut in futs:
            fut.result()
        return y
