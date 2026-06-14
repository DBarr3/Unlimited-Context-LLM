# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""MPO codec — a tensor-train (Matrix Product Operator) encoder/decoder for pool vectors.

Unlimited Context is virtual memory for attention: overflow is **encoded** to a local pool
and the right slice is **recovered** on demand. This module is the numeric half of that
contract — it compresses a slice's retrieval vector into a **tensor-train (TT)** factorization
on the way to disk and reconstructs it on recovery, so a long-running session's persistent
pool holds more reach per byte.

The technique is the standard **TT-SVD** decomposition (Oseledets, 2011): a length-``D``
vector is reshaped into a small multi-axis grid and factored into a chain of rank-bounded
3-D cores; truncating the bond rank trades a bounded amount of reconstruction error for a
smaller stored footprint. It is pure, deterministic linear algebra — no training, no model,
numpy only.

What it is / is not
-------------------
  * It *is* a lossy, bounded vector codec: ``recover(encode(v))`` returns an approximation of
    ``v`` whose fidelity (cosine to the original) rises with the bond ``rank``. Both the
    compression ratio and the measured fidelity are reported, never assumed.
  * It is *only* encode + recover. It does not rank, score, gate, or judge slices, and it has
    no notion of relevance, ground truth, or correctness — those are the retrieval/witness
    layers' jobs. The codec's sole purpose is making the disk/memory encoding smaller.
  * It is opt-in and fail-soft at the call sites that use it: the engine always works with the
    raw vectors; the codec is a footprint optimization, never a correctness dependency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import List, Sequence, Tuple

import numpy as np

log = logging.getLogger("aether_context.mpo")

#: Default retrieval-vector dimensionality the codec factors (the 256-dim embedding).
DEFAULT_DIM: int = 256
#: Default factorization grid for D=256 (4*4*4*4). More, smaller modes = more cores to share
#: the bond budget = better compression at a given rank. Must multiply to the vector dim.
DEFAULT_MODE_SHAPE: Tuple[int, ...] = (4, 4, 4, 4)
#: Default maximum bond (TT) rank. Caps every internal bond; lower = smaller + lossier.
DEFAULT_RANK: int = 4


def _factor_default(dim: int) -> Tuple[int, ...]:
    """A reasonable near-square factorization grid for ``dim`` (used when none is given)."""
    if dim == DEFAULT_DIM:
        return DEFAULT_MODE_SHAPE
    # Greedy small-factor split; falls back to a single mode for primes.
    modes: List[int] = []
    remaining = dim
    for p in (2, 3, 5, 7, 11, 13):
        while remaining % p == 0 and remaining > 1:
            modes.append(p)
            remaining //= p
    if remaining > 1:
        modes.append(remaining)
    return tuple(modes) if modes else (dim,)


@dataclass(frozen=True)
class TTVector:
    """A tensor-train-encoded vector: the ordered cores plus the grid they decompose.

    ``cores[k]`` has shape ``(r_k, mode_shape[k], r_{k+1})`` with ``r_0 == r_last == 1``.
    ``param_count`` is the total stored floats (the compressed size); ``recover`` contracts
    the chain back to the original-length vector.
    """

    cores: Tuple[np.ndarray, ...]
    mode_shape: Tuple[int, ...]

    @property
    def param_count(self) -> int:
        return int(sum(c.size for c in self.cores))


class MpoCodec:
    """Tensor-train vector codec: ``encode`` to cores, ``recover`` back. numpy-only, stateless.

    Construct with the vector ``dim``, an optional factorization ``mode_shape`` (must multiply
    to ``dim``), and a maximum bond ``rank``. The codec holds no per-vector state — it is safe
    to share one instance across a whole pool.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        *,
        mode_shape: Sequence[int] | None = None,
        rank: int = DEFAULT_RANK,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        shape = tuple(int(m) for m in (mode_shape if mode_shape is not None else _factor_default(dim)))
        if prod(shape) != dim:
            raise ValueError(
                f"mode_shape {shape} multiplies to {prod(shape)}, not dim={dim}"
            )
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.dim = dim
        self.mode_shape = shape
        self.rank = int(rank)

    # -- encode (TT-SVD) -------------------------------------------------------
    def encode(self, vector: np.ndarray) -> TTVector:
        """Compress a ``(dim,)`` vector into a rank-bounded tensor train via TT-SVD.

        Sequentially unfolds the reshaped tensor and truncates each bond to at most ``rank``
        singular values. Deterministic; the same vector always yields the same cores.
        """
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != self.dim:
            raise ValueError(f"vector has {v.size} elements, expected dim={self.dim}")
        cores: List[np.ndarray] = []
        c = v.reshape(self.mode_shape)
        r_prev = 1
        d = len(self.mode_shape)
        for k in range(d - 1):
            n_k = self.mode_shape[k]
            c = c.reshape(r_prev * n_k, -1)
            u, s, vt = np.linalg.svd(c, full_matrices=False)
            r = int(min(self.rank, s.size))
            u = u[:, :r]
            s = s[:r]
            vt = vt[:r, :]
            cores.append(u.reshape(r_prev, n_k, r).copy())
            c = (s[:, None] * vt)  # fold singular values into the carry (shape (r, rest))
            r_prev = r
        cores.append(c.reshape(r_prev, self.mode_shape[-1], 1).copy())
        return TTVector(cores=tuple(cores), mode_shape=self.mode_shape)

    # -- recover (contraction) -------------------------------------------------
    def recover(self, tt: TTVector) -> np.ndarray:
        """Contract a tensor train back into a ``(dim,)`` float32 vector (an approximation)."""
        res = tt.cores[0].reshape(tt.cores[0].shape[1], tt.cores[0].shape[2])  # (n0, r1)
        for core in tt.cores[1:]:
            r_prev, n_k, r_next = core.shape
            res = res.reshape(-1, r_prev) @ core.reshape(r_prev, n_k * r_next)
            res = res.reshape(-1, r_next)
        return res.reshape(-1).astype(np.float32)

    # -- accounting / fidelity -------------------------------------------------
    def compression_ratio(self, tt: TTVector | None = None) -> float:
        """Stored-floats ratio ``dim / param_count``. >1 means smaller than the raw vector.

        With no ``tt`` argument, returns the worst-case ratio for a full-rank encode (every
        bond saturated at ``rank``), the honest floor a caller can expect.
        """
        if tt is not None:
            return self.dim / max(1, tt.param_count)
        # Worst case: every bond hits the rank cap (bounded by the adjacent unfolding sizes).
        r_prev = 1
        params = 0
        left = 1
        for k, n_k in enumerate(self.mode_shape):
            if k < len(self.mode_shape) - 1:
                left *= n_k
                right = self.dim // left
                r = int(min(self.rank, left, right))
            else:
                r = 1
            params += r_prev * n_k * r
            r_prev = r
        return self.dim / max(1, params)

    def fidelity(self, vector: np.ndarray, tt: TTVector | None = None) -> float:
        """Cosine similarity between ``vector`` and its round-trip reconstruction in ``[-1,1]``."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        rec = np.asarray(self.recover(tt if tt is not None else self.encode(v)), dtype=np.float64)
        nv = float(np.linalg.norm(v))
        nr = float(np.linalg.norm(rec))
        if nv < 1e-12 or nr < 1e-12:
            return 0.0
        return float(np.dot(v, rec) / (nv * nr))

    # -- serde -----------------------------------------------------------------
    @staticmethod
    def tt_to_lists(tt: TTVector) -> dict:
        """Serialize a :class:`TTVector` to JSON-safe nested lists (for the pool sidecar)."""
        return {"mode_shape": list(tt.mode_shape), "cores": [c.tolist() for c in tt.cores]}

    @staticmethod
    def tt_from_lists(data: dict) -> TTVector:
        """Rebuild a :class:`TTVector` from :meth:`tt_to_lists` output."""
        cores = tuple(np.asarray(c, dtype=np.float64) for c in data["cores"])
        return TTVector(cores=cores, mode_shape=tuple(int(m) for m in data["mode_shape"]))

    def to_dict(self) -> dict:
        """Serialize the codec configuration (not per-vector data)."""
        return {"dim": self.dim, "mode_shape": list(self.mode_shape), "rank": self.rank}

    @classmethod
    def from_dict(cls, data: dict) -> "MpoCodec":
        return cls(int(data["dim"]), mode_shape=data.get("mode_shape"), rank=int(data.get("rank", DEFAULT_RANK)))


__all__ = [
    "MpoCodec",
    "TTVector",
    "DEFAULT_DIM",
    "DEFAULT_MODE_SHAPE",
    "DEFAULT_RANK",
]
