# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""MPO context chain — links the session's slices and assists retrieval.

Cosine/semantic search is the retrieval mechanism; this does not replace it. The MPO
(Matrix Product Operator) chain is a **coupling layer** over the slices: when cosine pulls an
entry slice, the chain pulls in the slices most coupled to it, **widening the working set with
connected context** (the thread) instead of isolated nearest-neighbors.

Coupling is ranked on **two session-local constants** — **cost** and **time** — and nothing
else. ``time`` is the slice's position in the session; ``cost`` is its token cost, discounted
when the slice is already cached (``cost/cache``). The two constants are lifted to small Fourier
features and contracted through a 2-site tensor train (the MPO) into a shared **chain manifold**;
coupling = semantic similarity gated by closeness on that manifold.

It is fixed, deterministic, numpy-only linear algebra (seeded cores, no training) and purely
additive: it only ever *adds* connected slices to a retrieval result, never blocks or replaces a
hit, so any failure degrades cleanly to plain cosine retrieval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger("aether_context.mpo")

#: Fourier features per axis (lifts each scalar constant to a smooth feature vector).
DEFAULT_FEATURE_DIM: int = 8
#: Tensor-train bond dimension between the two MPO sites.
DEFAULT_BOND: int = 4
#: Dimensionality of the chain embedding the MPO emits.
DEFAULT_EMBED_DIM: int = 4
#: How much cheaper a cached (resident) slice is to chain in: cost_eff = cost / (1 + bonus·cached).
DEFAULT_CACHE_BONUS: float = 1.0
#: Default coupled slices pulled per hop, and how many hops to follow the chain.
DEFAULT_CHAIN_WIDTH: int = 8
DEFAULT_CHAIN_HOPS: int = 1


@dataclass(frozen=True)
class ChainItem:
    """A retrieval candidate as the chain sees it: id + unit vector + the 2 constants.

    ``cost`` is the raw token cost, ``time`` the raw session position (both normalized over the
    candidate set inside :meth:`MpoChain.expand`). ``cached`` discounts the cost.
    """

    id: str
    vector: np.ndarray
    cost: float
    time: float
    cached: bool = False


def _fourier(x: float, p: int) -> np.ndarray:
    """Lift a scalar ``x`` (expected in [0,1]) to a ``p``-dim smooth Fourier feature vector."""
    half = max(1, p // 2)
    ks = np.arange(1, half + 1, dtype=np.float64)
    feats = np.concatenate([np.sin(np.pi * ks * x), np.cos(np.pi * ks * x)])
    if feats.size < p:  # pad odd p
        feats = np.concatenate([feats, np.zeros(p - feats.size)])
    return feats[:p]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MpoChain:
    """The 2-site MPO that couples slices on (cost, time) and expands a hit to its thread."""

    def __init__(
        self,
        *,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        bond: int = DEFAULT_BOND,
        embed_dim: int = DEFAULT_EMBED_DIM,
        seed: int = 0,
        cache_bonus: float = DEFAULT_CACHE_BONUS,
        width: int = DEFAULT_CHAIN_WIDTH,
        hops: int = DEFAULT_CHAIN_HOPS,
    ) -> None:
        rng = np.random.default_rng(seed)
        p = int(feature_dim)
        self.feature_dim = p
        self.bond = int(bond)
        self.embed_dim = int(embed_dim)
        self.cache_bonus = max(0.0, float(cache_bonus))
        self.width = max(1, int(width))
        self.hops = max(1, int(hops))
        # 2-site tensor train: site A consumes the time axis, site B the cost axis.
        self._core_a = (rng.standard_normal((1, p, self.bond)) * 0.5).astype(np.float64)
        self._core_b = (rng.standard_normal((self.bond, p, self.embed_dim)) * 0.5).astype(np.float64)

    # -- the MPO contraction ---------------------------------------------------
    def chain_embed(self, cost: float, time: float) -> np.ndarray:
        """Map the two constants to a unit chain embedding via the 2-site tensor train."""
        u = _fourier(float(time), self.feature_dim)   # time axis -> site A
        v = _fourier(float(cost), self.feature_dim)   # cost axis -> site B
        a = np.einsum("aib,i->ab", self._core_a, u)[0]      # (bond,)
        e = np.einsum("b,bjm,j->m", a, self._core_b, v)     # (embed_dim,)
        return _unit(e)

    # -- coupling --------------------------------------------------------------
    def coupling(self, vec_i, e_i, vec_h, e_h) -> float:
        """Semantic similarity gated by chain proximity, in roughly [0,1] for unit inputs."""
        sem = _cos(np.asarray(vec_i), np.asarray(vec_h))
        chain = _cos(e_i, e_h)
        return sem * (0.5 + 0.5 * chain)

    # -- expansion (assist the retrieval) --------------------------------------
    def expand(
        self,
        hit_ids: Sequence[str],
        candidates: Sequence[ChainItem],
        *,
        width: int | None = None,
        hops: int | None = None,
    ) -> list[str]:
        """Widen ``hit_ids`` with the candidates most coupled to them — follow the chain.

        Normalizes cost/time over ``candidates`` (cost discounted for cached items), embeds each
        on the chain manifold, then repeatedly pulls the top-``width`` most-coupled candidates to
        the current frontier for ``hops`` hops. Returns ordered ids: the hits first (in their
        given order), then the coupled slices in pull order, de-duplicated.
        """
        items = list(candidates)
        if not items:
            return list(hit_ids)
        width = self.width if width is None else max(1, int(width))
        hops = self.hops if hops is None else max(1, int(hops))

        # Normalize the two constants over the candidate set; discount cached cost.
        costs = np.array(
            [it.cost / (1.0 + self.cache_bonus * (1.0 if it.cached else 0.0)) for it in items],
            dtype=np.float64,
        )
        times = np.array([it.time for it in items], dtype=np.float64)
        cost_n = _minmax(costs)
        time_n = _minmax(times)
        embeds = {it.id: self.chain_embed(cost_n[i], time_n[i]) for i, it in enumerate(items)}
        by_id = {it.id: it for it in items}

        present_hits = [h for h in hit_ids if h in by_id]
        if not present_hits:
            present_hits = [items[0].id]
        selected: list[str] = []
        seen: set[str] = set()
        for h in present_hits:
            if h not in seen:
                selected.append(h)
                seen.add(h)
        frontier = list(selected)

        for _ in range(hops):
            scored: list[tuple[float, str]] = []
            for it in items:
                if it.id in seen:
                    continue
                best = max(
                    self.coupling(it.vector, embeds[it.id], by_id[h].vector, embeds[h])
                    for h in frontier
                )
                scored.append((best, it.id))
            if not scored:
                break
            scored.sort(key=lambda p: p[0], reverse=True)
            picked = [sid for _, sid in scored[:width]]
            for sid in picked:
                if sid not in seen:
                    selected.append(sid)
                    seen.add(sid)
            frontier = picked
        return selected


def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]; a flat array maps to 0.5 (no spurious gradient)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


__all__ = [
    "MpoChain",
    "ChainItem",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_BOND",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_CACHE_BONUS",
    "DEFAULT_CHAIN_WIDTH",
    "DEFAULT_CHAIN_HOPS",
]
