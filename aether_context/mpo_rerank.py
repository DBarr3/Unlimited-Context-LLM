# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""MPO bolt-on re-ranker — the opt-in second stage of retrieval.

Retrieval is two stages when enabled: a cheap cosine recall (``ContextPool.search`` top-M)
feeds this learned MPO re-ranker, which contracts each candidate through the tensor-train
operator (:mod:`aether_context.mpo`) and re-orders by ``|score|``. A CUSUM staleness guard
falls back to the original cosine order whenever the operator drifts from ground truth — so
the re-rank is an *optimization*, never a correctness dependency (engine design law 3).

The operator's cores persist to ``<pool_dir>/mpo.json``, so the learned re-ranker survives a
close + reopen — the "persistent session". This module is the only place that knows about
:class:`~aether_context.context_pool.Slice`; the core is domain-agnostic.

Polarity (Slice -> {+1,-1,0}) comes from session retention signals, not a trading edge:
  * a remembered fact (``meta["kind"]=="fact"``) or a witness-salient slice (score >= 0.60)
    is **+1** (load-bearing positive);
  * a stale/evicted slice, or one faded below ``_NEG_FLOOR`` (0.15), is **-1**;
  * anything in between is **0** (unjudged — skipped by the sweep, like an atlas synthetic prior).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aether_context._log import get_logger
from aether_context.context_pool import Slice
from aether_context.encoder import ENCODER_VERSION
from aether_context.mpo import CusumDeltaTracker, MpoCore

logger = get_logger(__name__)

#: Witness score at/above which a slice is load-bearing positive (mirrors witness SALIENT_THRESHOLD).
_POS_THRESHOLD: float = 0.60
#: Witness score below which a slice is treated as faded/negative.
_NEG_FLOOR: float = 0.15
#: Persistence format version for ``mpo.json``.
PERSIST_VERSION: int = 1
#: Default cosine over-scan factor: recall M = k * overscan candidates before re-ranking.
DEFAULT_OVERSCAN: int = 4


def _polarity_for(score: float, meta: dict[str, Any]) -> int:
    """Map a slice's retention ``score`` + ``meta`` to MPO polarity (+1 / -1 / 0).

    An explicit stale/evicted marker is a HARD negative that overrides a high score (mirrors
    the atlas lifecycle gate, where REFUTED/STALE/EVICTED is -1 regardless of edge). Otherwise
    a fact or witness-salient slice is +1, a faded slice (< _NEG_FLOOR) is -1, the rest 0.
    """
    if meta.get("stale") or meta.get("kind") == "evicted":
        return -1
    if meta.get("kind") == "fact" or score >= _POS_THRESHOLD:
        return +1
    if score < _NEG_FLOOR:
        return -1
    return 0


@dataclass(frozen=True)
class _SliceCandidate:
    """A :class:`~aether_context.mpo.MpoCandidate` built from a :class:`Slice`."""

    feature_code: np.ndarray
    polarity: int
    confidence: float
    id: str


def _to_candidate(sl: Slice, score: float) -> _SliceCandidate:
    return _SliceCandidate(
        feature_code=sl.vector,
        polarity=_polarity_for(score, sl.meta),
        confidence=float(min(1.0, max(0.0, score))),
        id=sl.id,
    )


class MpoReranker:
    """Guarded, persistent MPO re-ranker over cosine candidates. Fail-soft throughout."""

    def __init__(
        self,
        core: MpoCore | None = None,
        *,
        cusum: CusumDeltaTracker | None = None,
        cusum_threshold: float = 3.0,
        overscan: int = DEFAULT_OVERSCAN,
    ) -> None:
        self.core = core or MpoCore()
        self.cusum = cusum or CusumDeltaTracker(threshold=cusum_threshold)
        self.overscan = max(1, int(overscan))
        self._stale = False

    @property
    def is_stale(self) -> bool:
        """Whether the CUSUM guard has fired (operator drifted; serving cosine fallback)."""
        return self._stale

    def recall_width(self, k: int) -> int:
        """How many cosine candidates to pull before re-ranking down to ``k``."""
        return max(int(k), int(k) * self.overscan)

    # -- the read-path move ----------------------------------------------------
    def rerank(
        self,
        query_vec: np.ndarray,
        slices: list[Slice],
        k: int,
        score_of: Callable[[str], float],
    ) -> list[Slice]:
        """Re-rank cosine ``slices`` by the MPO operator; fall back to cosine order if stale.

        ``query_vec`` is accepted for interface symmetry and future query-conditioning; the
        current operator ranks candidates by their learned contraction scalar (the query
        already shaped the cosine recall upstream). Returns at most ``k`` slices. Any error,
        too-few candidates, or a CUSUM stale verdict yields the original cosine ``slices[:k]``.
        """
        if k <= 0:
            return []
        if len(slices) < 2:
            return slices[:k]
        try:
            cands = [_to_candidate(sl, float(score_of(sl.id))) for sl in slices]
            result = self.core.forward(cands)
            if self.cusum.observe(result.delta_norm):
                self._stale = True
                logger.warning(
                    "STALE_SERVED: MPO bonds diverged (delta=%.4f); serving cosine order",
                    result.delta_norm,
                )
                return slices[:k]
            self._stale = False
            if not result.scores:
                return slices[:k]
            by_id = {sl.id: i for i, sl in enumerate(slices)}
            score_by_id = dict(zip(result.cell_ids, result.scores))
            # Rank swept candidates by |score| desc; P=0 (unswept) keep cosine order, appended.
            swept = [sl for sl in slices if sl.id in score_by_id]
            unswept = [sl for sl in slices if sl.id not in score_by_id]
            swept.sort(key=lambda sl: (-abs(score_by_id[sl.id]), by_id[sl.id]))
            return (swept + unswept)[:k]
        except Exception as exc:  # noqa: BLE001 - fail-soft: re-rank is an optimization
            logger.warning("MPO rerank failed (%s); serving cosine order", exc)
            return slices[:k]

    # -- off-hot-path learning -------------------------------------------------
    def train(
        self,
        slices: list[Slice],
        score_of: Callable[[str], float],
        *,
        lr: float = 0.01,
        n_iter: int = 20,
    ) -> float:
        """Train the operator on a session's judged slices (those with P != 0). Fail-soft.

        Resets the CUSUM and clears the stale flag when any positive cell trained — the
        operator has just been re-fit to ground truth. Returns the post-train loss (0.0 if
        nothing trainable or on error).
        """
        try:
            cands = [_to_candidate(sl, float(score_of(sl.id))) for sl in slices]
            loss = self.core.train(cands, lr=lr, n_iter=n_iter)
            if any(c.polarity > 0 for c in cands):
                self.cusum.reset()
                self._stale = False
            return loss
        except Exception as exc:  # noqa: BLE001 - fail-soft
            logger.warning("MPO train failed (%s); operator left unchanged", exc)
            return 0.0

    # -- persistence (the persistent session) ----------------------------------
    def save(self, path: "str | Path") -> None:
        """Serialize cores + CUSUM to ``path`` atomically. Fail-soft (logs, never raises)."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": PERSIST_VERSION,
                "encoder_version": ENCODER_VERSION,
                "core": self.core.to_dict(),
                "cusum": self.cusum.to_dict(),
            }
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, p)
        except Exception as exc:  # noqa: BLE001 - fail-soft: persistence is best-effort
            logger.warning("MPO save failed (%s); operator not persisted this run", exc)

    @classmethod
    def load_or_new(cls, path: "str | Path", **kw: Any) -> "MpoReranker":
        """Load a persisted re-ranker from ``path``, or return a fresh one.

        A missing / corrupt file, or one written under a different ``encoder_version`` (the
        cores were trained on an incompatible embedding scheme), yields a fresh operator — the
        re-ranker never blocks a session on a bad sidecar.
        """
        p = Path(path)
        if not p.exists():
            return cls(**kw)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("encoder_version") != ENCODER_VERSION:
                logger.warning(
                    "MPO sidecar at %s has encoder_version=%r != %r; starting fresh operator",
                    p, data.get("encoder_version"), ENCODER_VERSION,
                )
                return cls(**kw)
            core = MpoCore.from_dict(data["core"])
            cusum = CusumDeltaTracker.from_dict(data.get("cusum", {}))
            return cls(core=core, cusum=cusum, **kw)
        except Exception as exc:  # noqa: BLE001 - fail-soft: corrupt sidecar -> fresh
            logger.warning("MPO load failed for %s (%s); starting fresh operator", p, exc)
            return cls(**kw)


__all__ = ["MpoReranker", "DEFAULT_OVERSCAN", "PERSIST_VERSION"]
