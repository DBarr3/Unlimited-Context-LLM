# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Domain-agnostic Matrix Product Operator (MPO) core — a learned read-path re-ranker.

This is a faithful, dependency-free port of the AETHER-ATLAS MPO
(``aether_atlas/mpo/tensor_core.py`` + ``polarity_sieve.py``). It is reproduced here — not
imported — so Unlimited Context carries no dependency on the atlas package while still using
**the same operator**: identical physical dims, bond chain, contraction, contrastive loss,
and CUSUM tracker. Keeping the math identical is deliberate — the session re-ranker and the
main atlas must remain correlated on one operator definition so they can later be bridged
(shared cores format) rather than diverging into two independent atlases.

CORRELATION INVARIANT: ``PHYSICAL_DIMS``, ``BOND_CHAIN``, ``ALPHA_DEFAULT``, the contraction
einsums, the loss formula, and the CUSUM constants MUST equal
``aether_atlas.mpo.tensor_core``. ``tests/test_mpo.py::TestConstantsDoNotDriftFromAtlas``
pins them.

The only adaptation from atlas: atlas derives polarity from a cell's ``edge``/``source``/
``lifecycle`` via ``_polarity(cell)``; here the polarity is **supplied by the caller** on each
:class:`MpoCandidate` (Unlimited Context derives it from session retention signals, not from a
trading edge). Everything downstream of polarity is verbatim.

The MPO is exclusively a read-path accelerator — it never touches the write path, the pool's
storage, or the witness.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple

import numpy as np

log = logging.getLogger("aether_context.mpo")

# ── operator definition (MUST match aether_atlas.mpo.tensor_core) ──────────────
PHYSICAL_DIMS = {"model": 16, "skill": 32, "prompt": 16, "time": 4}
BOND_CHAIN = (2, 8, 8, 2, 1)
ALPHA_DEFAULT = 1.0
#: Numerical guard only (NOT part of the loss): caps the global gradient norm per step.
MAX_GRAD_NORM = 10.0

_AXES = ("model", "skill", "prompt", "time")


class MpoCandidate(Protocol):
    """The structural contract the MPO core consumes (one per retrieval candidate).

    ``feature_code`` is the similarity-preserving vector the kernel sees (here, a slice's
    256-dim retrieval embedding). ``polarity`` is +1 (load-bearing positive), -1 (negative /
    faded), or 0 (unjudged — skipped by the sweep). ``confidence`` is the [0,1] weight.
    ``id`` is a stable identifier returned in the result for re-ordering.
    """

    feature_code: object
    polarity: int
    confidence: float
    id: str


def _embed_feature_code(feature_code) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed a feature code into the four physical axes by a cyclic read (verbatim from atlas).

    Axis k of size d takes ``fc[(offset + j) % len(fc)]``, offset advancing by d per axis, so
    EVERY axis receives signal regardless of code length (256 here, 8/64 in atlas producers).
    Deterministic and similarity-preserving.
    """
    fc = np.asarray(feature_code, dtype=np.float64).ravel()
    segments = []
    offset = 0
    for name in _AXES:
        d = PHYSICAL_DIMS[name]
        if fc.size == 0:
            segments.append(np.zeros(d, dtype=np.float64))
        else:
            idx = (offset + np.arange(d)) % fc.size
            segments.append(fc[idx])
        offset += d
    return tuple(segments)


def _embed_batch(cands) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per = [_embed_feature_code(c.feature_code) for c in cands]
    return tuple(np.stack([p[k] for p in per]) for k in range(4))


def _init_core(rng: np.random.Generator, l: int, r: int, p: int) -> np.ndarray:
    return rng.standard_normal((l, p, r)).astype(np.float64) * 0.01


def _init_cores(rng: np.random.Generator) -> List[np.ndarray]:
    return [
        _init_core(rng, BOND_CHAIN[0], BOND_CHAIN[1], PHYSICAL_DIMS["model"]),
        _init_core(rng, BOND_CHAIN[1], BOND_CHAIN[2], PHYSICAL_DIMS["skill"]),
        _init_core(rng, BOND_CHAIN[2], BOND_CHAIN[3], PHYSICAL_DIMS["prompt"]),
        _init_core(rng, BOND_CHAIN[3], BOND_CHAIN[4], PHYSICAL_DIMS["time"]),
    ]


@dataclass(frozen=True)
class MpoForwardResult:
    s_pos: np.ndarray
    s_neg: np.ndarray
    cell_ids: list
    delta_norm: float
    n_pos: int
    n_neg: int
    scores: list = field(default_factory=list)


class MpoCore:
    """Tensor-train MPO with contrastive loss. Verbatim atlas math; candidate-driven polarity."""

    def __init__(self, alpha: float = ALPHA_DEFAULT, seed: Optional[int] = 0):
        self.alpha = alpha
        self.cores = _init_cores(np.random.default_rng(seed))
        self._n_updates = 0

    # ── contraction ──────────────────────────────────────────────────
    def _axis_mats(self, xm, xs, xp, xt) -> List[np.ndarray]:
        c0, c1, c2, c3 = self.cores
        return [
            np.einsum("aib,ni->nab", c0, xm, optimize=True),
            np.einsum("aib,ni->nab", c1, xs, optimize=True),
            np.einsum("aib,ni->nab", c2, xp, optimize=True),
            np.einsum("aib,ni->nab", c3, xt, optimize=True),
        ]

    @staticmethod
    def _contract(mats: List[np.ndarray]) -> np.ndarray:
        a1, a2, a3, a4 = mats
        m = np.einsum("nab,nbc->nac", a1, a2, optimize=True)
        m = np.einsum("nab,nbc->nac", m, a3, optimize=True)
        out = np.einsum("nab,nbc->nac", m, a4, optimize=True)
        return out[:, 0, 0]

    def forward(self, cands) -> MpoForwardResult:
        kept, pols = [], []
        for c in cands:
            p = int(c.polarity)
            if p == 0:
                continue
            kept.append(c)
            pols.append(p)
        if not kept:
            return MpoForwardResult(
                s_pos=np.array([0.0]), s_neg=np.array([0.0]), cell_ids=[],
                delta_norm=0.0, n_pos=0, n_neg=0, scores=[])
        xm, xs, xp, xt = _embed_batch(kept)
        scalars = self._contract(self._axis_mats(xm, xs, xp, xt))
        conf = np.array([float(c.confidence) for c in kept], dtype=np.float64)
        pol = np.array(pols, dtype=np.float64)
        weighted = scalars * conf
        s_pos = np.array([float(weighted[pol > 0].sum())]) if (pol > 0).any() else np.array([0.0])
        s_neg = np.array([float(weighted[pol < 0].sum())]) if (pol < 0).any() else np.array([0.0])
        return MpoForwardResult(
            s_pos=s_pos, s_neg=s_neg,
            cell_ids=[c.id for c in kept],
            delta_norm=float(np.sum((s_pos - s_neg) ** 2)),
            n_pos=int((pol > 0).sum()), n_neg=int((pol < 0).sum()),
            scores=[float(w) for w in weighted])

    # ── training ─────────────────────────────────────────────────────
    def _loss_and_grads(self, pos_x, neg_x):
        n_pos = pos_x[0].shape[0]
        n_neg = neg_x[0].shape[0] if neg_x is not None else 0
        if n_neg:
            bundles = [np.concatenate([p, q]) for p, q in zip(pos_x[:4], neg_x[:4])]
            conf = np.concatenate([pos_x[4], neg_x[4]])
            sign = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])
        else:
            bundles = list(pos_x[:4])
            conf = pos_x[4]
            sign = np.ones(n_pos)
        n = bundles[0].shape[0]
        mats = self._axis_mats(*bundles)
        a1, a2, a3, a4 = mats
        scalars = self._contract(mats)

        l1 = np.tile(np.array([1.0, 0.0]), (n, 1))
        l2 = a1[:, 0, :]
        l3 = np.einsum("nb,nbc->nc", l2, a2, optimize=True)
        l4 = np.einsum("nb,nbc->nc", l3, a3, optimize=True)
        r4 = np.ones((n, 1))
        r3 = a4[:, :, 0]
        r2 = np.einsum("nbc,nc->nb", a3, r3, optimize=True)
        r1 = np.einsum("nbc,nc->nb", a2, r2, optimize=True)
        lefts, rights = (l1, l2, l3, l4), (r1, r2, r3, r4)

        target = sum(b[:n_pos].sum(axis=1) for b in bundles)
        diff = target - scalars[:n_pos]
        recon_loss = float(np.sum(diff ** 2))
        w = np.zeros(n)
        w[:n_pos] = -2.0 * diff

        sep_loss = 0.0
        if n_neg:
            weighted = scalars * conf
            gap = float(weighted[:n_pos].sum() - weighted[n_pos:].sum())
            sep_loss = -self.alpha * gap * gap
            w += -2.0 * self.alpha * gap * conf * sign

        grads = [
            np.einsum("na,ni,nb,n->aib", lefts[k], bundles[k], rights[k], w, optimize=True)
            for k in range(4)
        ]
        return recon_loss, sep_loss, grads

    def train(self, cands, lr: float = 0.01, n_iter: int = 10) -> float:
        pos = [c for c in cands if int(c.polarity) > 0]
        neg = [c for c in cands if int(c.polarity) < 0]
        if not pos:
            return 0.0
        pos_x = _embed_batch(pos) + (np.array([float(c.confidence) for c in pos]),)
        neg_x = (_embed_batch(neg) + (np.array([float(c.confidence) for c in neg]),)) if neg else None

        for _ in range(n_iter):
            _, _, grads = self._loss_and_grads(pos_x, neg_x)
            gnorm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))
            if gnorm > MAX_GRAD_NORM:
                grads = [g * (MAX_GRAD_NORM / gnorm) for g in grads]
            for i in range(len(self.cores)):
                self.cores[i] -= lr * grads[i]
            self._n_updates += 1

        recon_loss, sep_loss, _ = self._loss_and_grads(pos_x, neg_x)
        return float(recon_loss + sep_loss)

    # ── serde ────────────────────────────────────────────────────────
    def to_dict(self):
        return {"alpha": self.alpha, "n_updates": self._n_updates,
                "cores": [c.tolist() for c in self.cores],
                "bond_chain": list(BOND_CHAIN), "physical_dims": PHYSICAL_DIMS}

    @classmethod
    def from_dict(cls, data):
        mpo = cls(alpha=data.get("alpha", ALPHA_DEFAULT))
        mpo.cores = [np.array(c, dtype=np.float64) for c in data["cores"]]
        mpo._n_updates = data.get("n_updates", 0)
        return mpo


class CusumDeltaTracker:
    """Two-sided CUSUM regime-shift detector over ``delta_norm`` (verbatim from atlas)."""

    def __init__(self, threshold: float = 3.0, drift: float = 0.5):
        self.mu: float = 0.0
        self.cusum_high: float = 0.0
        self.cusum_low: float = 0.0
        self.threshold = threshold
        self.drift = drift
        self._n: int = 0

    def observe(self, delta_norm: float) -> bool:
        if self._n == 0:
            self.mu = delta_norm
            self._n = 1
            return False
        self._n += 1
        diff = delta_norm - self.mu - self.drift
        self.cusum_high = max(0.0, self.cusum_high + diff)
        self.cusum_low = max(0.0, self.cusum_low - diff)
        alpha = 1.0 / self._n
        self.mu = self.mu + alpha * (delta_norm - self.mu)
        return self.cusum_high > self.threshold or self.cusum_low > self.threshold

    def reset(self) -> None:
        self.mu = 0.0
        self.cusum_high = 0.0
        self.cusum_low = 0.0
        self._n = 0

    def to_dict(self) -> dict:
        return {"mu": self.mu, "cusum_high": self.cusum_high, "cusum_low": self.cusum_low,
                "threshold": self.threshold, "drift": self.drift, "n": self._n}

    @classmethod
    def from_dict(cls, data: dict) -> "CusumDeltaTracker":
        t = cls(threshold=data.get("threshold", 3.0), drift=data.get("drift", 0.5))
        t.mu = data.get("mu", 0.0)
        t.cusum_high = data.get("cusum_high", 0.0)
        t.cusum_low = data.get("cusum_low", 0.0)
        t._n = data.get("n", 0)
        return t


__all__ = [
    "MpoCandidate", "MpoCore", "MpoForwardResult", "CusumDeltaTracker",
    "PHYSICAL_DIMS", "BOND_CHAIN", "ALPHA_DEFAULT", "MAX_GRAD_NORM",
]
