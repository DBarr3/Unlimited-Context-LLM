# MPO Bolt-On Re-Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, persistent, fail-soft MPO learned re-ranker to the Unlimited Context retrieval path, with math cloned faithfully from AETHER-ATLAS so the two systems share one operator definition.

**Architecture:** Two new numpy-only modules — `aether_context/mpo.py` (ported `MpoCore` + `CusumDeltaTracker`, generic over an `MpoCandidate` protocol) and `aether_context/mpo_rerank.py` (`MpoReranker`: `Slice`→candidate adapter, guarded `rerank`, `train`, persistence). `Session` gains an opt-in `rerank="mpo"` second stage: cosine recall top-M → MPO re-rank top-k, with a CUSUM staleness fallback to cosine. Cores persist to `<pool_dir>/mpo.json`. Default `rerank="off"` = byte-identical to today.

**Tech Stack:** Python 3.10+, numpy only, pytest. No new runtime dependency. No import of `aether_atlas`.

**Spec:** `docs/superpowers/specs/2026-06-14-mpo-bolt-on-reranker-design.md`

---

## File Structure

- **Create** `aether_context/mpo.py` — domain-agnostic MPO core (TT decomposition, contrastive loss, CUSUM). Reads `MpoCandidate.{feature_code, polarity, confidence, id}`. No `Slice`/`Session` knowledge.
- **Create** `aether_context/mpo_rerank.py` — the bolt-on: `_SliceCandidate`, `MpoReranker` (adapter, `rerank`, `train`, `save`/`load_or_new`). Only this file knows about `Slice`.
- **Create** `tests/test_mpo.py` — ported atlas math tests + constants drift pin.
- **Create** `tests/test_mpo_rerank.py` — adapter, rerank, fallback, persistence, fail-soft.
- **Modify** `aether_context/session.py` — `rerank`/`rerank_overscan` params, `_cold_retrieve` two-stage, `close()` train+save, `status_dict()` fields.
- **Modify** `tests/test_session.py` — integration: on/off parity, persistence reopen, invalid arg.
- **Modify** `CHANGELOG.md` — one line under Unreleased.

Constants confirmed from the codebase: errors live in `aether_context.errors` (`AetherContextError`, `PoolBudgetError`); logger via `aether_context._log.get_logger`; encoder version is `aether_context.encoder.ENCODER_VERSION` (`"static_v1"`); witness `SALIENT_THRESHOLD = 0.60`.

---

### Task 1: MPO core module (`aether_context/mpo.py`)

**Files:**
- Create: `aether_context/mpo.py`
- Test: `tests/test_mpo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mpo.py`:

```python
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo — ported MPO core, polarity passthrough, CUSUM."""
from dataclasses import dataclass

import numpy as np
import pytest

from aether_context.mpo import (
    ALPHA_DEFAULT,
    BOND_CHAIN,
    PHYSICAL_DIMS,
    CusumDeltaTracker,
    MpoCore,
)


@dataclass(frozen=True)
class _Cand:
    """Minimal MpoCandidate for tests."""

    feature_code: tuple
    polarity: int
    confidence: float
    id: str


def _c(polarity, feature_code=(1.0, 0.0, 0.5, 0.1), confidence=0.8, cid="x"):
    return _Cand(feature_code=feature_code, polarity=polarity, confidence=confidence, id=cid)


class TestConstantsDoNotDriftFromAtlas:
    """Pin the operator definition to the atlas values (no atlas import — literal pin)."""

    def test_physical_dims(self):
        assert PHYSICAL_DIMS == {"model": 16, "skill": 32, "prompt": 16, "time": 4}

    def test_bond_chain(self):
        assert BOND_CHAIN == (2, 8, 8, 2, 1)

    def test_alpha(self):
        assert ALPHA_DEFAULT == 1.0


class TestForward:
    def test_forward_empty(self):
        r = MpoCore().forward([])
        assert r.n_pos == 0 and r.n_neg == 0
        assert float(r.s_pos[0]) == 0.0 and float(r.s_neg[0]) == 0.0
        assert r.scores == []

    def test_forward_pos_only(self):
        mpo = MpoCore()
        r = mpo.forward([_c(+1, cid="a"), _c(+1, feature_code=(1.0, 1.0, 0.8, 0.2), cid="b")])
        assert r.n_pos == 2 and r.n_neg == 0
        assert r.cell_ids == ["a", "b"]
        assert len(r.scores) == 2
        assert r.delta_norm >= 0.0

    def test_forward_mixed(self):
        r = MpoCore().forward([_c(+1, cid="a"), _c(-1, feature_code=(0.0, 1.0, 0.0, 0.5), cid="b")])
        assert r.n_pos == 1 and r.n_neg == 1
        assert len(r.cell_ids) == 2

    def test_forward_skips_zero_polarity(self):
        r = MpoCore().forward([_c(+1, cid="a"), _c(0, cid="z")])
        assert r.n_pos == 1
        assert r.cell_ids == ["a"]


class TestTrain:
    def test_train_reduces_loss(self):
        mpo = MpoCore()
        cells = [
            _c(+1, feature_code=(1.0, 0.0, 0.5, 0.1), confidence=1.0, cid="a"),
            _c(+1, feature_code=(0.5, 1.0, 0.3, 0.2), confidence=1.0, cid="b"),
        ]
        before = mpo.train(cells, lr=0.01, n_iter=0)
        after = mpo.train(cells, lr=0.01, n_iter=10)
        assert after < before or before == 0.0

    def test_train_no_positive_returns_zero(self):
        assert MpoCore().train([_c(-1)], lr=0.01, n_iter=5) == 0.0

    def test_train_mixed_no_crash(self):
        loss = MpoCore().train(
            [_c(+1, cid="a"), _c(-1, feature_code=(0.0, 1.0, 0.0, 0.5), cid="b")],
            lr=0.01, n_iter=5,
        )
        assert isinstance(loss, float)


class TestSerialization:
    def test_roundtrip(self):
        mpo = MpoCore()
        mpo.train([_c(+1, cid="a")], lr=0.01, n_iter=3)
        data = mpo.to_dict()
        mpo2 = MpoCore.from_dict(data)
        assert len(mpo2.cores) == len(mpo.cores)
        assert mpo2._n_updates == mpo._n_updates
        for c1, c2 in zip(mpo.cores, mpo2.cores):
            assert c1.shape == c2.shape
            assert np.allclose(c1, c2)


class TestCusum:
    def test_stable_stream_does_not_fire(self):
        cusum = CusumDeltaTracker(threshold=3.0, drift=0.5)
        fired = [cusum.observe(1.0) for _ in range(10)]
        assert not any(fired)

    def test_shift_fires(self):
        cusum = CusumDeltaTracker(threshold=3.0, drift=0.5)
        cusum.observe(1.0)
        fired = any(cusum.observe(100.0) for _ in range(5))
        assert fired

    def test_reset_clears(self):
        cusum = CusumDeltaTracker(threshold=3.0, drift=0.5)
        cusum.observe(1.0)
        cusum.observe(100.0)
        cusum.reset()
        assert cusum.cusum_high == 0.0 and cusum.cusum_low == 0.0 and cusum._n == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mpo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aether_context.mpo'`

- [ ] **Step 3: Write the implementation**

Create `aether_context/mpo.py` (math ported verbatim from `aether_atlas/mpo/tensor_core.py` + `polarity_sieve.py`; polarity is read off the candidate, not computed from edge/lifecycle):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mpo.py -q`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add aether_context/mpo.py tests/test_mpo.py
git commit -m "feat: port atlas MPO core (TT re-ranker + CUSUM) into aether_context, candidate-driven polarity"
```

---

### Task 2: Re-ranker bolt-on (`aether_context/mpo_rerank.py`)

**Files:**
- Create: `aether_context/mpo_rerank.py`
- Test: `tests/test_mpo_rerank.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mpo_rerank.py`:

```python
# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo_rerank — Slice adapter, guarded rerank, persistence."""
import numpy as np
import pytest

from aether_context.context_pool import Slice
from aether_context.encoder import ENCODER_VERSION
from aether_context.mpo_rerank import MpoReranker, _polarity_for


def _slice(cid, vec, score=0.5, meta=None):
    return Slice(id=cid, session="s", vector=np.asarray(vec, dtype=np.float32),
                 text=cid, tokens=1, meta=meta or {}, score=score)


def _vec(seed, dim=256):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestPolarityMapping:
    def test_fact_is_positive(self):
        assert _polarity_for(score=0.1, meta={"kind": "fact"}) == +1

    def test_salient_is_positive(self):
        assert _polarity_for(score=0.7, meta={}) == +1  # >= 0.60

    def test_stale_is_negative(self):
        assert _polarity_for(score=0.9, meta={"stale": True}) == -1

    def test_faded_is_negative(self):
        assert _polarity_for(score=0.05, meta={}) == -1  # < 0.15

    def test_neutral_is_zero(self):
        assert _polarity_for(score=0.3, meta={}) == 0


class TestRerank:
    def test_fewer_than_two_returned_unchanged(self):
        rr = MpoReranker()
        one = [_slice("a", _vec(1))]
        assert rr.rerank(_vec(9), one, k=8, score_of=lambda i: 0.7) == one

    def test_empty_returned_unchanged(self):
        rr = MpoReranker()
        assert rr.rerank(_vec(9), [], k=8, score_of=lambda i: 0.7) == []

    def test_changes_order_after_training(self):
        rr = MpoReranker()
        cands = [_slice("a", _vec(1), score=0.7), _slice("b", _vec(2), score=0.7),
                 _slice("c", _vec(3), score=0.7)]
        rr.train(cands, score_of=lambda i: 0.7, lr=0.05, n_iter=30)
        out = rr.rerank(_vec(9), cands, k=3, score_of=lambda i: 0.7)
        assert {s.id for s in out} == {"a", "b", "c"}  # same set, possibly reordered
        assert len(out) == 3

    def test_truncates_to_k(self):
        rr = MpoReranker()
        cands = [_slice(c, _vec(i), score=0.7) for i, c in enumerate("abcd")]
        out = rr.rerank(_vec(9), cands, k=2, score_of=lambda i: 0.7)
        assert len(out) == 2

    def test_stale_falls_back_to_cosine_order(self):
        rr = MpoReranker(cusum_threshold=0.0)  # threshold 0 -> any shift fires
        cands = [_slice("a", _vec(1), score=0.7), _slice("b", _vec(2), score=0.7)]
        # prime CUSUM with one observation, then a divergent one fires stale
        rr.rerank(_vec(9), cands, k=2, score_of=lambda i: 0.7)
        out = rr.rerank(_vec(9), cands, k=2, score_of=lambda i: 5.0)
        assert [s.id for s in out] == ["a", "b"]  # original cosine order preserved
        assert rr.is_stale

    def test_failsoft_on_core_error(self, monkeypatch):
        rr = MpoReranker()
        cands = [_slice("a", _vec(1), score=0.7), _slice("b", _vec(2), score=0.7)]

        def boom(_):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(rr.core, "forward", boom)
        out = rr.rerank(_vec(9), cands, k=2, score_of=lambda i: 0.7)
        assert [s.id for s in out] == ["a", "b"]  # cosine order, no raise


class TestPersistence:
    def test_save_then_load(self, tmp_path):
        path = tmp_path / "mpo.json"
        rr = MpoReranker()
        rr.train([_slice("a", _vec(1), score=0.7), _slice("b", _vec(2), score=0.7)],
                 score_of=lambda i: 0.7, lr=0.05, n_iter=10)
        rr.save(path)
        assert path.exists()
        rr2 = MpoReranker.load_or_new(path)
        assert rr2.core._n_updates == rr.core._n_updates
        for c1, c2 in zip(rr.core.cores, rr2.core.cores):
            assert np.allclose(c1, c2)

    def test_missing_file_is_fresh(self, tmp_path):
        rr = MpoReranker.load_or_new(tmp_path / "nope.json")
        assert rr.core._n_updates == 0

    def test_corrupt_file_is_fresh(self, tmp_path):
        path = tmp_path / "mpo.json"
        path.write_text("{ not json", encoding="utf-8")
        rr = MpoReranker.load_or_new(path)
        assert rr.core._n_updates == 0

    def test_encoder_version_mismatch_is_fresh(self, tmp_path):
        path = tmp_path / "mpo.json"
        rr = MpoReranker()
        rr.train([_slice("a", _vec(1), score=0.7)], score_of=lambda i: 0.7, n_iter=5)
        rr.save(path)
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        data["encoder_version"] = "something_else"
        path.write_text(json.dumps(data), encoding="utf-8")
        rr2 = MpoReranker.load_or_new(path)
        assert rr2.core._n_updates == 0  # mismatched embedding scheme -> cold start

    def test_saved_encoder_version_is_current(self, tmp_path):
        path = tmp_path / "mpo.json"
        MpoReranker().save(path)
        import json
        assert json.loads(path.read_text(encoding="utf-8"))["encoder_version"] == ENCODER_VERSION
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mpo_rerank.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aether_context.mpo_rerank'`

- [ ] **Step 3: Write the implementation**

Create `aether_context/mpo_rerank.py`:

```python
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
    """Map a slice's retention ``score`` + ``meta`` to MPO polarity (+1 / -1 / 0)."""
    if meta.get("kind") == "fact" or score >= _POS_THRESHOLD:
        return +1
    if meta.get("stale") or meta.get("kind") == "evicted" or score < _NEG_FLOOR:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mpo_rerank.py -q`
Expected: PASS

> Note: `test_changes_order_after_training` asserts the **set** is preserved (not a strict
> reorder) — with random init the contraction may or may not flip order; the meaningful
> guarantees (truncation, fallback, fail-soft, persistence) are asserted explicitly.

- [ ] **Step 5: Commit**

```bash
git add aether_context/mpo_rerank.py tests/test_mpo_rerank.py
git commit -m "feat: MPO re-ranker bolt-on (Slice adapter, CUSUM-guarded rerank, persistence)"
```

---

### Task 3: Wire the re-ranker into `Session`

**Files:**
- Modify: `aether_context/session.py`
- Test: `tests/test_session.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_session.py` (the file already imports `Session`; add `import numpy as np`
and `import json` at the top if not present):

```python
from pathlib import Path as _Path


class TestMpoRerank:
    def test_off_by_default_writes_no_sidecar(self, tmp_path):
        s = Session("mock", pool_gb=5, pool_dir=tmp_path)
        assert s.status_dict()["rerank"] == "off"
        s.run("hello world build something")
        s.close()
        assert not (_Path(tmp_path) / "mpo.json").exists()

    def test_off_cold_retrieve_matches_pool_search(self, tmp_path):
        s = Session("mock", pool_gb=5, pool_dir=tmp_path)
        s.remember("the database password is in vault")
        s.remember("the api runs on port 8080")
        import numpy as np
        q = s.encoder.encode("database vault")
        key = s._key()
        direct = s.pool.search(q, 8, session=s._scope())
        via = s._cold_retrieve(key, q, 8)
        assert [x.id for x in via] == [x.id for x in direct]
        s.close()

    def test_mpo_mode_writes_sidecar_on_close(self, tmp_path):
        s = Session("mock", pool_gb=5, pool_dir=tmp_path, rerank="mpo")
        assert s.status_dict()["rerank"] == "mpo"
        s.remember("load-bearing fact one about auth tokens")
        s.remember("load-bearing fact two about rate limits")
        s.run("explain the auth tokens and rate limits")
        s.close()
        assert (_Path(tmp_path) / "mpo.json").exists()

    def test_mpo_mode_reopen_loads_operator(self, tmp_path):
        s1 = Session("mock", pool_gb=5, pool_dir=tmp_path, rerank="mpo")
        s1.remember("fact alpha for training the operator")
        s1.remember("fact beta for training the operator")
        s1.run("use fact alpha and fact beta")
        s1.close()
        import json
        data = json.loads((_Path(tmp_path) / "mpo.json").read_text(encoding="utf-8"))
        updates1 = data["core"]["n_updates"]
        assert updates1 > 0
        s2 = Session("mock", pool_gb=5, pool_dir=tmp_path, rerank="mpo")
        assert s2.status_dict()["mpo_updates"] == updates1
        s2.close()

    def test_invalid_rerank_raises(self, tmp_path):
        import pytest
        from aether_context.errors import AetherContextError
        with pytest.raises(AetherContextError):
            Session("mock", pool_gb=5, pool_dir=tmp_path, rerank="bogus")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session.py::TestMpoRerank -q`
Expected: FAIL (`Session() got an unexpected keyword argument 'rerank'`)

- [ ] **Step 3: Edit `aether_context/session.py`**

3a. Add the import near the other `aether_context` imports (after the `slice_loader` import line):

```python
from aether_context.mpo_rerank import MpoReranker
```

3b. Add constants beside the existing module constants (after `_EXTENDED_RESIDENT_BONUS`):

```python
#: Valid re-rank modes for Session(rerank=...). "off" = today's pure-cosine path.
_VALID_RERANK = ("off", "mpo")
#: Sidecar filename for the persisted MPO operator, written inside the pool dir.
_MPO_SIDECAR_NAME = "mpo.json"
```

3c. Add two parameters to `Session.__init__` signature (after `session_id: str | None = None,`):

```python
        rerank: str = "off",
        rerank_overscan: int = 4,
```

3d. Validate and store them. Insert right after `self.pool_mode: str = pool_mode` (near the top of `__init__`):

```python
        if rerank not in _VALID_RERANK:
            raise AetherContextError(
                f"rerank={rerank!r} is not one of {_VALID_RERANK}.",
                hint="Use rerank='off' (default, pure cosine) or rerank='mpo'.",
            )
        self.rerank_mode: str = rerank
        self._rerank_overscan: int = max(1, int(rerank_overscan))
```

3e. Build the re-ranker after the pager is constructed. Insert immediately after the
`self.pager: Pager = Pager(...)` assignment block (and before `# -- run state`):

```python
        # -- optional MPO re-rank bolt-on (read-path only; off by default) -----
        self.reranker: MpoReranker | None = None
        self._mpo_sidecar: Path | None = None
        if self.rerank_mode == "mpo":
            self._mpo_sidecar = Path(pool_config.dir) / _MPO_SIDECAR_NAME
            self.reranker = MpoReranker.load_or_new(
                self._mpo_sidecar, overscan=self._rerank_overscan
            )
```

3f. Replace the body of `_cold_retrieve` with the two-stage version:

```python
    def _cold_retrieve(self, key: SliceKey, query_vec: np.ndarray, k: int) -> list[Slice]:
        """Pager cold path. With rerank='mpo', cosine-recall top-M then MPO re-rank to top-k.

        With rerank='off' (default) this is exactly a session-scoped ``pool.search`` — no
        behavior change. The re-rank stage is fail-soft (the reranker itself degrades to the
        cosine order on any error or a CUSUM stale verdict).
        """
        if self.reranker is None:
            return self.pool.search(query_vec, k, session=self._scope())
        width = self.reranker.recall_width(k)
        candidates = self.pool.search(query_vec, width, session=self._scope())
        return self.reranker.rerank(
            query_vec, candidates, k, score_of=self.witness.score
        )
```

3g. Persist + train on close. In `close()`, before `self._flush_pool()`, add:

```python
        self._persist_reranker()
```

And add these two methods next to `_flush_pool`:

```python
    def _persist_reranker(self) -> None:
        """Train the MPO operator on this session's judged slices and save it (fail-soft).

        No-op when rerank='off'. Training set is every slice this session externalized into
        the pool (polarity is derived inside the re-ranker from the witness score + meta).
        Both train and save are best-effort — a re-rank failure must never block a clean close.
        """
        if self.reranker is None or self._mpo_sidecar is None:
            return
        try:
            slices = self._live_session_slices()
            if slices:
                self.reranker.train(slices, score_of=self.witness.score)
            self.reranker.save(self._mpo_sidecar)
        except Exception as exc:  # noqa: BLE001 - fail-soft: close must never raise
            logger.warning("MPO persist failed (%s); operator not saved this run", exc)

    def _live_session_slices(self) -> list[Slice]:
        """The live pool slices this session externalized (the operator's training set)."""
        out: list[Slice] = []
        for sid in list(self.pool._slices.keys()):  # pool is the source of truth (read-only)
            sl = self.pool._slices.get(sid)
            if sl is not None and sl.session == self.id:
                out.append(sl)
        return out
```

> Note: `_live_session_slices` reads `pool._slices` directly. That dict is the pool's live
> in-RAM slice map (see `context_pool.py`); we only read it, never mutate it. This keeps the
> training set exact without adding a public pool API for a single internal consumer.

3h. Add re-rank fields to `status_dict()`. Inside the returned dict literal, add:

```python
            "rerank": self.rerank_mode,
            "rerank_stale": (self.reranker.is_stale if self.reranker else False),
            "mpo_updates": (self.reranker.core._n_updates if self.reranker else 0),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session.py::TestMpoRerank -q`
Expected: PASS

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `python -m pytest -q`
Expected: PASS (all pre-existing tests still green; new tests included)

- [ ] **Step 6: Commit**

```bash
git add aether_context/session.py tests/test_session.py
git commit -m "feat: wire opt-in MPO re-rank into Session (two-stage retrieve, train+persist on close)"
```

---

### Task 4: Docs + final verification + PR

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a CHANGELOG entry**

Open `CHANGELOG.md`; under the top/Unreleased section's "Added" list (match the file's existing
heading style — if there is no Unreleased section, add one above the latest version), add:

```markdown
- **MPO re-ranker (opt-in).** `Session(rerank="mpo")` adds a learned tensor-train re-rank
  stage after cosine recall, with a CUSUM staleness fallback to pure cosine. The operator
  persists to `<pool_dir>/mpo.json` so re-rank quality survives a close + reopen. Math is a
  faithful, dependency-free port of the AETHER-ATLAS MPO (numpy-only). Default `rerank="off"`
  is byte-identical to prior behavior.
```

- [ ] **Step 2: Run lint + types + full tests**

Run: `python -m ruff check aether_context/mpo.py aether_context/mpo_rerank.py aether_context/session.py`
Expected: no errors (line-length 100; fix any reported).

Run: `python -m mypy aether_context/mpo.py aether_context/mpo_rerank.py` (if mypy is installed)
Expected: no errors (add precise hints if mypy flags numpy untyped einsum returns).

Run: `python -m pytest -q`
Expected: PASS (entire suite).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog entry for the MPO re-ranker bolt-on"
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/mpo-bolt-on-reranker
gh pr create --base main --title "feat: MPO bolt-on re-ranker for the persistent session" --body "$(cat <<'EOF'
## Summary
Adds an opt-in, persistent, fail-soft MPO learned re-ranker to the retrieval path.
Cosine recall (top-M) -> MPO tensor-train re-rank (top-k), guarded by a CUSUM staleness
detector that falls back to pure cosine. Cores persist to `<pool_dir>/mpo.json` so the
learned operator survives a close + reopen (the "persistent session"). Default `rerank="off"`
is byte-identical to today.

The MPO math is a faithful, dependency-free **clone** of the AETHER-ATLAS MPO
(`aether_atlas/mpo/`) — same dims, bond chain, contraction, loss, CUSUM — so the session
re-ranker and the main atlas stay correlated on one operator definition (bridge-ready). No
import of, or change to, the atlas repo. numpy-only; no new dependency.

## Changes
- `aether_context/mpo.py` — ported `MpoCore` + `CusumDeltaTracker`, generic over an
  `MpoCandidate` protocol (polarity supplied by caller).
- `aether_context/mpo_rerank.py` — `MpoReranker`: Slice->candidate adapter, CUSUM-guarded
  `rerank`, off-hot-path `train`, JSON persistence with encoder-version guard.
- `aether_context/session.py` — opt-in `rerank="mpo"` / `rerank_overscan`; two-stage
  `_cold_retrieve`; train + save on `close()`; `status_dict()` fields. Default path unchanged.
- Tests: `tests/test_mpo.py`, `tests/test_mpo_rerank.py`, `tests/test_session.py::TestMpoRerank`.

## Read-path only
The MPO never touches the write path, the pool storage, or the witness — exactly as in atlas.

## Test plan
- [x] `pytest -q` green (ported math, adapter, guarded rerank, fallback, fail-soft, persistence,
  session on/off parity + reopen).
- [ ] Manual: `Session("ollama/...", rerank="mpo")` over a long run; confirm `mpo.json` grows
  `n_updates` across sessions and `status_dict()["rerank"]` reports honestly.

Spec: `docs/superpowers/specs/2026-06-14-mpo-bolt-on-reranker-design.md`
Plan: `docs/superpowers/plans/2026-06-14-mpo-bolt-on-reranker.md`
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- §4.1 MPO core clone → Task 1 (incl. drift pin for the correlation invariant). ✓
- §4.2 `MpoReranker` adapter/rerank/train/persistence → Task 2. ✓
- §4.3 Session wiring (params, two-stage `_cold_retrieve`, train+save on close, status) → Task 3. ✓
- §4.4 persistence format + encoder_version guard → Task 2 (`save`/`load_or_new`) + tests. ✓
- §6 fail-soft table → exercised by `test_failsoft_on_core_error`, `test_stale_falls_back...`,
  corrupt/missing/mismatch persistence tests, and `_persist_reranker` try/except. ✓
- §7 testing → Tasks 1–3 test files. ✓
- §8 reversibility (default off, no write-path change) → `test_off_by_default...` +
  `test_off_cold_retrieve_matches_pool_search`. ✓
- §9 correlation note → captured in `mpo.py` docstring + drift test. ✓

**Placeholder scan:** none — every code step is complete.

**Type consistency:** `MpoReranker.rerank(query_vec, slices, k, score_of)`, `.train(slices,
score_of, *, lr, n_iter)`, `.save(path)`, `.load_or_new(path, **kw)`, `.recall_width(k)`,
`.is_stale`, `.core`, `_polarity_for(score, meta)` used identically across plan and tests.
`Session` new attrs: `rerank_mode`, `_rerank_overscan`, `reranker`, `_mpo_sidecar`,
`_persist_reranker`, `_live_session_slices`. `MpoCore.from_dict`/`to_dict`,
`CusumDeltaTracker.to_dict`/`from_dict` match Task 1.
