# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo — ported MPO core, polarity passthrough, CUSUM."""
from dataclasses import dataclass

import numpy as np

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
    def test_short_stable_stream_does_not_fire(self):
        # A constant stream accumulates the `drift` slack on the low side each step (atlas
        # design), so over a short window (4 increments * 0.5 = 2.0 < 3.0) it must NOT fire.
        cusum = CusumDeltaTracker(threshold=3.0, drift=0.5)
        fired = [cusum.observe(1.0) for _ in range(5)]
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
