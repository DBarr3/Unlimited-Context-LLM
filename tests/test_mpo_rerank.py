# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo_rerank — Slice adapter, guarded rerank, persistence."""
import json

import numpy as np

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
        data = json.loads(path.read_text(encoding="utf-8"))
        data["encoder_version"] = "something_else"
        path.write_text(json.dumps(data), encoding="utf-8")
        rr2 = MpoReranker.load_or_new(path)
        assert rr2.core._n_updates == 0  # mismatched embedding scheme -> cold start

    def test_saved_encoder_version_is_current(self, tmp_path):
        path = tmp_path / "mpo.json"
        MpoReranker().save(path)
        assert json.loads(path.read_text(encoding="utf-8"))["encoder_version"] == ENCODER_VERSION
