# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo — the tensor-train (MPO) vector codec."""
import numpy as np
import pytest

from aether_context.mpo import DEFAULT_DIM, MpoCodec, TTVector


def _unit(seed, dim=256):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    return v / np.linalg.norm(v)


class TestConstruction:
    def test_default_dim_and_grid(self):
        c = MpoCodec()
        assert c.dim == DEFAULT_DIM
        assert int(np.prod(c.mode_shape)) == DEFAULT_DIM

    def test_bad_mode_shape_rejected(self):
        with pytest.raises(ValueError):
            MpoCodec(256, mode_shape=(3, 3))  # 9 != 256

    def test_bad_dim_rejected(self):
        with pytest.raises(ValueError):
            MpoCodec(0)

    def test_bad_rank_rejected(self):
        with pytest.raises(ValueError):
            MpoCodec(256, rank=0)


class TestEncodeRecover:
    def test_roundtrip_shape(self):
        c = MpoCodec()
        tt = c.encode(_unit(1))
        rec = c.recover(tt)
        assert rec.shape == (256,)
        assert rec.dtype == np.float32

    def test_encode_rejects_wrong_length(self):
        c = MpoCodec()
        with pytest.raises(ValueError):
            c.encode(np.zeros(100))

    def test_cores_have_boundary_bonds_one(self):
        c = MpoCodec()
        cores = c.encode(_unit(2)).cores
        assert cores[0].shape[0] == 1
        assert cores[-1].shape[-1] == 1
        # adjacent bonds line up
        for a, b in zip(cores, cores[1:]):
            assert a.shape[2] == b.shape[0]

    def test_full_rank_is_near_lossless(self):
        # rank high enough to saturate every bond -> reconstruction is essentially exact.
        c = MpoCodec(256, mode_shape=(4, 4, 4, 4), rank=16)
        v = _unit(3)
        assert c.fidelity(v) > 0.999

    def test_low_rank_is_lossy_but_bounded(self):
        c = MpoCodec(256, mode_shape=(4, 4, 4, 4), rank=4)
        # average fidelity over several vectors stays usefully high
        fids = [c.fidelity(_unit(s)) for s in range(20)]
        assert min(fids) > 0.5
        assert np.mean(fids) > 0.6

    def test_determinism(self):
        c = MpoCodec()
        v = _unit(4)
        a = c.recover(c.encode(v))
        b = c.recover(c.encode(v))
        assert np.array_equal(a, b)


class TestCompression:
    def test_low_rank_compresses(self):
        c = MpoCodec(256, mode_shape=(4, 4, 4, 4), rank=4)
        tt = c.encode(_unit(5))
        assert tt.param_count < 256
        assert c.compression_ratio(tt) > 1.0

    def test_worst_case_ratio_matches_actual(self):
        c = MpoCodec(256, mode_shape=(4, 4, 4, 4), rank=4)
        worst = c.compression_ratio()  # no tt -> worst-case (all bonds saturated)
        actual = c.compression_ratio(c.encode(_unit(6)))
        assert actual >= worst - 1e-9  # actual never worse than the advertised floor


class TestSerialization:
    def test_tt_lists_roundtrip(self):
        c = MpoCodec()
        tt = c.encode(_unit(7))
        rebuilt = MpoCodec.tt_from_lists(MpoCodec.tt_to_lists(tt))
        assert rebuilt.mode_shape == tt.mode_shape
        assert np.allclose(c.recover(rebuilt), c.recover(tt))

    def test_codec_config_roundtrip(self):
        c = MpoCodec(256, mode_shape=(4, 4, 4, 4), rank=5)
        c2 = MpoCodec.from_dict(c.to_dict())
        assert c2.dim == c.dim and c2.mode_shape == c.mode_shape and c2.rank == c.rank
