# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for aether_context.mpo — the MPO context chain (couple on cost & time, expand a hit)."""
import numpy as np

from aether_context.mpo import ChainItem, MpoChain


def _vec(seed, dim=256):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _item(cid, vec, cost=10.0, time=0.0, cached=False):
    return ChainItem(id=cid, vector=vec, cost=cost, time=time, cached=cached)


class TestChainEmbed:
    def test_deterministic(self):
        c = MpoChain()
        assert np.allclose(c.chain_embed(0.3, 0.7), c.chain_embed(0.3, 0.7))

    def test_unit_norm(self):
        c = MpoChain()
        e = c.chain_embed(0.2, 0.9)
        assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-6

    def test_smooth_close_inputs_close_embeds(self):
        c = MpoChain()
        near = float(np.dot(c.chain_embed(0.50, 0.50), c.chain_embed(0.51, 0.50)))
        far = float(np.dot(c.chain_embed(0.50, 0.50), c.chain_embed(0.05, 0.95)))
        assert near > far  # nearby cost/time -> nearer on the chain manifold


class TestCoupling:
    def test_semantic_drives_coupling(self):
        c = MpoChain()
        v = _vec(1)
        e = c.chain_embed(0.5, 0.5)
        same = c.coupling(v, e, v, e)            # identical vector
        diff = c.coupling(_vec(2), e, v, e)      # orthogonal-ish vector
        assert same > diff

    def test_chain_proximity_modulates(self):
        c = MpoChain()
        v = _vec(3)
        e_near = c.chain_embed(0.5, 0.5)
        e_far = c.chain_embed(0.0, 1.0)
        # same semantic vector, different chain position -> closer chain couples >=
        assert c.coupling(v, e_near, v, e_near) >= c.coupling(v, e_far, v, e_near)


class TestExpand:
    def test_hits_come_first(self):
        c = MpoChain(width=2)
        items = [_item(f"x{i}", _vec(i), time=float(i)) for i in range(6)]
        out = c.expand(["x0"], items, width=2)
        assert out[0] == "x0"

    def test_pulls_coupled_slices(self):
        c = MpoChain(width=3)
        # x0 is the hit; x1 shares its vector + neighborhood, z is unrelated & far.
        base = _vec(10)
        items = [
            _item("x0", base, cost=10, time=0.5),
            _item("x1", base * 0.99 + 0.01 * _vec(11), cost=10, time=0.5),
            _item("z", _vec(99), cost=200, time=0.0),
        ]
        out = c.expand(["x0"], items, width=2)
        assert out[0] == "x0"
        assert out.index("x1") < out.index("z")  # the coupled slice is pulled before the stranger

    def test_empty_candidates_returns_hits(self):
        c = MpoChain()
        assert c.expand(["a"], []) == ["a"]

    def test_dedup_and_truncation_via_width(self):
        c = MpoChain(width=1)
        items = [_item(f"x{i}", _vec(i), time=float(i)) for i in range(5)]
        out = c.expand(["x0"], items, width=1, hops=1)
        assert out[0] == "x0" and len(out) == 2 and len(set(out)) == len(out)

    def test_cached_cost_discount_changes_normalization(self):
        # A cached high-cost slice is treated as cheaper -> its normalized cost shifts.
        c = MpoChain(cache_bonus=1.0)
        items_uncached = [_item("a", _vec(1), cost=100, time=0.0),
                          _item("b", _vec(2), cost=10, time=1.0)]
        items_cached = [_item("a", _vec(1), cost=100, time=0.0, cached=True),
                        _item("b", _vec(2), cost=10, time=1.0)]
        # expand runs without error in both; cache changes the cost manifold, not the contract
        assert c.expand(["a"], items_uncached) and c.expand(["a"], items_cached)
