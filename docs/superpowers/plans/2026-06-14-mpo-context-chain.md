# MPO Context Chain Implementation Plan

> Implemented. See `docs/superpowers/specs/2026-06-14-mpo-context-chain-design.md` for the design.

**Goal:** Add an MPO (Matrix Product Operator) context chain that links the session's slices
and assists cosine retrieval by expanding a hit into its most-coupled slices — widening the
working set with connected context. Coupling ranks on 2 session-local constants: cost & time.

**Architecture:** `MpoChain` (2-site tensor train over cost & time → chain manifold; coupling =
semantic similarity gated by chain proximity; `expand` follows the chain from cosine hits).
Cosine remains the retrieval mechanism. On by default; disable with `mpo_chain=False`.

---

## Tasks (all complete)

### Task 1 — `aether_context/mpo.py` (`MpoChain`)
- `ChainItem(id, vector, cost, time, cached)`; `chain_embed(cost,time)` (the MPO contraction);
  `coupling(...)`; `expand(hits, candidates, width, hops)` (cost discounted for cached items;
  cost/time normalized over the candidate set).
- Tests: `tests/test_mpo.py` (embedding determinism/smoothness, coupling, expand pulls coupled
  slices hits-first, cache discount, fail-soft).

### Task 2 — `aether_context/session.py`
- `Session(mpo_chain=True, chain_width=8, chain_hops=1, chain_fanout=4)` (default on).
- `_encode_slice` stamps `meta["t"]` (session position = the time constant).
- `_cold_retrieve`: cosine-recall `k·fanout`, seed with top cosine hits, `MpoChain.expand` to
  fill the `k` working set with connected context (`cached` = pager resident ids); fail-soft.
- `status_dict()` surfaces `mpo_chain`.
- Tests: `tests/test_session.py::TestMpoChain` (on-by-default, disable parity, returns k,
  widens with a planted connected thread, fail-soft).

### Task 3 — CLI
- `--no-mpo-chain` on run/chat/status; `_build_session` passthrough.

### Task 4 — Bench + docs
- `bench/chain_recall.py` — connected-thread recall: cosine 0.15 vs chain 0.78 @k=8.
- README "MPO: the context chain" section; CHANGELOG; this plan + the design spec.

## Verification
`python -m pytest -q` — full suite green. `python -m bench.chain_recall` — 0.15 → 0.78.
