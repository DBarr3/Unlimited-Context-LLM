# MPO Context Chain — Design

**Date:** 2026-06-14
**Status:** Implemented
**Repo:** Unlimited-Context-LLM (`aether-context`)

## 1. Summary

Add an **MPO (Matrix Product Operator) context chain** that links the session's slices into one
connected structure and **assists retrieval** by expanding a hit into its most-coupled slices —
widening the working set with *connected* context instead of isolated nearest-neighbors.

Cosine/semantic search **remains the retrieval mechanism**. The MPO does not rank relevance,
compress vectors, or replace search. It is a coupling layer: when cosine pulls an entry slice,
the chain pulls in the slices most coupled to it, so the model receives a coherent thread.

Coupling is ranked on **two session-local constants only — cost and time** (cache folds into
cost as cost/cache). Two constants, nothing else — session-focused.

## 2. Goals / Non-Goals

### Goals
- Link slices into a chain and **expand a cosine hit to its most-coupled slices** — widen the
  working set with connected context.
- Rank/couple on **2 constants: cost & time** (public, session-local; cache discounts cost).
- numpy-only, deterministic, fail-soft (the engine always works on plain cosine retrieval; the
  chain only ever *adds* connected context, never blocks or replaces a hit).

### Non-Goals
- Not the retrieval mechanism (cosine stays). Not a codec/compression. Not relevance scoring,
  ground truth, hallucination, or cost-accounting/billing.
- Two session-local constants only (cost, time). No extra axes, no relevance/confidence signals.
- No training, no GPU, no new dependency.

## 3. Mechanism

```
query ─cosine─► entry hits        (retrieval mechanism, unchanged)
                   │
                   ▼
       MPO chain couples slices on (cost, time) ─► pull slices most coupled to the hits
                   │
                   ▼
       widened working set: hits + connected context  ─► handed to the model
```

### 3.1 Per-slice chain coordinate (the 2 constants)
- **time** — the slice's session position, min/max-normalized to `[0,1]` over the candidate set
  (slices created near each other in the session are part of the same line of work).
- **cost** — the slice's token cost, normalized, **divided by a cache factor** (a slice already
  resident/warm is cheaper to pull, so `cost_eff = tokens / (1 + cache_bonus·cached)`).

### 3.2 The MPO (2-site tensor train)
Each constant is lifted to a small Fourier feature vector (smooth, deterministic), then the two
are contracted through a 2-core tensor train (bond `r`) into an `m`-dim **chain embedding** `e`.
This is the Matrix Product Operator over the two axes — fixed, deterministic linear algebra
(seeded cores; no training). Its role is to place every slice on a shared **chain manifold**
where closeness means "same cost/time neighborhood."

### 3.3 Coupling + expansion
- `coupling(i, h) = cosine(vec_i, vec_h) · (0.5 + 0.5·cosine(e_i, e_h))` — semantic similarity
  **gated** by chain proximity. A slice is strongly coupled to a hit when it's both semantically
  related and in the same cost/time neighborhood.
- `expand(hits, candidates, width, hops)` — from the seed hits, rank candidates by max coupling
  to any hit, take the top `width`, add them, repeat for `hops` (follow the chain). Returns hits
  first, then the coupled slices, de-duplicated.

## 4. Integration

### 4.1 `aether_context/mpo.py` — `MpoChain`
- `chain_embed(cost, time) -> e` (the MPO contraction); `coupling(...)`; `expand(...)`.
- Operates on lightweight `ChainItem(id, vector, cost, time)` tuples; normalizes cost/time over
  the provided candidate set (self-contained, no global state).

### 4.2 `aether_context/session.py`
- Opt-in `Session(mpo_chain=True, chain_width=..., chain_hops=...)`. Default off → today's
  cosine path is byte-identical.
- `_encode_slice` stamps `meta["t"]` (monotonic session clock) so each slice has a `time`.
- `_cold_retrieve` when chain on: cosine-recall a wider candidate set (`k·fanout`), seed with the
  top cosine hits, `MpoChain.expand` to fill the `k` working-set slots with connected context;
  `cached` = the pager's current resident ids (so warm slices are cheaper to chain in).
- `status_dict()` surfaces `mpo_chain`.

## 5. Fail-soft
Any error in embed/couple/expand → return the plain cosine `slices[:k]`. The chain only ever
augments the working set; a chain failure degrades to ordinary retrieval, never a crash.

## 6. Honest scope
The chain changes *which connected slices* fill the working set; it does not change the reach
math (that's the disk pool) or the matcher (that's cosine). Benefit is **coherence of recalled
context** (pulling the thread), measured by a connected-thread recall bench, not compression.

## 7. Testing
- `tests/test_mpo.py` — chain embedding determinism + smoothness; coupling rises with semantic +
  chain proximity; `expand` returns hits first then most-coupled; cache discounts cost; fail-soft.
- `tests/test_session.py` — off-by-default parity; chain on widens the window with same-thread
  slices; invalid args.
- `bench/chain_recall.py` — connected-thread recall: cosine-only vs chain, on planted threads.
