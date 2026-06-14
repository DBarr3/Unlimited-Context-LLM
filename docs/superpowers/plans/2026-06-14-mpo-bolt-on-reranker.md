# MPO Vector Codec Implementation Plan

> Implemented. See `docs/superpowers/specs/2026-06-14-mpo-bolt-on-reranker-design.md` for the design.

**Goal:** Add an opt-in, numpy-only tensor-train (MPO) vector codec that compresses the
context pool's on-disk vector store and reconstructs it on load, extending the persistent
pool's reach per byte. Encode & recover only — no ranking, scoring, gating, or grounding.

**Architecture:** `MpoCodec` (TT-SVD encode/recover) + opt-in pool persistence
(`vector_codec="mpo"`): compress on `close()`, reconstruct on `open()`. Default `"none"` =
byte-identical to today.

---

## Tasks (all complete)

### Task 1 — `aether_context/mpo.py` (`MpoCodec`)
- TT-SVD `encode(vector) -> TTVector`, `recover(TTVector) -> vector` (contraction).
- `compression_ratio`, `fidelity`, `to_dict`/`from_dict`, `tt_to_lists`/`tt_from_lists`.
- Tests: `tests/test_mpo.py` (validation, round-trip, full-rank near-lossless, low-rank bounded
  fidelity, determinism, compression, serialization).

### Task 2 — `aether_context/config.py`
- `PoolConfig.vector_codec` (`"none"|"mpo"`) + `codec_rank` (≥1), validated in `__post_init__`.

### Task 3 — `aether_context/context_pool.py`
- Build `MpoCodec` from config; `_compressed_file()` path; `COMPRESSED_VECTORS_FILENAME`.
- `close()` → `_persist_compressed()`: write `vectors.mpo.json`, drop `vectors.f32`.
- `_load()`: reconstruct mmap from the compressed store when raw is absent; `PoolCorrupt` when
  the compressed store is present but codec is off, or when the store is malformed.
- `stats()` gains `vector_codec` + `vector_codec_ratio`.
- Tests: `tests/test_context_pool.py::TestMpoVectorCodec`.

### Task 4 — `aether_context/session.py`
- `Session(vector_codec=..., codec_rank=...)` passthrough to `PoolConfig`; `status_dict()`
  surfaces `vector_codec` + `vector_codec_ratio`.
- Tests: `tests/test_session.py::TestMpoVectorCodec`.

### Task 5 — Docs
- CHANGELOG entry; this plan + the design spec.

## Verification
`python -m pytest -q` — full suite green.
