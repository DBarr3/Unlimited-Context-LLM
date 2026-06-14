# MPO Vector Codec for the Persistent Pool — Design

**Date:** 2026-06-14
**Status:** Implemented
**Repo:** Unlimited-Context-LLM (`aether-context`)

## 1. Summary

Add a **tensor-train (Matrix Product Operator) vector codec** to Unlimited Context's
disk/memory encoding. The engine's whole premise is *encode & recover* — overflow is encoded
to a local pool and the right slice is recovered on demand. This adds the numeric half of that
contract: each slice's retrieval vector is **compressed into a tensor-train factorization on
the way to disk and reconstructed on recovery**, so a long-running session's persistent pool
holds more reach per byte.

It is **opt-in** (`vector_codec="mpo"`, default `"none"` = byte-identical to today),
**numpy-only** (no new dependency), and **encode/recover only** — it does not rank, score,
gate, judge, or otherwise touch relevance, retention, or correctness. Those remain the
retrieval/witness layers' jobs. The codec's sole purpose is making the disk/memory encoding
smaller, with a measured, bounded reconstruction error.

## 2. Goals / Non-Goals

### Goals
- Compress the **on-disk vector store** (the reach-bounding resource) with a tensor-train codec.
- Opt-in; default off → identical behavior and on-disk layout to today.
- Numpy-only, deterministic, fail-soft (the engine always works on raw vectors; the codec is a
  footprint optimization, never a correctness dependency).
- Honest: report compression ratio and measured reconstruction fidelity; never a silent claim.

### Non-Goals
- No relevance ranking, scoring, gating, training, ground-truth, hallucination handling, cost
  accounting, or correctness grounding. **Encode and recover only.**
- No change to the search semantics, the witness, or the slice schema.
- No GPU / torch. No during-run hot/cold tiering (future work, explicitly not claimed).

## 3. Background: the reach model

`cli.py` documents that the **float32 vector store lives on disk**; RAM holds only the
~29 MB/GB index. Per slice the pool charges `dim*4 + 1200` bytes (`context_pool.py`): a 1024 B
vector plus a 1200 B payload (text + meta + bookkeeping). Reach is `pool_gb × 233M tokens`,
derived straight from that byte cost (`config.py`). So the on-disk vector block is a direct,
compressible contributor to how much reach fits per GB — the lever this codec pulls.

## 4. Architecture

```
  add(slice)  ──►  working mmap (vectors.f32, exact float32)  ──►  cosine search (unchanged)
                                   │ close()
                                   ▼
        encode each vector → tensor-train cores → vectors.mpo.json   (durable, smaller)
                                   │  then drop vectors.f32 (at-rest = compressed only)
                                   ▼
  open()  ──►  reconstruct cores → working mmap  ──►  cosine search over reconstructed vectors
```

During a live run the working store is the exact float32 mmap, so cosine search is unaffected.
The codec acts at the **persistence boundary**: on `close()` the durable form becomes the
compressed `vectors.mpo.json` and the raw `vectors.f32` is dropped; on `open()` the working
mmap is reconstructed from the compressed store. Net: the **persistent pool's at-rest footprint
shrinks** (~1.4× overall; the vector block ~2.7× at rank 4), so a larger persistent reach fits
on disk and reloads. Reconstruction is lossy-but-bounded; fidelity rises with the bond rank.

### 4.1 `aether_context/mpo.py` — `MpoCodec`
Standard **TT-SVD** (Oseledets, 2011). Pure, deterministic numpy.
- `MpoCodec(dim=256, mode_shape=(4,4,4,4), rank=4)` — `mode_shape` must multiply to `dim`.
- `encode(vector) -> TTVector` — reshape to the grid, sequential SVD with bond truncation to
  `rank`; returns the cores.
- `recover(TTVector) -> vector` — contract the cores back to a `(dim,)` float32 vector.
- `compression_ratio(tt=None)` — stored-floats ratio; with no arg, the worst-case (all bonds
  saturated) floor.
- `fidelity(vector, tt=None)` — cosine between a vector and its round-trip reconstruction.
- `to_dict`/`from_dict` (config) and `tt_to_lists`/`tt_from_lists` (per-vector JSON).

### 4.2 `aether_context/context_pool.py` — opt-in persistence
- `MpoCodec` built from `PoolConfig.vector_codec`/`codec_rank`.
- `close()` → `_persist_compressed()`: encode every live slice's stored unit vector, write
  `vectors.mpo.json` (`{version, codec, rows:[{row, tt}]}`), drop `vectors.f32`.
- `_load()`: if `vectors.f32` is absent but the compressed store is present, reconstruct the
  working mmap (re-normalizing each row to unit, since stored vectors are unit). If the
  compressed store is present but the pool was opened with `vector_codec="none"`, raise
  `PoolCorrupt` with a clear hint.
- `stats()` gains `vector_codec` + `vector_codec_ratio`.

### 4.3 `aether_context/config.py` / `session.py`
- `PoolConfig.vector_codec` (`"none"|"mpo"`) + `codec_rank` (≥1), validated.
- `Session(vector_codec=..., codec_rank=...)` passes through to the pool; `status_dict()`
  surfaces `vector_codec` + `vector_codec_ratio`.

### 4.4 Persistence format `vectors.mpo.json`
```json
{
  "version": 1,
  "codec": {"dim": 256, "mode_shape": [4, 4, 4, 4], "rank": 4},
  "rows": [{"row": 0, "tt": {"mode_shape": [4, 4, 4, 4], "cores": [[[...]]]}}]
}
```

## 5. Error handling / fail-soft

| Failure | Behavior |
|---|---|
| compressed write fails on close | log, keep raw `vectors.f32` (valid, larger pool) |
| compressed store malformed on load | `PoolCorrupt` with the file named |
| compressed present, codec off | `PoolCorrupt` with hint to reopen `vector_codec="mpo"` |
| `vector_codec="none"` (default) | raw float32 path, no compressed artifact — unchanged |

## 6. Testing
- `tests/test_mpo.py` — construction validation, encode/recover round-trip, full-rank near-lossless,
  low-rank lossy-but-bounded fidelity, determinism, compression ratio, serialization.
- `tests/test_context_pool.py::TestMpoVectorCodec` — off-by-default keeps raw + no sidecar;
  close writes compressed + drops raw; reopen reconstructs searchably; reopen without codec
  raises; stats report the codec.
- `tests/test_session.py::TestMpoVectorCodec` — off by default; status surfacing; persistent
  pool recovers a planted fact after reopen; invalid codec raises.

## 7. Reversibility
Pure addition: one new module + additive config/pool/session params. Default `"none"` keeps the
on-disk format and behavior identical. Removing the feature = delete `mpo.py` + the wiring; no
migration for `"none"` pools.
