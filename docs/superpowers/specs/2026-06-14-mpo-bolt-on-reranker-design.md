# MPO Bolt-On Re-Ranker for the Persistent Session — Design

**Date:** 2026-06-14
**Status:** Approved-for-planning
**Repo:** Unlimited-Context-LLM (`aether-context`)
**Author:** Aether research

## 1. Summary

Bolt a **Matrix Product Operator (MPO) learned re-ranker** onto the retrieval path of
Unlimited Context. Retrieval today is a single-stage cosine ANN scan over 256-dim static
embeddings (`ContextPool.search`). This design adds an **opt-in second stage**: cheap
cosine recall (top-M) → MPO learned re-rank → top-k, guarded by a CUSUM staleness detector
that falls back to pure cosine whenever the operator drifts from ground truth.

The MPO **cores persist to disk** alongside the pool, so the learned operator survives a
close + reopen. That is the "persistent session": re-rank quality compounds across
sessions instead of cold-starting every run.

The MPO math is a **faithful, dependency-free port** of the AETHER-ATLAS MPO
(`aether_atlas/mpo/`). Same physical dims, same bond chain, same contraction, same
contrastive loss, same CUSUM. This is deliberate: the two systems must share **one operator
definition** so they are correlated by construction and can later be bridged (one cores
format, one math) rather than diverging into two independent atlases.

## 2. Goals / Non-Goals

### Goals
- A learned, **read-path-only** re-ranker that improves which slices are surfaced for a turn.
- **Opt-in**: `rerank="mpo"`. Off by default → byte-identical behavior to today.
- **Fail-soft**: any error in embed/contract/train degrades to cosine order; never raises
  into a long run (engine design law 3).
- **Persistent**: cores + CUSUM state saved to `<pool_dir>/mpo.json`, restored on open.
- **Numpy-only**: no new runtime dependency (matches the `aether-context` core contract).
- **Atlas-correlated**: identical PHYSICAL_DIMS / BOND_CHAIN / contraction / loss / CUSUM
  to `aether_atlas.mpo`, so a future bridge shares the operator and cores format.

### Non-Goals
- No change to the **write path** (encode-on-spill, pool `add`, eviction). The MPO is a
  read-path accelerator, exactly as in atlas.
- **No dependency on or modification of** the AETHER-ATLAS repo. We clone the relevant math,
  we do not import it.
- No replacement of the `Witness` (retention/eviction stays cosine+salience based).
- No GPU / torch. No training during generation (training is an explicit, off-hot-path call).
- No quantum, no Protocol-C signing, no `AtlasCell` schema.

## 3. Background: the two systems

### 3.1 Atlas MPO (what we are cloning)
`aether_atlas/mpo/tensor_core.py` — `MpoCore`:
- `PHYSICAL_DIMS = {"model":16, "skill":32, "prompt":16, "time":4}` (sum 68).
- `BOND_CHAIN = (2, 8, 8, 2, 1)`.
- `_embed_feature_code(cell)` — cyclic read of `feature_code` into the 4 axes (handles any
  code length; identical embedding for identical codes; similarity-preserving).
- `forward(cells)` — per-cell transfer matrices `A_k = Σ_i v_i · core_k[:,i,:]`, chained and
  read at `[0,0]` → one scalar per cell. `score = scalar × confidence`. `s_pos/s_neg` are
  confidence-weighted polarity sums; `delta_norm = ||s_pos − s_neg||²`.
- `train(cells, lr, n_iter)` — analytic gradients of `L = Σ_{P+}||x_c − x̃_c||² − α||s+ − s−||²`,
  with global grad-norm clipping (`MAX_GRAD_NORM`).
- `to_dict` / `from_dict` — cores round-trip.

`aether_atlas/mpo/polarity_sieve.py` — `PolaritySieve` + `CusumDeltaTracker`:
- `sweep(cells) -> SieveResult` (s_pos, s_neg, cell_ids, delta_norm, fidelity, scores).
- `sweep_with_guard(cells) -> (result, is_stale)` — CUSUM over `delta_norm`; stale ⇒ caller
  falls back to classical ANN.
- `train_and_recover(cells)` — retrain bonds, reset CUSUM, clear stale flag.

`_polarity(cell)`: +1 iff `edge>0 and source=="real" and lifecycle==VERIFIED`; −1 iff
`edge<0 or lifecycle in {REFUTED,STALE,EVICTED}`; else 0.

### 3.2 Unlimited Context retrieval (what we are bolting onto)
- `StaticEncoder.encode(text) -> (256,) unit vector` (numpy-only, deterministic).
- `ContextPool.search(query_vec, k, session) -> list[Slice]` — cosine top-k (flat/HNSW),
  session-namespaced. `Slice = {id, session, vector(256), text, tokens, meta, score}`.
- `Pager` — warm cache; cold path injected as `retrieve_fn` (defaults to a session-scoped
  pool search). `Session._cold_retrieve` is the single funnel for the cold path.
- `Witness` — per-slice retention score (`SALIENT_THRESHOLD=0.60`), decay, eviction order.
- `Session` — ties them together; `remember()` plants high-salience facts; `recall()` reads.

## 4. Architecture

```
                       ┌─────────────────────────────────────────────┐
   query text ──encode─┤ Session._cold_retrieve(key, qvec, k)         │
                       │                                              │
                       │   M = k * OVERSCAN  (default 4)              │
                       │   cand = pool.search(qvec, M, scope)  ←cosine recall (cheap)
                       │                                              │
                       │   if reranker enabled:                       │
                       │     ranked, stale = reranker.rerank(qvec, cand)
                       │     return (cosine top-k if stale            │
                       │             else ranked top-k)               │
                       │   else: return cand[:k]      (today's path)  │
                       └─────────────────────────────────────────────┘
        on close: reranker.save(<pool_dir>/mpo.json)   ← persistent operator
        on open : reranker = MpoReranker.load_or_new(<pool_dir>/mpo.json)
```

### 4.1 New module: `aether_context/mpo.py`
A **dependency-free clone** of the atlas MPO math. No import of `aether_atlas`. Generic over
a tiny structural protocol instead of `AtlasCell`:

```python
class MpoCandidate(Protocol):
    feature_code: tuple[float, ...] | np.ndarray   # the vector the kernel sees
    polarity: int                                   # +1 / -1 / 0
    confidence: float                               # [0,1] weight
    id: str                                          # stable identifier
```

Contents (ported verbatim in math, renamed for our domain where helpful):
- Constants: `PHYSICAL_DIMS`, `BOND_CHAIN`, `ALPHA_DEFAULT`, `MAX_GRAD_NORM` — **identical
  values to atlas**.
- `_embed_feature_code`, `_embed_batch`, `_init_cores`, `_axis_mats`, `_contract` — verbatim.
- `MpoCore` with `forward`, `train`, `_loss_and_grads`, `to_dict`, `from_dict` — verbatim,
  except it reads `candidate.polarity` / `candidate.confidence` / `candidate.feature_code`
  directly (atlas's `_polarity(cell)` is replaced by the candidate's precomputed `polarity`,
  because our polarity comes from session signals, not edge/lifecycle).
- `CusumDeltaTracker` — verbatim.
- `MpoForwardResult`, `SieveResult` — verbatim shapes.

> **Correlation invariant (documented in the module docstring):** the dims, bond chain,
> contraction einsums, loss formula, and CUSUM constants MUST equal
> `aether_atlas.mpo.tensor_core`. If atlas changes the operator, this file is updated to
> match. A drift test (§7) pins the constants.

### 4.2 New module: `aether_context/mpo_rerank.py`
The bolt-on adapter + façade. This is the only place that knows about `Slice`.

```python
@dataclass(frozen=True)
class _SliceCandidate:           # implements MpoCandidate
    feature_code: np.ndarray     # = slice.vector (256-dim; cyclic read handles len != 68)
    polarity: int
    confidence: float
    id: str

class MpoReranker:
    def __init__(self, core: MpoCore | None = None, *,
                 cusum_threshold: float = 3.0, overscan: int = 4): ...

    # adapter: Slice + retention signal -> candidate
    @staticmethod
    def _to_candidate(sl: Slice, *, score: float, hit: bool) -> _SliceCandidate: ...

    # the read-path move: re-rank cosine candidates, guarded
    def rerank(self, query_vec, slices: list[Slice], k: int,
               score_of: Callable[[str], float]) -> list[Slice]: ...

    # off-hot-path learning from a session's judged slices
    def train(self, slices, score_of, lr=0.01, n_iter=20) -> float: ...

    # persistence (the "persistent session")
    def save(self, path) -> None: ...
    @classmethod
    def load_or_new(cls, path, **kw) -> "MpoReranker": ...
```

**Polarity mapping** (`Slice` → P), via the witness score `s = score_of(slice.id)` and meta:
- `meta.get("kind") == "fact"` **or** `s >= SALIENT_THRESHOLD (0.60)` → **+1**
- `meta.get("stale")` truthy, or `meta.get("kind") == "evicted"`, or `s < _NEG_FLOOR (0.15)`
  → **−1**
- else → **0** (skipped by the sweep, exactly like atlas synthetic priors)

`confidence` = `clamp(s, 0, 1)` (the retention score doubles as the MPO weight).
`feature_code` = `slice.vector` (the 256-dim retrieval embedding). The cyclic read in
`_embed_feature_code` maps 256 dims onto the 68-dim physical layout deterministically and
similarity-preservingly, so the operator is identical to atlas's even though our codes are
longer than trading's (8) or qopc's (64).

**`rerank` semantics (fail-soft, guarded):**
1. If reranker disabled or `< 2` candidates → return `slices[:k]` unchanged.
2. Build candidates; run `core.forward` + CUSUM `observe(delta_norm)`.
3. If CUSUM fires (**stale**) → log `STALE_SERVED`, return cosine order `slices[:k]`.
4. Else sort by `|score|` desc (ties → original cosine order, stable), return top-k.
5. Any exception → log + return `slices[:k]`. The re-rank is never a correctness dependency.

### 4.3 Wiring into `Session`
- New `__init__` param `rerank: str = "off"` (`"off" | "mpo"`). Validated like `pool_mode`.
- New param `rerank_overscan: int = 4` (M = k × overscan, floored at k).
- When `rerank == "mpo"`:
  - `self.reranker = MpoReranker.load_or_new(<pool_dir>/mpo.json, overscan=rerank_overscan)`.
  - `_cold_retrieve` pulls `M` cosine candidates, then
    `self.reranker.rerank(qvec, cand, k, score_of=self.witness.score)`.
  - `close()` calls `self.reranker.train(judged_slices, ...)` then `self.reranker.save(...)`
    (both fail-soft). Training set = the session's pool slices (those with P≠0).
- When `rerank == "off"`: `self.reranker is None`; `_cold_retrieve` is **exactly today's**
  `pool.search(qvec, k, scope)` — zero behavior change, zero new disk artifact.
- `status_dict()` gains `"rerank"` (the mode) and, when on, `"rerank_stale"` (bool) and
  `"mpo_updates"` (core `n_updates`) — all honest, never a silent capability claim.

### 4.4 Persistence format `<pool_dir>/mpo.json`
```json
{
  "version": 1,
  "encoder_version": "static_v1",
  "core": { "...MpoCore.to_dict()...": "alpha, n_updates, cores, bond_chain, physical_dims" },
  "cusum": {"mu": 0.0, "cusum_high": 0.0, "cusum_low": 0.0, "threshold": 3.0, "drift": 0.5, "n": 0}
}
```
- `encoder_version` mismatch on load → start a **fresh** core (stale cores trained on a
  different embedding scheme are not reused; logged, not raised).
- Missing/corrupt file → fresh core (fail-soft; never blocks a session).
- The file lives next to `pool.json` / `vectors.f32`; in `shared` mode one operator serves the
  shared pool, in `separate` mode it is still one file per pool dir (the operator is
  domain-agnostic — it ranks by learned relevance, the session scope is enforced upstream by
  the cosine recall stage).

## 5. Data flow (one turn, rerank=mpo)

1. Model reasons; `Session._page_working_set` / prefetch calls the pager.
2. Pager cold path → `Session._cold_retrieve(key, qvec, k)`.
3. `pool.search(qvec, M=k×4, scope)` → M cosine candidates (recall).
4. `reranker.rerank(qvec, candidates, k, witness.score)`:
   - candidates ← Slice→candidate (polarity from witness score + meta).
   - `core.forward` → per-candidate scores + `delta_norm`; CUSUM observes.
   - stale ⇒ cosine top-k; else MPO-sorted top-k.
5. Pager warms the returned k; model is handed the resident window.
6. On `close()`: train cores on judged slices, save `mpo.json`.

## 6. Error handling / fail-soft

| Failure | Behavior |
|---|---|
| Encoder error building query | pager already handles (returns `[]`); rerank not reached |
| `core.forward` raises | log, return cosine `slices[:k]` |
| CUSUM fires (stale operator) | log `STALE_SERVED`, return cosine `slices[:k]` |
| `train` raises on close | log, skip training, still attempt save of current cores |
| `save` raises | log, close continues (operator just isn't persisted this run) |
| `mpo.json` corrupt / version mismatch | log, fresh core |
| `rerank="off"` | reranker is `None`; original cosine path verbatim |

Mirrors the engine's existing fail-soft discipline (`Session`, `Pager` already log-and-degrade).

## 7. Testing

Port the atlas MPO suite, then add bolt-on + integration tests. Pytest, AAA, ≥80% on new code.

**`tests/test_mpo.py`** (ported math, adapted to candidates):
- polarity passthrough; forward empty / pos-only / mixed / skips-P0; train reduces loss;
  train no-positive no-crash; serialization round-trip; **drift test**: assert
  `PHYSICAL_DIMS`, `BOND_CHAIN`, `ALPHA_DEFAULT` equal the atlas values (string-literal pin,
  no atlas import) so a silent divergence fails CI.

**`tests/test_mpo_rerank.py`**:
- `_to_candidate` polarity mapping (fact→+1, salient→+1, faded→−1, neutral→0).
- rerank changes order vs cosine when cores favor a different slice (train then assert).
- **stale → fallback**: force CUSUM fire → returns exact cosine order.
- **fewer than 2 candidates / empty** → returned unchanged.
- **fail-soft**: monkeypatch `core.forward` to raise → cosine order, no exception.
- persistence: save → load_or_new restores cores (`n_updates`, shapes); corrupt file → fresh;
  encoder_version mismatch → fresh.
- off-by-default: `MpoReranker` not constructed when mode off (covered in session test).

**`tests/test_session.py`** (extend):
- `Session(rerank="mpo")` runs end-to-end on the mock; produces `mpo.json` on close.
- `Session(rerank="off")` (default) → **no `mpo.json`**, `_cold_retrieve` returns the same
  slices as a direct `pool.search` (parity assertion).
- reopen with `rerank="mpo"` loads the persisted operator (`status_dict()["mpo_updates"] > 0`).
- invalid `rerank=` value raises a typed `AetherContextError`/`PoolBudgetError`.

## 8. Rollout / reversibility

- Pure addition: two new modules + additive `Session`/`status` params. No edits to encoder,
  pool write path, witness, or default behavior.
- Default `rerank="off"` ⇒ the feature is dormant until explicitly enabled. Removing the
  feature = delete two files + the wiring; no migration, no data format change to the pool.
- `mpo.json` is independent of `pool.json`; deleting it just cold-starts the operator.

## 9. Open correlation note (for a future bridge, not this PR)

Because the operator math is identical to `aether_atlas.mpo`, a later bridge can:
- share the `core` cores JSON between an atlas instance and a session (same shapes/format), and
- treat session-trained cores as a warm start for an atlas pool (and vice versa),
so the session and the main atlas **collaborate on one operator** rather than drifting into
two. This PR only establishes the shared math + format; the bridge itself is out of scope.
