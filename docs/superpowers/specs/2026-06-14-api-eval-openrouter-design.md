# API Eval (OpenRouter) — Design

**Date:** 2026-06-14
**Status:** Approved-for-planning
**Repo:** Unlimited-Context-LLM (`aether-context`)

## 1. Summary

Prove the engine works on a real hosted model (not local): a vendor-neutral OpenAI-compatible
backend plus a paid/manual eval bench that runs against OpenRouter. The eval proves two things
on a real **reasoning** model:

1. **Recovery** — with the engine on, an early load-bearing fact stays reachable across a run
   that overflows the model's window; without it, the fact is lost.
2. **Chain A/B** — the MPO context chain (on vs off) lifts connected-context retrieval.

Scored two ways: **needle** (deterministic exact-match) and an **LLM-judge** (coherence 1–10).
Pure OpenRouter — no AETHER-CLOUD, no aether infra. The adapter is a genuine OSS feature.

## 2. Goals / Non-Goals

### Goals
- A vendor-neutral `OpenAICompatLLM` backend (`openai/<model>` spec) — works with any
  OpenAI-compatible API (OpenRouter, OpenAI, vLLM, …). stdlib `urllib` only, no new dependency.
- A `bench/api_eval.py` harness measuring recovery (ON/OFF) + chain (on/off), needle + judge.
- Default to a **cheap reasoning model** (DeepSeek-class), overridable via `--model`.
- Key from env (`OPENROUTER_API_KEY`); never committed/logged. Bench skips cleanly with no key.

### Non-Goals
- No AETHER-CLOUD endpoint, no aether infra, no private namespaces (moat-seal guard must pass).
- Not in the CI matrix (it costs money + needs a key). CI only covers the adapter's unit tests
  (mocked HTTP, no network).
- No new runtime dependency.

## 3. Adapter — `OpenAICompatLLM`

In `aether_context/local_llm.py`, implementing the `LocalLLM` protocol.

- **Spec:** `openai/<model>` → `OpenAICompatLLM(ref, base_url=…, api_key=…, context_window=…)`.
  Added to `parse_spec` (`openai` backend) and `load_model`.
- **Config resolution:** `base_url`/`api_key` from kwargs, else env. If `OPENROUTER_API_KEY`
  is set and no base_url given, default `base_url=https://openrouter.ai/api/v1`. Generic
  fallback: `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
- **`generate`** — POST `<base_url>/chat/completions` with `stream: true`; parse SSE lines
  (`data: {json}`, terminated by `data: [DONE]`); yield `choices[0].delta.content`. A
  reasoning model's separate `delta.reasoning` field is **ignored** (we page the final answer,
  not the scratchpad). `system`/`stop`/`max_tokens` forwarded as the OpenAI params.
- **`context_window`** — caller-provided (no generic probe). The eval sets it **small** to
  force overflow. Default fallback `DEFAULT_CONTEXT_WINDOW`.
- **`count_tokens`** — chars/4 estimate (no count endpoint).
- **Errors** — typed/hinted (`BackendUnavailable` on missing key / network; map HTTP 401/429/
  5xx). Key never appears in logs or error text.

## 4. Eval harness — `bench/api_eval.py`

Manual/paid bench (sibling of `drift_vs_window.py`). **No key → print a skip notice and exit 0.**

Common: `Session(model="openai/<m>", context_window=W, pool_dir=<tmp>)` with a small `W`
(default ~2048) so the run overflows into the pool. Flags: `--model`, `--trials`, `--window`,
`--max-calls` (hard cap), `--judge/--no-judge`, `--dry-run`.

### 4.1 Recovery (engine ON vs OFF), N≈20 trials
- Plant a unique needle early (`s.remember("the vault code is ZORN-7741")`, needle randomized
  per trial by index — no `Math.random`; seed from trial number).
- Drive a long body of filler that overflows `W` several times, then ask: "What is the vault code?"
- **ON** = `Session.run` (engine pages the needle back). **OFF** = a direct
  `OpenAICompatLLM.generate` call with the transcript truncated to `W` (no pool — the needle
  has fallen off). Same model, same window.
- **Score:** exact-match — does the answer contain the needle? → recovery rate ON vs OFF.

### 4.2 Chain A/B (chain on vs off), N≈20 trials
- Plant a thread of ~5 linked needles spread across the session (e.g. steps of one procedure),
  plus unrelated filler. Ask a question that needs several thread members.
- Two engine-ON sessions differing only in `mpo_chain` (True/False).
- **Score:** thread-recall — how many of the thread needles appear in the recovered working set
  (and in the answer) → on vs off.

### 4.3 Judge pass (when `--judge`)
- A judge model (OpenRouter, `--judge-model`, default a cheap instruct model) scores each
  ON/OFF answer 1–10 for coherence + completeness against the expected facts. Report mean
  on vs off for both tests.

## 5. Output / cost / safety
- Prints a table: recovery rate ON/OFF; chain thread-recall on/off; judge means. Writes
  `api_eval_results.json` (**gitignored**) with per-trial detail + model + counts + est. cost.
- `--max-calls` hard-caps total API calls; the harness refuses to exceed it.
- Key from env only; results file gitignored; no aether endpoints.

## 6. Testing
- `tests/test_local_llm_openai.py` (CI, mocked HTTP, **no network**): spec parsing
  `openai/<model>`; SSE stream parse → chunks; `[DONE]` handling; reasoning-delta ignored;
  error mapping (401/429/5xx → typed); env resolution (OpenRouter default base_url);
  `count_tokens` estimate.
- `bench/api_eval.py --dry-run` uses `MockLLM` so the harness logic (needle plant, overflow,
  scoring, table) is exercised with no key/network; CI can run the dry-run.

## 7. Reversibility
Pure addition: one new backend in `local_llm.py` (+ spec wiring), one new bench, one new test.
Default behavior unchanged. The moat-seal guard scans the additions and must stay green
(vendor-neutral, no private namespaces).
