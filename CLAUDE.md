# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG application (a teaching-assistant Q&A bot over 8 VTT lecture transcripts in `data/`) plus a
full DeepEval-based evaluation suite around it. The interesting part of the repo is the eval suite
and its regression-testing machinery, not the RAG app itself.

Requires `OPENAI_API_KEY` in a `.env` at the project root (loaded via `load_dotenv()` in almost
every module). Python 3.11, managed with `uv`.

## Commands

All commands run **from the project root** — `data/` and `chroma_store/` are relative paths, and
the eval modules import `src.*`, so always use `python -m` (or `uv run python -m`) rather than
`python evals/foo.py`.

```bash
uv sync                                      # install deps from uv.lock
uv run streamlit run src/app.py              # chat UI (fetch_k / top_k sliders in the sidebar)
uv run python -m src.rag_pipeline            # smoke-test the pipeline end to end
uv run python src/retriever.py               # build chroma_store/ (first run embeds; ~one-time)
uv run python export_chroma_chunks.py        # dump every stored chunk -> chunks_dump.json
```

Running evals — each eval is the unit test here; there is no pytest suite despite the `pytest`
dependency:

```bash
# one eval at a time (each builds its own pipeline)
uv run python -m evals.eval_retriever        # contextual recall + precision
uv run python -m evals.eval_generator        # faithfulness + answer relevancy on GOLDEN context
uv run python -m evals.eval_rag_pipeline     # the RAG triad on live pipeline output
uv run python -m evals.eval_application      # GEval correctness / completeness / style
uv run python -m evals.eval_safety           # scope + leakage(protected,pii) + toxicity
uv run python -m evals.eval_ops              # latency + cost + reliability

# the full suite -> a snapshot JSON
uv run python -m evals.run_suite --baseline --label "base k=5"   # -> baselines/baseline.json
uv run python -m evals.run_suite --label "k=10 + reranker"       # -> baselines/candidate.json
uv run python -m evals.run_suite --full --quiet                  # keep info metrics; less chatter

# the verdict (exit code: PASS=0, FAIL=1, REVIEW=2)
uv run python -m evals.compare
uv run python -m evals.compare --baseline a.json --candidate b.json --all
```

Every eval makes many live OpenAI calls (pipeline generation + `gpt-4o-mini` judge). A full
`run_suite` is slow and costs real money — prefer running a single eval while iterating.

`goldens/generate_goldens.py` regenerates *draft* goldens with the DeepEval `Synthesizer` and
**overwrites `goldens/retriever_deepeval_goldens.json`**. Drafts are meant to be reviewed by hand;
the goldens actually used by the suite are the hand-authored ones.

## Architecture

### The application (`src/`)

`retriever.py` → `reranker.py` → `generator.py`, composed by `rag_pipeline.py`:

- **retriever.py** — strips VTT timestamps, tags each transcript with a `session` metadata field
  parsed from the filename, chunks at 1000/150, embeds with `text-embedding-3-large` into a
  persisted Chroma store at `chroma_store/`. `load_store()` is the single entry point and is
  **build-or-load**: if `chroma_store/` exists it never re-embeds, so any change to chunking or the
  embedding model requires deleting `chroma_store/` first.
- **reranker.py** — `RerankingRetriever` over-fetches `fetch_k` with the bi-encoder, then reranks
  with the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder and keeps `top_k`. Downloads ~80MB
  on first use.
- **generator.py** — `gpt-4o-mini`, temperature 0, behind one long faithfulness-first prompt. The
  prompt is also the safety surface: scope, toxicity, prompt-leakage, protected-content and PII
  rules all live in it, and the transcript context and student question are wrapped in
  `<COURSE_CONTEXT>` / `<STUDENT_QUESTION>` blocks that the prompt declares untrusted. Editing this
  prompt is the main lever the safety evals are measuring — `run_suite` stamps a `prompt_hash` into
  every snapshot so silent prompt edits surface in a diff. Exposes `generate()` and
  `generate_stream()` (same chain; the streaming variant is what the latency eval clocks TTFT on),
  plus `prompt` and `llm`, which `eval_ops` recomposes as `prompt | llm` to read token usage off
  the `AIMessage` (the `StrOutputParser` in `generate()` discards it).
- **rag_pipeline.py** — `RagPipeline.invoke(query)` returns `{"query", "context", "answer"}`. That
  three-key dict is the contract every eval depends on: it is the three legs of the RAG triad, so
  one call scores retrieval and generation together.

There is no central config module. `fetch_k`/`top_k`, the models, chunk size, and the judge model
are hardcoded at their point of use (and duplicated in eval `hyperparameters` dicts, which can and
do drift from the real values).

### The eval suite (`evals/`)

Three layers, and it matters which is which:

1. **Per-eval modules with a `run(...)` function** — `eval_retriever`, `eval_generator`,
   `eval_rag_pipeline`, `eval_application`, `eval_safety` (`run_safety`), `eval_ops` (`run_ops`).
   These take an already-built pipeline/retriever as an argument and *return* a metrics dict. These
   are the ones the orchestrator uses.
2. **`run_suite.py`** — the orchestrator. Builds **one** `RagPipeline` and injects it into every
   eval, so a snapshot is provably the measurement of a single pipeline. Flattens each eval's output
   into a 3-level dotted metric id space (`retriever.contextual_recall.pass_rate`,
   `safety.scope.avg_score`, `ops.latency.e2e_p95_ms`), stamps metadata (git sha, prompt hash,
   label, elapsed), and writes a snapshot JSON. Quality evals return nested
   `{metric: {stat: val}}` (via `harness.summarize_by_metric`) and are flattened with
   `flatten_nested`; safety/ops return already-flat dicts and are prefixed with `prefix_flat`.
3. **`metric_registry.py` + `compare.py`** — the decision layer. `rule_for(metric_id)` resolves any
   dotted id (by suffix/prefix pattern, not an explicit table) to a rule: `direction`
   (higher/lower is better), `kind` (`gate` blocks, `guardrail` needs human review, `info` never
   affects the verdict), and a tolerance (absolute + relative) that defines the noise band.
   `compare.py` classifies every metric and returns PASS / REVIEW / FAIL.

Key conventions encoded in the registry — read its docstring before changing thresholds:

- Judge metrics are gated on **`avg_score`**, not `pass_rate` (pass rate is threshold-anchored and
  swings wildly when scores cluster near the threshold). `pass_rate` is kept as `info`.
- `safety.*avg_score` are **hard gates** (tol 0.02); other `*avg_score` are quality guardrails
  (tol 0.05, above the measured ~0.033 judge noise floor). `safety.toxicity.avg_toxicity` is
  lower-is-better.
- Only `ops.latency.e2e_p95_ms` (25% rel tol), `ops.cost.cost_per_query_usd` (15%),
  `ops.reliability.success_rate` / `error_rate`, and `*_pass` SLO booleans drive the verdict. TTFT
  is deliberately `info` — its run-to-run noise measured ~80%.
- `run_suite` drops `info` metrics from the snapshot unless `--full`, using the same registry that
  decides the verdict (~88 metrics → ~22).

`harness.py` holds `load_goldens`, `summarize_by_metric`, `print_summary`. The metric extractors in
both `harness.py` and `eval_safety.py` are deliberately defensive across DeepEval versions
(`.test_results` or a bare list; `.metrics_data` or `.metrics`) and key pass/fail off each metric's
own `success` flag so they stay direction-safe.

### Superseded standalone eval scripts

`eval_cost.py`, `eval_latency.py`, `eval_reliability.py` (merged into `eval_ops.py`),
`eval_leakage.py`, `eval_scope_safety.py`, `eval_toxicity.py` (merged into `eval_safety.py`), and
`eval_retriever_with_reranker.py` are earlier single-concern versions kept for reading. The safety
three and `eval_retriever_with_reranker.py` have **no `__main__` guard — they execute the whole eval
at import time**, so never import them from other code. Fix bugs in the merged `eval_ops.py` /
`eval_safety.py`, which are what `run_suite` actually calls.

### Goldens (`goldens/`)

Hand-authored, 15 rows each, and each has its own schema — check the keys before wiring a new eval:

| file | keys | used by |
|---|---|---|
| `retriever_goldens.json` | `id, query, ideal_answer, source` | `eval_retriever` |
| `faithfulness_dataset.json` | `id, query, ideal_context, source_sessions` | `eval_generator` (uses `ideal_context` for isolation), `eval_rag_pipeline` (queries only) |
| `correctness_goldens.json` | `id, question, ideal_answer, source_session` | `eval_application` (note: `question`, not `query`) |
| `scope_goldens.json` | `id, case_type, technique, input, expected_action, success_criteria` | scope GEval; `expected_action` + `success_criteria` are folded into `expected_output` as judge ground truth |
| `leakage_goldens.json` | `id, subtype, case_type, technique, input, expected_action` | split on `subtype`: `prompt`/`course_content` → protected-info GEval, `pii` → `PIILeakageMetric` |
| `toxicity_goldens.json` | `id, case_type, technique, input` | `ToxicityMetric` |

Safety goldens carry ANSWER / DECLINE / PARTIAL `expected_action`s — the judges are told to treat
that as ground truth rather than re-deciding scope themselves, and to judge only their own
dimension (scope judges scope, not correctness or style). Keep that single-dimension framing when
adding rubrics.

`baselines/` is created on demand by `run_suite`; `chroma_store/`, `.deepeval*`, and
`chunks_dump.json` are generated and gitignored. `uv.lock` is intentionally committed.
