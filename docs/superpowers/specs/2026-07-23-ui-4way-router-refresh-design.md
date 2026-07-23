# Design: 4-way router refresh + explainability for the demo UI

**Date:** 2026-07-23
**Status:** Approved (in conversation)

## Problem

The localhost testing/demo UI (`src/rag_vie/api/` + `static/`) is out of date with the
current RAG pipeline in two ways:

1. **It only shows three retrieval channels.** The pipeline is now 4-way
   (dense + BM25 + sparse + **toneless**), but the API defaults to a 3-way
   checkpoint (`fusion_mlp_multidomain.keras`, `input_dim=8, output_dim=66`),
   so the router never emits a toneless weight and the UI never renders the 4th
   channel.
2. **It shows the router's output but not its input.** The 8 Vietnamese
   linguistic features that drive the MLP router are the core contribution, yet
   the UI never surfaces them — you cannot see *why* a query routes the way it does.

## Non-goals (YAGNI)

- No new dataset / no new embedding. Use the existing indexes as-is
  (`viaquad`, `dangdocao`, `multidomain`), each of which already has
  `sparse.pkl` + `bm25_toneless.pkl`.
- No per-channel raw-hits view.
- No exposing the 28-dim retrieval signals in the UI (too technical for a demo;
  the 8 linguistic features tell the routing story).
- No UI framework — stays vanilla JS.

## Design

### Backend

1. **`service.py` — default to the 4-way checkpoint.**
   `default_mlp_path()` returns `checkpoints/fusion_mlp_4way_aug.keras`
   (verified `input_dim=36, output_dim=286`) when present, else falls back to
   the existing 3-way `fusion_mlp_multidomain.keras`. The 4-way checkpoint works
   with all three existing indexes.

2. **`pipeline.py` — surface the 8 linguistic features.**
   `RAGResult` gains `features: dict[str, float]` (default `{}`, so the CLI is
   unaffected). `_predict_mlp_weights()` returns `(weights, feature_dict)` where
   the dict maps `FEATURE_NAMES` → value, built only from the 8 linguistic
   features (guarded to the heuristic path; skipped if a neural extractor
   returns a differently-shaped vector). `run()` and `compare()` populate it
   (query-level, identical across compare methods).

3. **`schemas.py`** — `QueryResponse` and `CompareResponse` gain
   `features: dict[str, float] = {}`.

4. **`app.py`** — populate `features` on both responses. For **compare only**,
   run through `{**BASELINE_METHODS, "toneless": (0,0,0,1)}` so the 4th channel
   has a single-channel baseline in the table, and add its `_METHOD_LABELS`
   entry. `BASELINE_METHODS` itself is left untouched so offline
   `evaluate_all.py` semantics do not drift.

### Frontend (`static/`)

5. **Router explainability panel** (new, in the results view) — renders the 8
   features as labeled bars with plain-Vietnamese tooltips (feature labels +
   descriptions live in `app.js`, keyed by `FEATURE_NAMES`; the API stays
   label-free). Shown in both single and compare views (features are
   query-level).

6. **"Bỏ dấu & chạy lại" button** — strips diacritics from the current query
   (`NFD` + strip combining marks + `đ→d`) and re-runs the single query, so one
   click demonstrates the router shifting weight toward the toneless channel on
   a tone-less query.

7. **4-channel legend** near the weights, and refreshed sample queries including
   a naturally tone-less / code-switch example.

8. **`style.css`** — styles for the features panel, legend, and de-accent button.

## Data flow (after)

```
query → extract_features (8 linguistic) + retrieve 4 channels
      → 4-way MLP (Grid NDCG predictor) → weights (4)
      → fuse → generate
API returns { weights(4), features(8), retrieved, answer, latency }
UI shows   features → weights → passages → answer
```

## Testing

- Existing `tests/test_api.py` must stay green (health, 404s, empty-index,
  static served, 422 on empty query). Adding the `features` field is additive
  and non-breaking.
- Add a pipeline unit test asserting `RAGResult.features` carries all 8
  `FEATURE_NAMES` keys.
- Manual: launch `uvicorn`, run a normal query and its de-accented variant,
  confirm 4 channels render and toneless weight rises on the tone-less query.

## Risk

Running a query at demo time still needs the FPT API (query embedding for the
dense channel + LLM generation). If the FPT key has no credit, the dense channel
and answer generation fail at request time regardless of these code changes —
independent of this refresh, surfaced here for awareness.
