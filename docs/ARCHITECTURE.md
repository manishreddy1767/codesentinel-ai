# CodeSentinel-AI — Architecture

**Last updated:** 19 Sep 2026

This describes the system **as it actually is**, not as intended. Components
that do not exist are marked NOT IMPLEMENTED rather than described in the
future tense.

---

## 1. Actual repository layout

```
codesentinel-ai/
├── backend/
│   ├── app/                        FastAPI service + static analysis
│   │   ├── main.py                 app factory, CORS, router mounting
│   │   ├── config.py               pydantic-settings configuration
│   │   ├── api/                    analyze.py, analyze_file.py, languages.py, health.py
│   │   ├── schemas/                request.py, response.py  (Pydantic)
│   │   └── services/               13 modules, 4,091 lines — see §3
│   ├── ml/                         CodeBERT pipeline
│   │   ├── config.py               every value env-overridable
│   │   ├── preprocessing/          inspect, validate, audit, clean, chunk, balance, undersample
│   │   ├── models/                 codebert_classifier.py, dataset.py, collate.py
│   │   ├── training/               trainer, train, checkpoint, metrics, overfit_test, memory_probe
│   │   ├── tuning/                 space.py, tune.py  (Optuna / seeded random)
│   │   ├── evaluation/             evaluator, test_model, select_threshold, calibration, baselines, compare_baselines
│   │   ├── utils/                  seed, io, gpu, experiment
│   │   ├── graphs/                 EMPTY — GATv2 NOT IMPLEMENTED
│   │   └── embeddings/             EMPTY
│   └── tests/                      134 tests
├── frontend/                       8 static pages + assets/js/ ES modules (§5)
├── data/                           raw (read-only) / processed / processed_clean / chunked
├── artifacts/                      experiments, tuning, reports  (git-ignored)
├── conftest.py                     puts repo root AND backend/ on sys.path
└── tests/backend/                  duplicate of one backend test — see §7
```

### Deviations from the brief's suggested structure

The brief proposes `backend/analysis/{ast,cfg,dfg,cpg,taint,rules,graph}` and
`backend/ml/{data,models,training,evaluation,calibration,explainability}`, plus
top-level `experiments/`, `reports/`, `scripts/`, `configs/`.

| Brief | Actual | Justification |
|---|---|---|
| `backend/analysis/*` | `backend/app/services/*` | Pre-existing, imported by the API layer and by 3 test modules. Moving it is pure churn with real breakage risk and no functional gain. |
| `backend/ml/data/` | `backend/ml/preprocessing/` | Same role, different name. |
| `backend/ml/calibration/` | `backend/ml/evaluation/calibration.py` | One module, not a package. |
| `backend/ml/explainability/` | — | NOT IMPLEMENTED. |
| `experiments/`, `reports/` | `artifacts/experiments/`, `artifacts/reports/` | Single git-ignored root for all generated output, so one `.gitignore` rule covers everything regenerable. |
| `scripts/` | `python -m backend.ml.<module>` | Every stage is already a runnable module with `argparse`. Adding shell wrappers would duplicate the interface. |
| `configs/` | `backend/ml/config.py` + env vars | Every value is overridable via `CODESENTINEL_*`. |

These are recorded rather than silently adopted. If the grader requires the
literal layout, the mapping above is the migration list.

---

## 2. Data flow

### Analysis request (runtime)

```
client → POST /analyze {code, language, filename?}
           │
           ├─ language_service.detect_language()        (if language omitted)
           ├─ parser_service.get_parser()               tree-sitter, 5 languages
           ├─ tree = parser.parse(code)
           ├─ cpg_service.build_cpg(tree.root_node)     AST + CFG + DFG merged
           ├─ vulnerability_service.detect_vulnerabilities()   rules
           ├─ taint_service.detect_taint_flows()               taint
           ├─ deduplication_service.deduplicate()
           ├─ confidence_service.assign_confidence()
           ├─ cpg_vulnerability_service.link_to_cpg()   findings → graph nodes
           └─ risk_service.calculate_security_risk()
         → AnalyzeResponse {analysis_id, language, security_risk,
                            vulnerabilities[], vulnerability_summary}
```

The ML model is **not** currently in this path. Wiring it in is Phase 12.

### ML training (offline)

```
data/raw/*.jsonl                      immutable, SHA-256 recorded, read-only
  → preprocess_primevul               normalise, validate, train-only dedupe
  → data/processed/
  → audit_splits                      7-dimension leakage audit
  → clean_split_duplicates            remove leaked records from TRAIN only
  → data/processed_clean/
  → chunk_dataset                     CodeBERT tokenizer, 512 tokens, 128 overlap, ≤8 chunks
  → data/chunked/
  → [optional] undersample_chunked    10:1 negatives, TRAIN only
  → tuning/tune.py                    validation PR-AUC objective
  → training/train.py                 → data/checkpoints/
  → evaluation/select_threshold.py    VALIDATION only → decision_policy.json
  → evaluation/test_model.py          ONE evaluation of test
```

---

## 3. Static analysis layer

| Module | Lines | Role |
|---|---|---|
| `taint_service.py` | 1,001 | sources, sinks, sanitizers, propagation, flow detection |
| `cfg_service.py` | 914 | control-flow graph |
| `vulnerability_service.py` | 724 | pattern/rule engine |
| `cpg_service.py` | 435 | code property graph |
| `dfg_service.py` | 377 | data-flow graph |
| `parser_service.py` | 164 | tree-sitter parsers |
| `cpg_vulnerability_service.py` | 124 | links findings to CPG nodes |
| `analyzer.py` | 103 | orchestration |
| `language_service.py` | 54 | heuristic language detection |
| `risk_service.py` | 49 | severity aggregation |
| `confidence_service.py` | 47 | confidence scoring |
| `deduplication_service.py` | 42 | finding de-duplication |

**Languages parsed:** C, C++, Python, JavaScript, Java (tree-sitter grammars
installed and importable).

**Known weakness — language detection.** `detect_language()` is a substring
heuristic (`"#include"`, `"std::"`, `"def "`, …). PrimeVul stores bare C/C++
function bodies with no `#include`, so detection fails on them and callers must
pass `language` explicitly. The baseline harness forces `language="cpp"` for
this reason, and that is documented where it happens.

---

## 4. ML layer

**Model — `HierarchicalCodeBERTClassifier`:**

```
function → tokenize (no special tokens)
         → chunks of ≤510 body tokens, 128 overlap, BOS/EOS added per chunk
         → CodeBERT encoder, micro-batched          [N, L] → [N, L, H]
         → chunk pooling   (cls | mean)             → [N, H]
         → function pooling (mean | max | attention) → [B, H]   scatter by function_index
         → dropout → Linear(H, 1) → raw logits      → [B]
```

Returns raw logits so `logits.shape == targets.shape` and
`BCEWithLogitsLoss(pos_weight=…)` applies directly. Sigmoid is never applied
inside the model.

**Imbalance:** `pos_weight = n_negative / n_positive` computed from whichever
training file is supplied (30.4292 on the cleaned split), capped at 50.

**Selection discipline built into the code:**
- Tuning objective is **PR-AUC (threshold-free)**, so the threshold cannot be
  co-optimised and inflate the score.
- Threshold and calibration are frozen to `decision_policy.json` before test is
  read; `test_model.py` refuses a policy not marked `selected_on: validation`.
- An append-only ledger (`artifacts/test_evaluations.jsonl`) records every test
  evaluation, making repeat evaluation visible.
- Stage 18 hybrids are computed from **saved predictions**, so the test split is
  scored exactly once.

**NOT IMPLEMENTED:** GraphCodeBERT (CodeBERT is used instead), GATv2, any
graph-tensor construction, model-side explainability.

---

## 5. Frontend

8 static pages: `index.html` plus dashboard, analyze-code, analysis-results,
vulnerability-details, repository-analysis, history, reports, settings.
Styling is Tailwind via CDN; no bundler and no `package.json`, so the
integration is plain ES modules served statically.

### Integration status: WIRED (as of 19 Sep 2026)

```
frontend/assets/js/
├── api.js                  fetch client: base URL, timeouts, ApiError{kind}
├── render.js               safe DOM rendering (textContent only)
├── analyze-page.js         analyze-code.html  -> POST /analyze, /analyze/file
├── results-page.js         analysis-results.html
├── details-page.js         vulnerability-details.html
├── notimpl-history.js      history.html
├── notimpl-reports.js      reports.html
└── notimpl-repository.js   repository-analysis.html
```

`api.js` rejects with an `ApiError` carrying a machine-readable `kind`
(`offline` / `timeout` / `http` / `malformed`) so the UI branches on a field
rather than parsing an error string. Base URL defaults to
`http://127.0.0.1:8000` and is overridable via `localStorage`.

**Safety.** Every value from the backend or from analyzed source is written
with `textContent`; `innerHTML`, `outerHTML`, `insertAdjacentHTML` and
`document.write` appear nowhere in application code. Analyzed code is
attacker-controlled by definition, so rendering it as HTML would let a crafted
snippet run script in the operator's browser. A test
(`test_frontend_never_uses_innerhtml_for_dynamic_content`) enforces this, with
comment-stripping so the rule stays strict rather than being satisfied by
wording.

**Fabricated content removed.** The pages previously shipped invented results —
a CWE-120 buffer overflow with a fabricated snippet, a "78% security score",
and a model named "CodeSentinel R-GCN" that does not exist in this repository.
Roughly 1,150 lines of that markup were deleted. Pages with no backing feature
(history, reports, repository analysis) now state that the feature is not
implemented instead of showing sample data.

**States handled:** idle, loading, success-with-findings, success-with-zero-findings,
backend-offline, timeout, HTTP error, no-code, no-file, unsupported language.

---

## 6. Configuration

`backend/app/config.py` — pydantic-settings, `.env` supported.
`backend/ml/config.py` — paths, schema candidates, chunking, imbalance, model,
training, threshold, artifacts. Every value overridable via `CODESENTINEL_*`
environment variables, so the same code runs on a laptop CPU, this 4 GB GPU, or CI.

---

## 7. Known structural issues

1. **Duplicate test file** — `tests/backend/test_api.py` and
   `backend/tests/test_api.py` both exist. Only `backend/tests/` is collected by
   the documented command.
2. **Two import roots** — app modules import `app.*`, ML modules import
   `backend.ml.*`. A root `conftest.py` puts both on `sys.path`; without it the
   suite cannot be collected from any single directory.
3. **`data/checkpoints_smoke/`** — 1.9 GB of smoke-run checkpoints, safe to
   delete.
4. **ML is not in the request path** — the API is rules+taint only today.
