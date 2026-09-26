# CodeSentinel-AI — Completion Audit

**Last updated:** 19 Sep 2026
**Audit state:** INTERIM — final training in progress, test split not evaluated

Status vocabulary, used strictly. No other words are used.

| Status | Meaning |
|---|---|
| **VERIFIED** | Implemented, tested, and executed on real data |
| **IMPLEMENTED BUT NOT VERIFIED** | Code exists and passes unit tests, but has not run on real data |
| **BLOCKED** | Cannot proceed; blocker named |
| **NOT IMPLEMENTED** | No code exists |
| **NOT RUN** | Implemented, not executed |

**Test suite: 154 passing** (was 58 at project start).

---

## 1. Requirement table

| Requirement | Status | Evidence | Remaining limitation |
|---|---|---|---|
| **Backend** |
| FastAPI service | VERIFIED | `main.py`, live server tested | — |
| `GET /health` | VERIFIED | returns 200 | no version/dependency detail |
| `GET /supported-languages` | VERIFIED | 5 languages, parser availability probed | — |
| `POST /analyze` | VERIFIED | live: finds `strcpy` at line 2 | ML model not in this path |
| `POST /analyze/file` | VERIFIED | 7 tests incl. traversal, binary, oversize | no archive support |
| `POST /analyze/repository` | NOT IMPLEMENTED | — | descoped; page states this |
| `GET /analysis/{id}` | NOT IMPLEMENTED | — | needs persistence layer |
| Pydantic validation | VERIFIED | 422 on bad language/empty code | — |
| CORS for `localhost:5500` | VERIFIED | preflight returns the origin | — |
| Malformed source handling | VERIFIED | returns 200, not 500 | — |
| No arbitrary code execution | VERIFIED | uploads parsed, never executed | — |
| **Static analysis** |
| AST extraction | IMPLEMENTED BUT NOT VERIFIED | `ast_service.py` + parsers | per-language coverage untested |
| CFG construction | IMPLEMENTED BUT NOT VERIFIED | `cfg_service.py` (914 lines) | unsupported constructs undocumented |
| DFG construction | IMPLEMENTED BUT NOT VERIFIED | `dfg_service.py` (377 lines) | interprocedural scope unclear |
| Code property graph | IMPLEMENTED BUT NOT VERIFIED | `cpg_service.py` (435 lines) | — |
| Security-aware graph | IMPLEMENTED BUT NOT VERIFIED | `cpg_vulnerability_service.py` | links findings only; never a model input |
| 5 languages parse | VERIFIED | all report `parser_available: true` | rules/taint depth per language unmeasured |
| **Rules and taint** |
| Rules engine | VERIFIED (on real data) | test split: **F1 0.1065**, PR-AUC 0.0378 | 1.41× chance only |
| Taint engine | VERIFIED (on real data) | test split: 4 flags / 25,911, **0 TPs** | no measurable signal on PrimeVul |
| Deduplication | IMPLEMENTED BUT NOT VERIFIED | `deduplication_service.py` | — |
| Confidence scoring | IMPLEMENTED BUT NOT VERIFIED | `confidence_service.py` | not calibrated |
| **Dataset** |
| Official source + checksum | VERIFIED | SHA-256 in `data/raw_checksums.json` | — |
| Raw immutability | VERIFIED | read-only; re-verified after every stage | — |
| Leakage audit (7 dimensions) | VERIFIED | `artifacts/reports/leakage_raw.json` | normalized-hash finds copies, not clones |
| Leakage remediation | VERIFIED | 2,324 train records removed (1.3%) | valid↔test 14 functions remain |
| Test isolation | VERIFIED | ledger absent; never evaluated | — |
| **ML pipeline** |
| Preprocessing | VERIFIED | 174,432 / 25,430 / 25,911, zero silent drops | — |
| Chunking | VERIFIED | 298,192 chunks, mean 1.71 | **22.5% token loss; vulnerable truncate 7.9-8.7x more than benign** |
| Dataset/collate | VERIFIED | real-batch shape + mask assertions | — |
| Model forward/backward | VERIFIED | gradients non-zero in all 197 encoder tensors | — |
| Memory probe | VERIFIED | measured grid on real GPU | — |
| Overfit test | VERIFIED | loss −89.8%, train F1 0.980 | — |
| Checkpoint + resume | VERIFIED | resumed at batch 2,501 with RNG restored | — |
| Training smoke test | VERIFIED | full metric suite, early stopping, best-restore | — |
| Hyperparameter tuning | VERIFIED | 4 Optuna trials → val PR-AUC **0.1387** | 1 epoch, 9k subsample only |
| **Final training** | IN PROGRESS | resumed; epoch 1 of 3 | crashed once at batch 2,500 (CUBLAS) |
| Threshold optimisation | NOT RUN | tool built, unit-tested | needs trained model |
| Calibration | NOT RUN | Platt + isotonic built, unit-tested | needs trained model |
| **Test evaluation** | NOT RUN | — | **guarantee intact** |
| **Models** |
| GraphCodeBERT | NOT IMPLEMENTED | CodeBERT used instead | deviation documented in `RESEARCH_CONTRIBUTION.md` §3 |
| GATv2 | NOT IMPLEMENTED | `graphs/` is an empty package | scope impact documented §4 |
| Explainability (model-side) | NOT IMPLEMENTED | — | no attribution or attention analysis |
| Explainability (static) | IMPLEMENTED BUT NOT VERIFIED | severity, confidence, line, CWE, remediation, CPG links | not evaluated for usefulness |
| **Hybrid** |
| Rules + taint fusion | VERIFIED | `rules_taint` baseline measured | indistinguishable from rules alone |
| ML + rules fusion | NOT RUN | `compare_baselines.py` built | needs trained model |
| **Frontend** |
| Pages render | VERIFIED | 8 pages | — |
| **API integration** | VERIFIED | live `/analyze`, `/analyze/file`, `/supported-languages` | only analyze page is fully interactive |
| Loading / error / empty states | VERIFIED | all handled in `analyze-page.js` | — |
| No fabricated results | VERIFIED | ~1,150 lines removed; enforced by test | — |
| Safe rendering | VERIFIED | `textContent` only; enforced by test | — |
| Language list from backend | VERIFIED | populated from `/supported-languages` | — |
| History / reports / repository pages | NOT IMPLEMENTED | state the limitation instead of faking | need persistence |
| **Process** |
| Git hygiene | VERIFIED | no datasets, checkpoints or secrets tracked | 30 files uncommitted |
| Reproducibility docs | IMPLEMENTED BUT NOT VERIFIED | `EXPERIMENT_LOG.md` has exact commands | `REPRODUCIBILITY.md` not written |
| Dataset card | NOT IMPLEMENTED | — | `docs/DATASET_CARD.md` missing |
| Experiment protocol | NOT IMPLEMENTED | — | `docs/EXPERIMENT_PROTOCOL.md` missing |

---

## 2. The twenty audit questions

| # | Question | Answer |
|---|---|---|
| 1 | Backend fully functional? | **PARTIAL** — 4 of 6 endpoints. `/analyze/repository` and `/analysis/{id}` not implemented (no persistence layer). |
| 2 | Frontend integrated? | **PARTIAL** — analyze/results/details wired to the live API; history/reports/repository declare themselves unimplemented rather than faking data. |
| 3 | Supported languages tested? | **PARTIAL** — all 5 parse and are served by the API; rule/taint depth per language is not measured. C/C++ is the only one exercised at scale. |
| 4 | Rules engine tested? | **YES** — unit tests plus a full-test-split measurement (F1 0.1065). |
| 5 | Taint analysis tested? | **YES** — and the honest result is that it finds nothing on PrimeVul (0 TPs in 25,911). |
| 6 | AST/CFG/DFG tested? | **PARTIAL** — exercised indirectly through the analyzer; no dedicated per-construct tests for branches, loops, scopes across all 5 languages. |
| 7 | Security-aware graph implemented? | **PARTIAL** — built and used to attach findings to nodes; never used as a model input, so its value is unmeasured. |
| 8 | GraphCodeBERT implemented and tested? | **NO** — CodeBERT is used. Reason: GraphCodeBERT needs token-aligned data-flow inputs this pipeline does not produce; loading its weights without them would be the label without the mechanism. |
| 9 | GATv2 implemented and tested? | **NO** — not implemented. `backend/ml/graphs/` is empty; `torch_geometric` is not installed. |
| 10 | Real training completed? | **NOT YET** — in progress; crashed once at batch 2,500 and resumed from checkpoint. |
| 11 | Hyperparameter tuning completed? | **YES** — 4 Optuna trials on real data. Small budget (1 epoch, 9k subsample) forced by the 4 GB GPU. |
| 12 | Threshold optimisation completed? | **NO** — tool built and unit-tested; requires the trained model. |
| 13 | Calibration completed? | **NO** — same. |
| 14 | Test split evaluated correctly? | **NOT RUN** — never touched. Discipline enforced in code (policy must be `selected_on: validation`) and by an append-only ledger. |
| 15 | Baselines compared fairly? | **PARTIAL** — rules/taint measured on the full test split with identical data; the ML comparison awaits the model. |
| 16 | Ablations completed? | **NO** — graph ablations are unrunnable without a graph model. Chunking/pooling ablations are possible but not run. |
| 17 | Distribution shift considered? | **YES** — measured and documented: train is CWE-119/20/264 (linux/Chrome/qemu), test is CWE-617 (vim/gpac/tensorflow). |
| 18 | Explainability available? | **PARTIAL** — static evidence (line, CWE, severity, confidence, remediation, CPG links) is returned and rendered. No model-side attribution. |
| 19 | Results reproducible? | **PARTIAL** — seeds fixed, checksums recorded, env captured, exact commands logged. Not reproduced on a second machine; two CUDA crashes mean run-to-run stability is not guaranteed. |
| 20 | Suitable for preparing a research paper? | **NOT YET, and not the originally proposed paper.** The architecture contribution (security-aware graph + GATv2) does not exist. A narrower empirical paper on PrimeVul measurement practice is defensible once Stages 15–17 complete. |

---

## 3. Every NO or PARTIAL, with the remaining work

| Item | Why | Remaining work |
|---|---|---|
| Backend endpoints (Q1) | No persistence layer | SQLite store + `GET /analysis/{id}`; repository traversal with limits (~5 h) |
| Frontend partial (Q2) | 3 pages back unimplemented features | Implement persistence, then wire them (~3 h after backend) |
| Language coverage (Q3) | Only C/C++ exercised at scale | Per-language rule/taint test corpora (~4 h) |
| AST/CFG/DFG tests (Q6) | No per-construct assertions | Tests for branch, loop, call, assignment, syntax error × 5 languages (~6 h) |
| Security graph (Q7) | Never a model input | Requires the graph model; see Q9 |
| GraphCodeBERT (Q8) | Needs token-aligned DFG | Offline DFG extraction + re-chunk + re-tune + retrain (multi-day) |
| GATv2 (Q9) | Not implemented | Offline graph pipeline over 174k functions, node/edge schema, batching, training, evaluation (multi-day) |
| Training (Q10) | In progress | ~5 h remaining |
| Threshold / calibration (Q12, Q13) | Need the model | ~40 min after training |
| Test evaluation (Q14) | Deliberately last | ~20 min after threshold freeze |
| Baselines (Q15) | ML side missing | ~5 min after test evaluation |
| Ablations (Q16) | Graph ablations impossible | Chunking/pooling ablations ~4 h; graph ablations blocked on Q9 |
| Reproducibility (Q19) | Single machine | `REPRODUCIBILITY.md` + a second-machine run (~2 h + access) |
| Paper readiness (Q20) | Core claims unmeasured | Complete Q10–Q16, then draft the narrower empirical paper |

---

## 4. Statement of completeness

**The project is not complete.** It is not research-complete, and no part of
this audit should be read as claiming otherwise.

What is genuinely finished and measured: the data pipeline, the leakage audit
and its remediation, the training/tuning infrastructure, the non-ML baselines,
and the frontend-to-API integration.

What is not: the trained model's test performance, calibration, the hybrid
comparison, the graph model, and every experiment that depends on them.

Two proposed components — **GraphCodeBERT** and **GATv2** — are **NOT
IMPLEMENTED**, and their absence removes the originally proposed architectural
contribution. That is recorded in `reports/RESEARCH_CONTRIBUTION.md` §3–4
rather than worked around.
