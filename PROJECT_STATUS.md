# CodeSentinel-AI — Project Status

**Last updated:** 26 Sep 2026
**Git commit:** `b681ff3` (working tree clean, pushed)

> ## TRAINING PAUSED — 26 Sep 2026, 11:50
>
> Stopped deliberately at the user's request before unplugging from AC power.
> This was not a failure. On battery the GPU is clamped to 30 W instead of 60 W,
> which is the condition associated with all three earlier crashes, so training
> on battery would risk another `nvlddmkm` reset rather than make progress.
>
> **State, verified after the stop:**
>
> | Item | Value |
> |---|---|
> | Stopped at | epoch 3 of 3, batch 751 of 7,632, global step 16,015 |
> | Restarts during this run | 0 (attempt 1 of 20) |
> | `latest_checkpoint.pt` | **26 of 26 integrity checks passed**, resumable |
> | `best_codebert.pt` | **16 of 16 passed**, holds **epoch-2** weights |
> | Best validation PR-AUC | **0.13076** (epoch 2), F1 0.22685 |
> | Test split | **untouched** — `artifacts/test_evaluations.jsonl` absent |
> | GPU after stop | 0% util, 0 MiB, no compute processes |
>
> Both checkpoints were verified *after* the process was killed, so neither was
> left half-written. Training logs were copied out of `/tmp` to
> `artifacts/logs/` (gitignored, not committed) so they survive temp cleanup.
>
> **To resume, on AC power:** re-run the `train_supervised.py` command in
> `reports/REPRODUCIBILITY.md` §5. It resumes from the checkpoint automatically;
> roughly 6,900 batches of epoch 3 remain, about 95 minutes.
>
> **Before any test evaluation**, `select_threshold.py` must be re-run on
> whatever checkpoint ends up in `best_codebert.pt`. The existing
> `decision_policy.json` is fitted to the **epoch-1** model and is stale — epoch
> 2 overwrote the weights it was derived from. See `reports/FINAL_EVALUATION.md`
> §6.
>
> Epochs 1 and 2 are complete and their results are recorded in
> `reports/FINAL_EVALUATION.md` §4c. Epoch 3 is the only incomplete step; the
> decision on whether to train longer than 3 epochs was deliberately deferred
> until its validation figure exists, because two points cannot distinguish
> sustained improvement from one good epoch.

Status vocabulary, used strictly:

| Status | Meaning |
|---|---|
| **VERIFIED** | Implemented, tested, and executed on real data |
| **IMPLEMENTED BUT NOT VERIFIED** | Code exists and unit-tests pass, but it has not run on real data |
| **SYNTHETIC ONLY** | Exercised only on generated fixtures |
| **BLOCKED** | Cannot proceed; blocker named |
| **FAILED** | Attempted and did not work |
| **NOT RUN** | Not attempted |
| **NOT IMPLEMENTED** | No code exists |

---

## 1. Environment (inspected 19 Sep 2026)

| Item | Value |
|---|---|
| OS / shell | Windows 11 Home Single Language · MINGW64 (Git Bash) + PowerShell |
| CPU | AMD Ryzen 7 5800H — 8 cores / 16 threads |
| RAM | 15.3 GB |
| Disk free (C:) | 250.6 GB |
| GPU | NVIDIA RTX 3050 Laptop — **4.0 GiB VRAM**, capability 8.6, bf16 supported |
| Python | 3.14.3 (`.venv`) |
| PyTorch | 2.14.0+**cu130** — `cuda.is_available()` True, **verified by a real matmul** |
| Node / npm | v24.19.0 / 11.17.0 (present, unused by the current frontend) |
| transformers / sklearn / numpy / scipy | 5.17.0 / 1.9.1 / 2.5.3 / 1.18.1 |
| optuna | 5.0.0 |
| tree-sitter (+c, cpp, python, java, javascript) | 0.26.0 |
| fastapi / pydantic-settings / httpx | 0.141.1 / 2.15.0 |
| **torch_geometric** | **NOT INSTALLED** — required for GATv2 (Phase 10) |

**GPU note.** During the previous session the CUDA primary context wedged after
~4 h of sustained load (`cuDevicePrimaryCtxRetain` returning 999) while the GPU
was idle and cool. It recovered when the holding processes were torn down. This
is a known risk for long runs on this machine; per-epoch and mid-epoch
checkpointing with verified resume is the mitigation.

---

## 2. Phase status

| Phase | Area | Status | Evidence / blocker |
|---|---|---|---|
| 1 | Project inspection | **VERIFIED** | This document, `docs/ARCHITECTURE.md`, `docs/IMPLEMENTATION_PLAN.md` |
| 2 | Architecture | **IMPLEMENTED BUT NOT VERIFIED** | Structure differs from the brief; deviations documented in ARCHITECTURE.md |
| 3 | Backend API | **PARTIAL** | `/`, `/health`, `/analyze`, **`/supported-languages`**, **`/analyze/file`** VERIFIED. Still missing: `/analyze/repository`, `/analysis/{id}` (need persistence) |
| 4 | AST / CFG / DFG / CPG | **IMPLEMENTED BUT NOT VERIFIED** | 4,091 lines across 13 services; tests exist but per-language coverage unconfirmed |
| 5 | Rules engine | **IMPLEMENTED BUT NOT VERIFIED** | `vulnerability_service.py` (724 lines) + `test_vulnerability_service.py` |
| 6 | Taint analysis | **IMPLEMENTED BUT NOT VERIFIED** | `taint_service.py` (1,001 lines) + `test_taint_service.py` |
| 7 | Dataset + leakage control | **VERIFIED** | PrimeVul downloaded, checksummed, read-only; 7-dimension audit; leakage found and removed |
| 8 | ML preprocessing | **VERIFIED** | 174,432 train / 25,430 valid / 25,911 test chunked; truncation quantified |
| 9 | ML baseline | **PARTIAL** | CodeBERT (**not** GraphCodeBERT); smoke + tuning runs only; **no final model trained** |
| 10 | GATv2 | **NOT IMPLEMENTED** | `backend/ml/graphs/` and `backend/ml/embeddings/` contain only empty `__init__.py` |
| 11 | Security-aware graph | **PARTIAL** | `cpg_vulnerability_service.py` (124 lines) links findings to CPG nodes; no ablation |
| 12 | Hybrid detection | **PARTIAL** | `compare_baselines.py` built; cannot run until a model exists |
| 13 | Training + memory tooling | **VERIFIED** | Overfit test, memory probe, checkpoint/resume all verified on real data |
| 14 | Hyperparameter tuning | **VERIFIED** | 4 Optuna trials on real data; best config selected |
| 15 | Threshold + calibration | **SYNTHETIC ONLY** | Tools built and unit-tested; not yet run on a real trained model |
| 16 | Experimental design | **PARTIAL** | Baselines done; ablations not run |
| 17 | Test evaluation | **NOT RUN** | **Test split has never been evaluated — guarantee intact** |
| 18 | Explainability | **PARTIAL** | Findings carry severity, confidence, location, remediation; no model-side attribution |
| 19 | Frontend | **VERIFIED (integrated)** | 8 ES modules wired to the live API; ~1,150 lines of fabricated findings removed; states + safe rendering enforced by tests |
| 20 | Documentation | **PARTIAL** | 6 ML docs exist; `DATASET_CARD.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`, `RESEARCH_CONTRIBUTION.md` missing |
| 21 | Testing | **PARTIAL** | **154 tests passing** (+20 frontend/API integration); no graph tests |
| 22 | Git + security | **VERIFIED** | No datasets, checkpoints or secrets tracked; `.gitignore` covers `data/`, `artifacts/`, `*.pt` |
| 23 | Completion audit | **NOT RUN** | Requires Phases 10, 17, 19 |

---

## 3. What is genuinely done and measured

Real PrimeVul results obtained so far (no synthetic numbers in this section):

| Item | Value |
|---|---|
| Dataset | train 184,427 / valid 25,430 / test 25,911 · ~3% positive |
| Leakage found | **796 test + 1,310 valid** functions were whitespace-variants of training functions |
| Leakage removed | 2,324 train records (1.3%); train↔valid and train↔test now clean on all 7 dimensions |
| Chunking | 298,192 train chunks, mean 1.71/function |
| Truncation | 2.57% of functions, but **22.5% of all tokens**; **vulnerable truncate 19.24% vs benign 2.43% (train), 20.86% vs 2.40% (test)** |
| `pos_weight` | 30.4292, measured from the cleaned training split |
| Rules-only baseline (full test) | P 0.1213 · R 0.0950 · **F1 0.1065** · PR-AUC 0.0378 (chance 0.0268) |
| Taint-only baseline (full test) | Flags 4 / 25,911 functions, **zero** true positives — no signal |
| Best tuned config | lr **1.827e-05**, dropout 0.480, wd 0.0732, warmup 0.090 → **val PR-AUC 0.1387** |
| Throughput | 10.5 functions/s at batch 8 + gradient checkpointing (2.5 GB peak) |

**No test-set model metric exists.** Any such number would be fabricated.

---

## 4. Critical gaps

### 4.1 Frontend — **RESOLVED 19 Sep 2026**

Previously 8 pages with zero application JavaScript, displaying invented
findings (a CWE-120 buffer overflow, a "78% security score", a nonexistent
"CodeSentinel R-GCN" model).

Now: 8 ES modules in `frontend/assets/js/` wired to the live API.
~1,150 lines of fabricated markup deleted. Pages backing unimplemented
features (history, reports, repository) state that plainly instead of showing
sample data. Two tests enforce it — one bans `innerHTML`/`document.write` in
application code, one bans hardcoded finding strings in the pages.

Remaining: history/reports/repository stay unimplemented until there is a
persistence layer.

### 4.2 GATv2 does not exist

Phase 10 is a headline requirement and there is no code for it.
`torch_geometric` is not installed. Nothing in the repository produces graph
tensors for a GNN.

### 4.3 No trained model

Phases 9, 12, 15, 17 all depend on a completed training run. Tuning selected a
configuration; final training has not been executed (~6 h estimated).

### 4.4 GraphCodeBERT vs CodeBERT

The brief specifies GraphCodeBERT; the implementation uses `microsoft/codebert-base`.
This is a real deviation and must either be changed or justified in writing.

---

## 5. Immediate next actions

1. **Phase 14** — final training, IN PROGRESS (resumed after a CUBLAS crash at batch 2,500)
2. **Phases 15–17** — freeze threshold + calibration on validation, then one test evaluation
3. **Phase 10** — GATv2 **descoped and documented** as NOT IMPLEMENTED (`reports/RESEARCH_CONTRIBUTION.md` §4)
4. **Phase 20** — `DATASET_CARD.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md` still missing
5. **Phase 23** — `reports/COMPLETION_AUDIT.md` written, interim until Phase 17 lands

See `docs/IMPLEMENTATION_PLAN.md` for sequencing and effort.
