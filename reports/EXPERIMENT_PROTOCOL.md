# Experiment Protocol — CodeSentinel-AI

**Last updated:** 19 Sep 2026
**Status:** final training in progress; Phases 4–9 not yet executed.

This document fixes the protocol **before** results exist, so that the
evaluation cannot be reshaped after seeing the numbers.

---

## 1. Split discipline

| Split | Permitted uses | Forbidden uses |
|---|---|---|
| **train** (174,432, cleaned) | gradient updates, `pos_weight` | — |
| **valid** (25,430) | early stopping, best-checkpoint selection, hyperparameter choice, threshold selection, calibration fitting | training gradients |
| **test** (25,911) | **exactly one** final evaluation after everything is frozen | any selection whatsoever |

### Enforcement in code, not just intent

1. Tuning objective is **PR-AUC**, which is threshold-free — so the threshold
   cannot be co-optimised with hyperparameters and inflate the reported score.
2. `select_threshold.py` writes `decision_policy.json` with
   `selected_on: "validation"`. `test_model.py` **refuses** any policy not
   carrying that marker.
3. Every test evaluation appends to `artifacts/test_evaluations.jsonl`. If that
   ledger ever lists more than one checkpoint, the held-out claim is void and
   the tool says so.
4. Stage-18 hybrids are computed from **saved per-sample predictions**, never
   by re-running the model, so the test split is scored once.

---

## 2. Model actually implemented

`HierarchicalCodeBERTClassifier` — `microsoft/codebert-base`.

```
function → tokenize (add_special_tokens=False)
         → chunks ≤510 body tokens, 128 overlap, BOS/EOS per chunk, ≤8 chunks
         → CodeBERT encoder, micro-batched
         → chunk pooling (cls)      → [N, H]
         → function pooling (mean)  → [B, H]   scatter by function_index
         → dropout → Linear(768, 1) → raw logits [B]
```

Loss: `BCEWithLogitsLoss(pos_weight=…)` on raw logits.

**NOT IMPLEMENTED, and not to be claimed:** GraphCodeBERT, GATv2, any graph
neural network, any graph tensor construction, model-side explainability.
See `reports/RESEARCH_CONTRIBUTION.md` §3–4.

---

## 3. Training configuration (frozen before the run)

Selected by Optuna on **validation PR-AUC only**, 4 trials × 1 epoch on a
9,000-function subsample:

| Trial | Learning rate | Dropout | Val PR-AUC |
|---|---|---|---|
| 1 | 1.841e-06 | 0.162 | 0.1110 |
| 2 | 4.328e-06 | 0.480 | 0.1259 |
| 3 | 1.285e-05 | 0.162 | 0.1275 |
| **4** | **1.827e-05** | **0.480** | **0.1387** ← selected |

PR-AUC rose monotonically with learning rate across all four trials.
`batch_size` and `grad_accum` were **pinned** (not searched) to the values
Stage-10 memory probing had already settled.

**Caveat:** dropout is confounded with learning rate here — both 0.480 trials
are also the higher-LR trials. No independent claim is made about dropout.

### Final run

```
train file      data/chunked/primevul_train_chunked_under10.jsonl  (61,050 fn, 10:1)
validation      data/chunked/primevul_valid_chunked.jsonl          (25,430 fn, real 35:1)
epochs          3            batch_size 8          grad_accum 1
lr              1.827e-05    dropout 0.480         weight_decay 0.0732
warmup_ratio    0.090        chunk_micro_batch 16  eval_batch_size 16
pos_weight      10.0 (from the undersampled file)
AMP             on (fp16 + GradScaler)   gradient checkpointing on
pin_memory      OFF (see §6)             seed 42
monitor         pr_auc       patience 2  min_delta 1e-4
checkpoint      every 250 batches + every epoch
```

---

## 4. Selection procedure

**Best checkpoint:** highest **validation PR-AUC**. PR-AUC is chosen over F1
because it is threshold-free — the monitored number cannot move just because
the tuned threshold moved — and because under 36:1 imbalance ROC-AUC stays
flatteringly high (demonstrated: a smoke model scored ROC-AUC 0.8277 while
predicting **zero** positives at threshold 0.5).

**Threshold (Phase 4):** grid over 0.05–0.95, 91 points, on validation
predictions only. Four objectives are computed and reported — `f1`, `fbeta`
(β=2), `mcc`, `recall_at_precision` — and the configured one is selected.
0.5 is **not** assumed optimal; it is reported as a comparison point only.

**Calibration (Phase 5):** adopted only if all three hold —
(a) measured ECE exceeds an analytic sampling-noise floor by ≥2×,
(b) out-of-fold ECE improves ≥10% relative,
(c) Brier does not regress.
Judged by stratified K-fold **inside** validation, so the improvement estimate
comes from data the calibrator never saw. Calibration changes probabilities,
not ranking — no PR-AUC/ROC-AUC improvement will be claimed from it.

---

## 5. Evaluation protocol

Reported for every approach: precision, recall, F1, PR-AUC, ROC-AUC, MCC,
Brier, confusion matrix, support, predicted-positive count, and the
always-positive PR-AUC baseline (0.0268 on test).

**Comparison set** — identical split, identical prediction unit (one label per
function), identical metric code:

| # | Approach | Status |
|---|---|---|
| 1 | rules_only | **measured** — F1 0.1065, PR-AUC 0.0378 |
| 2 | taint_only | **measured** — F1 0.0000, PR-AUC 0.0268 (4 flags, 0 TPs) |
| 3 | rules_taint | **measured** — F1 0.1065, PR-AUC 0.0377 |
| 4 | ML-only | pending training |
| 5 | ML ∪ rules / ML ∩ rules / ML+rules blended | pending |
| 6 | ML + taint | pending |
| 7 | full hybrid | pending |

Hybrids reuse the **frozen** ML threshold. Re-tuning per hybrid would be
selection on test.

---

## 6. Reliability protocol

The GPU (RTX 3050 Laptop, 4 GB) failed three times under sustained load:

| Event | Error | Position |
|---|---|---|
| Tuning trial 3 | pinned allocator abort | ~20 min in |
| Training run 1 | `CUBLAS_STATUS_EXECUTION_FAILED` | batch 2,500 |
| Training run 2 | `cudaErrorIllegalAddress` | batch 3,501 |

Mitigations, all in-repo:

- `pin_memory` defaults **off** (implicated in two aborts; transfer cost here
  is negligible — a batch moves ~14 chunks of int64 ids).
- `scripts/train_supervised.py` auto-resumes from checkpoint, and **requires
  forward progress**: two crashes at the same `global_step` aborts the loop
  rather than masking a deterministic bug.
- `scripts/verify_checkpoint.py` — 27 checks in a fresh process before any
  checkpoint is trusted.
- `scripts/gpu_monitor.py` — telemetry to `artifacts/reports/gpu_telemetry.jsonl`.

**Observation, not proven cause.** Windows logged 213 `nvlddmkm` Event-153
driver resets, with 12-event clusters at the two training-crash timestamps, and
no TDR (4101) or WHEA hardware events. The GPU was running on battery with
`enforced.power.limit = 30 W` against a 60 W default, continuously reporting
`SwPowerCap + SwThermalSlowdown`. On AC the limit rose to 60 W and
`SwThermalSlowdown` disappeared. Whether AC power *resolves* the instability is
**being tested by the current run** and is not yet established.

Any interrupted training will be disclosed in the final report with crash and
restart counts.

---

## 7. Planned error analysis (Phase 8)

From saved per-sample predictions — no extra test evaluation:
false positives/negatives by CWE and by project; performance on truncated vs
non-truncated functions (testing whether the 7.9-8.7x truncation asymmetry shows up
as a recall deficit); score distribution by class; rule/ML agreement matrix;
calibration on the held-out split.

## 8. Ablations that are actually runnable

Threshold objectives · calibration on/off · ML-only vs hybrid combinations ·
chunk-cap sensitivity (re-chunk at a different cap, compute cost permitting).

**Not runnable:** anything requiring a graph model or GraphCodeBERT. Those will
be reported as excluded, never estimated.
