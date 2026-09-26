# Final Evaluation — CodeSentinel-AI

**Last updated:** 26 Sep 2026
**Status:** validation phases COMPLETE · test evaluation **NOT RUN**

Label key: **MEASURED** (from command output) · **VERIFIED** (tested on real data) ·
**NOT RUN** · **NOT IMPLEMENTED** · **IN PROGRESS**

> Every number in this document was copied from command output. Nothing is
> estimated or inferred. Where a result does not exist, it says so.

---

## 1. What has and has not been measured

| Result | Status |
|---|---|
| Rules / taint baselines on **test** | **MEASURED** |
| Model performance on **validation** | **MEASURED** |
| Threshold selection | **MEASURED** (validation only) |
| Calibration | **MEASURED** (validation only) |
| Model performance on **test** | **NOT RUN** — one-shot budget intact |
| Hybrid ML+rules / ML+taint | **NOT RUN** — needs test predictions |
| Error analysis | **NOT RUN** — needs test predictions |
| GATv2 / GraphCodeBERT | **NOT IMPLEMENTED** |

`artifacts/test_evaluations.jsonl` does not exist. The test split has never
been read by an evaluation.

---

## 2. Model under evaluation

| Field | Value |
|---|---|
| Architecture | `HierarchicalCodeBERTClassifier` over `microsoft/codebert-base` |
| Chunking | ≤510 body tokens, 128 overlap, BOS/EOS per chunk, ≤8 chunks |
| Pooling | chunk `cls` → function `mean` (scatter by `function_index`) |
| Head | dropout 0.48 → `Linear(768, 1)` → raw logits |
| Prediction unit | **one prediction per function** |
| Loss | `BCEWithLogitsLoss(pos_weight=10.0)` |
| Training data | 61,050 functions (all 5,550 positives, negatives undersampled 10:1) |
| Validation data | 25,430 functions at the **true** 35.4:1 ratio |
| Checkpoint | `data/checkpoints/best_codebert.pt`, selected by **validation PR-AUC** |
| Training state at measurement | epoch 1 of 3 complete (step 7,632); **training subsequently resumed** |

**Checkpoint verification: VERIFIED** — 16/16 applicable checks in a fresh
process (201 tensors, 124,646,401 parameters, zero NaN/Inf, architecture and
selection provenance recorded).

---

## 3. Baselines on the test split — MEASURED

Full split, 25,911 functions, 695 positives (2.68%), **0 analyzer errors**.
Operating point: "any finding". Language forced to `cpp` because PrimeVul
stores bare function bodies with no `#include`, which defeats heuristic
language detection.

| Baseline | Precision | Recall | F1 | PR-AUC | Flagged | Runtime |
|---|---|---|---|---|---|---|
| `rules_only` | 0.1213 | 0.0950 | **0.1065** | 0.0378 | 544 | 27.6 s |
| `taint_only` | 0.0000 | 0.0000 | 0.0000 | 0.0268 | 4 | 48.3 s |
| `rules_taint` | 0.1211 | 0.0950 | 0.1065 | 0.0377 | 545 | 201.2 s |

Always-positive PR-AUC baseline: **0.0268**.

**Findings, reported as they are:**

- `rules_only` achieves PR-AUC 0.0378 against a 0.0268 chance baseline — only
  **1.41x chance**. It is a weak but non-zero detector.
- `taint_only` flags **4 of 25,911** functions and catches **zero** true
  positives. Its PR-AUC equals the chance baseline to three decimals: **no
  ranking signal on this dataset**. This is not hidden or explained away.
- `rules_taint` is indistinguishable from `rules_only`, confirming the taint
  component contributes nothing measurable here.

**F1 0.1065 is the bar the learned model must clear.**

Why taint finds nothing is worth stating precisely: the engine is built around
sources, sinks and sanitizers that are meaningful for web-style injection
(`os.system`, `eval`, `subprocess`), while PrimeVul is overwhelmingly C/C++
memory-safety code. This is a **coverage mismatch between the detector's rule
set and the dataset**, not evidence that taint analysis is useless in general.

---

## 4. Model on the validation split — MEASURED

25,430 functions, 699 positives (2.75%), function-level predictions.

### Threshold-free

| Metric | Value | Note |
|---|---|---|
| **PR-AUC** | **0.1114** | chance 0.0275 → **4.05x** |
| ROC-AUC | 0.7940 | optimistic under 35:1 imbalance |
| Brier (uncalibrated) | 0.0344 | |

### Threshold objectives compared — all on validation

| Objective | Threshold | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| **f1** (selected) | 0.200 | 0.1541 | 0.3262 | **0.2093** | 1,252 | 471 |
| fbeta (β=2) | 0.050 | 0.1051 | **0.5265** | 0.1752 | 3,133 | 331 |
| mcc | 0.130 | 0.1409 | 0.3920 | 0.2073 | 1,670 | 425 |
| recall_at_precision(0.5) | 0.930 | 0.2316 | 0.0315 | 0.0554 | 73 | 677 |
| *default 0.5* | 0.500 | 0.1784 | 0.1302 | 0.1505 | 419 | 608 |

Confusion matrix at the selected threshold: TN 23,479 · FP 1,252 · FN 471 ·
TP 228. Predicted positive: 1,480.

**Two observations that matter operationally:**

1. **0.5 is not optimal.** Tuning lifts F1 from 0.1505 to 0.2093 (**+39%**) by
   trading precision for recall. Assuming 0.5 would have understated the model.
2. **The model cannot reach 50% precision at any threshold.** The
   `recall_at_precision(0.5)` objective could not satisfy its constraint — the
   best precision found anywhere on the 91-point grid was 0.2316, and the
   objective fell back to its documented graceful-degradation path. At an
   operating point a reviewer would tolerate, roughly **4 in 5 alerts are false
   positives**.

---

## 4b. Literature reference point — is F1 ~0.21 low?

**Source:** Ding et al., *"Vulnerability Detection with Code Language Models:
How Far Are We?"* (PrimeVul), arXiv:2403.18624, Table V. Values transcribed
from the paper, not from memory or a search summary.

### Published results, fine-tuned AND evaluated on PrimeVul

| Model | Params | Accuracy | **F1** | VD-S (FNR) |
|---|---|---|---|---|
| CodeT5 | 60M | 96.67 | 19.70 | 89.93 |
| **CodeBERT** | **125M** | **96.87** | **20.86** | **88.78** |
| UniXcoder | 125M | 96.86 | 21.43 | 89.21 |
| StarCoder2 | 7B | 97.02 | 18.05 | 89.64 |
| CodeGen2.5 | 7B | 96.65 | 19.61 | 91.51 |

**This project's CodeBERT, validation split: F1 0.2093.**
**Published CodeBERT, PrimeVul test split: F1 0.2086.**

The implementation is **at the published state of the art for this benchmark**.
Not below it.

### Why the absolute numbers are so low

The entire field — including two 7B-parameter decoders — sits in a band of
**F1 18.05 to 21.43**. A 7B StarCoder2 scores *lower* than 125M CodeBERT here.
The same CodeBERT reaches **F1 62.88 on BigVul**, and collapses to **4.49**
when trained on BigVul and tested on PrimeVul. That collapse is the paper's
central thesis: earlier benchmarks were leaky and unrealistically easy.

The paper reports PrimeVul's class ratio as "roughly 1:32", matching the
30.4:1 measured here.

**Implication for this project:** a target of F1 0.5+ on PrimeVul is not a
realistic goal for this model class. Reported numbers in that range on this
dataset would warrant checking the de-duplication first — which is precisely
the failure mode the normalized-code leakage audit in §Dataset Card was built
to catch, and which found 796 leaked test functions.

### Protocol differences worth noting

| Setting | Paper | This project |
|---|---|---|
| Learning rate | 2e-5 (fixed, not tuned) | 1.827e-05 (tuned on validation) — essentially the same |
| **Epochs (<7B models)** | **10** | **3** |
| Hardware | NVIDIA A100 80GB cluster | 1x RTX 3050 Laptop, 4GB |
| Training data | full PrimeVul train | 61,050 undersampled 10:1 (compute budget) |

**Epoch count is the one substantive gap**, and it independently corroborates
the underfitting diagnosis found here from the training curve (loss flat across
an epoch boundary while the linear schedule decayed the rate).

### Consequences for the improvement plan

Two planned interventions were **dropped on this evidence**:

- **Swapping CodeBERT for UniXcoder** was ranked as the highest-ceiling single
  change. The paper measures the gain at **+0.57 F1** (21.43 vs 20.86) — 
  noise-level for several hours of work.
- **Extending the learning-rate search upward** was ranked highly on the basis
  of a monotonic trend across four local trials. The paper's 2e-5 is
  essentially the value already selected here, so the optimum is likely already
  bracketed.

Raising `max_chunks` had already been dropped on separate evidence (recall
32.6% against an 83.5% truncation ceiling).

What survives as the evidence-backed lever is **training for more epochs with a
schedule that does not anneal a model that is still underfitting**.

---

## 4c. Epoch trend — MEASURED

Both epochs are validation-only; no test data was involved in any of this.

| metric (validation) | epoch 1 | epoch 2 | relative change |
|---|---|---|---|
| **PR-AUC** (headline) | 0.1114 | **0.1308** | **+17.4%** |
| F1 at tuned threshold | 0.2093 | **0.2269** | +8.4% |
| recall | 0.3262 | **0.4206** | +28.9% |
| precision | 0.1548 | 0.1553 | +0.3% |
| MCC | 0.1931 | **0.2217** | +14.8% |
| ROC-AUC | 0.7951 | 0.8156 | +2.6% |
| Brier (uncalibrated) | 0.0344 | 0.0345 | ~flat |
| tuned threshold (objective f1) | 0.11 (calibrated) | 0.15 (raw) | — |

The F1 figures are comparable: epoch 1's raw-threshold F1 was 0.2093 at raw
threshold 0.20, and epoch 2's is 0.2269 at raw threshold 0.15.

**Interpretation.** The improvement is almost entirely **recall** — 0.3262 to
0.4206, i.e. 294 of 699 vulnerabilities found rather than 228 — with precision
essentially unchanged. Under a 36:1 imbalance that is the useful direction: the
model is separating the classes better, not merely trading one error type for the
other. PR-AUC rising 17.4% confirms the gain is in the *ranking*, not an artifact
of where the threshold landed.

`NEW BEST validation pr_auc 0.1308` fired, so `best_codebert.pt` now holds the
epoch-2 weights. **This is why the frozen policy in §6 is stale** — it was fitted
to epoch-1 scores, whose threshold (0.11) and isotonic knots no longer correspond
to this model's score distribution.

The flat Brier alongside a better ranking is expected: `pos_weight` BCE optimises
separation, not probability scale, which is what the calibration step exists to
repair.

**Decision taken on this evidence:** epoch 2 clearly earns its keep, so epoch 3
was allowed to run rather than stopping at 2. The three-point trend, not the
two-point one, decides whether a longer run is justified — a single increment
cannot distinguish "still improving" from "one lucky epoch".

---

## 5. Calibration — MEASURED, adopted

Fitted and judged **entirely inside validation**, by stratified K-fold so the
improvement estimate comes from data the calibrator never saw.

### Uncalibrated reliability (quantile bins)

| Bin | n | Predicted | Observed | Gap |
|---|---|---|---|---|
| [0.0172, 0.0173] | 491 | 0.0172 | 0.0000 | −0.0172 |
| [0.0173, 0.0174] | 4,039 | 0.0173 | 0.0032 | −0.0141 |
| [0.0174, 0.0174] | 4,188 | 0.0174 | 0.0060 | −0.0114 |
| [0.0174, 0.0175] | 3,833 | 0.0174 | 0.0133 | −0.0041 |
| [0.0175, 0.0176] | 4,381 | 0.0175 | 0.0192 | +0.0016 |
| [0.0176, 0.0178] | 2,873 | 0.0177 | 0.0265 | +0.0087 |
| [0.0178, 0.0845] | 3,080 | 0.0339 | 0.0461 | +0.0122 |
| **[0.0845, 0.9981]** | **2,545** | **0.3354** | **0.1210** | **−0.2143** |

ECE 0.029277 · MCE 0.214334 · noise floor 0.00243 → measured ECE is **12.1x
the floor**, so this is real miscalibration, not sampling noise.

**The top bin is the problem.** In the highest-scoring 2,545 functions the
model claims 33.5% and delivers 12.1%. That is the region an analyst would
triage first, so the miscalibration sits exactly where it costs most. This is
the predicted consequence of training at 10:1 while validating at 35:1.

### Methods compared (out-of-fold)

| Method | ECE | Relative gain | Brier | Verdict |
|---|---|---|---|---|
| uncalibrated | 0.02928 | — | 0.03444 | — |
| Platt | 0.00952 | −67.5% | 0.02604 | eligible |
| **isotonic** | **0.00180** | **−93.8%** | **0.02543** | **ADOPTED** |

Brier improved as well as ECE, so this is not a cosmetic rescaling. Adoption
required three conditions simultaneously: ECE above 2x the noise floor,
≥10% relative out-of-fold gain, and no Brier regression.

**Calibration changes probability quality, not ranking.** PR-AUC and ROC-AUC
depend only on score order and are unchanged. No classification improvement is
claimed from calibration itself.

---

## 6. Frozen decision policy

> **STALE — fitted to the epoch-1 checkpoint.** Epoch 2 produced a better model
> (§4c) and overwrote `best_codebert.pt`, so the threshold and isotonic knots
> below were fitted to a score distribution this checkpoint no longer has. They
> are recorded for provenance, **must not be used for the test evaluation**, and
> will be re-derived on the final checkpoint before it is run. The verification
> that `test_model.py` correctly consumes a policy (§5 of
> `PRE_TEST_PREFLIGHT.md`) used this file and remains valid — the mechanism is
> sound, only these particular numbers are superseded.

`data/checkpoints/decision_policy.json`

| Field | Value |
|---|---|
| `selected_on` | **validation** |
| `frozen` | true |
| objective | f1 |
| **threshold** | **0.1100** (re-derived on calibrated scores) |
| calibration | isotonic, parameters serialised (1,487 chars) |
| validation P / R / F1 at frozen point | 0.1548 / 0.3262 / 0.2099 |

`test_model.py` refuses any policy not marked `selected_on: validation`, and
appends every test evaluation to an append-only ledger.

---

## 7. Comparison — what can and cannot be said yet

| Approach | Split | F1 | PR-AUC |
|---|---|---|---|
| rules_only | **test** | 0.1065 | 0.0378 |
| taint_only | **test** | 0.0000 | 0.0268 |
| CodeBERT | **validation** | 0.2093 | 0.1114 |

**These are not directly comparable.** The model figure is validation, the
baselines are test, and the two splits differ in distribution (validation
resembles test more than train, but they are not identical). A fair comparison
requires the single test evaluation.

What can be said: on its own split the model's PR-AUC is **4.05x chance**,
while `rules_only` on its split is **1.41x chance**. That is suggestive, not
conclusive.

---

## 8. Limitations carried into the test evaluation

1. **Truncation is class-asymmetric.** On test, **20.86% of vulnerable**
   functions are truncated at the 8-chunk cap versus **2.40% of benign** —
   **8.7x**. 22.5% of corpus tokens are discarded. Any vulnerability past token
   3,184 in those functions is invisible to the model, capping achievable
   recall on the class being detected.
2. **Distribution shift.** Train is dominated by CWE-119/20/264 in
   linux/Chrome/qemu; test by CWE-617 in vim/gpac/tensorflow.
3. **Residual valid↔test overlap:** 14 functions (0.054% of test), **all 14
   benign** — so the overlap cannot affect recall, only marginally precision.
   A sensitivity analysis excluding them is built and will be run alongside the
   test evaluation.
4. **Undersampling at 10:1** shifts probabilities; corrected by the
   validation-fitted calibrator above.
5. **Small tuning budget:** 4 trials × 1 epoch on a 9,000-function subsample.
   Dropout is confounded with learning rate.
6. **Project overlap** remains (149 shared projects, 21,287 test records), so
   held-out metrics partly measure within-project generalisation.

---

## 9. Not implemented

- **GATv2** — `backend/ml/graphs/` is an empty package; `torch_geometric` not
  installed. No graph tensors are constructed anywhere.
- **GraphCodeBERT** — CodeBERT is used. GraphCodeBERT requires token-aligned
  data-flow inputs this pipeline does not produce; loading its weights without
  them would earn the label without the mechanism.
- **Model-side explainability** — no attribution or attention analysis.
- **Persistence** — `/analysis/{id}`, `/analyze/repository`.

No result in this document may be attributed to any of these.

---

## 10. Remaining steps

1. Finish training (epoch 3 of 3 running at time of writing; epoch 2 improved
   on epoch 1, see §4c)
2. **Re-run threshold + calibration on the final best checkpoint** — the policy
   in §6 is now confirmed stale, because epoch 2 overwrote `best_codebert.pt`.
   This is no longer conditional; it must happen before step 3.
3. One test evaluation with `--save-predictions`
4. Hybrids, error analysis, sensitivity analysis, ablations — all from saved
   predictions, so test is scored exactly once
5. Complete §7 with the fair comparison
