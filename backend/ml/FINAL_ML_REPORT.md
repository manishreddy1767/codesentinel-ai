# CodeSentinel ML — Project Report

**Date:** 18 Sep 2026
**Machine:** Windows 11 · RTX 3050 Laptop 4 GB · Python 3.14.3 · torch 2.14.0+cu130
**Dataset:** PrimeVul, official release (SHA-256 in `data/raw_checksums.json`)
**Status legend:** VERIFIED · PARTIAL · RUNNING · NOT RUN · FAILED · NOT IMPLEMENTED

> Supersedes `VERIFICATION_STAGE1/2/3.md`, which describe rounds where the
> dataset was unavailable and **every number was synthetic**. This is the first
> report containing real PrimeVul results.
>
> Command-by-command detail: [`artifacts/reports/EXPERIMENT_LOG.md`](../../artifacts/reports/EXPERIMENT_LOG.md)

---

## 1. Executive summary

The project spent three verification rounds blocked on two things. Both are now
resolved, and the pipeline has run end-to-end on real data through Stage 13.

| Blocker | Rounds 1–3 | Now |
|---|---|---|
| PyTorch could not use the GPU | FAIL | **RESOLVED** — `torch 2.14.0+cu130`, proven by a real CUDA matmul |
| PrimeVul dataset absent | FAIL ×3 | **RESOLVED** — downloaded from the official source, checksummed, read-only |

**12 of 19 stages are VERIFIED on real data.** Stage 13 is running. Stages 14–17
(final training, threshold, calibration, test) have not run.

**No model performance claim is made yet.** The test split has never been
evaluated. The honest headline number arrives at Stage 17.

---

## 2. Stage status

| # | Stage | Status |
|---|---|---|
| 1 | Environment | **VERIFIED** |
| 2 | Dataset acquisition | **VERIFIED** |
| 3 | Data inspection | **VERIFIED** |
| 4 | Leakage audit | **VERIFIED** — real leakage found and removed |
| 5 | Preprocessing | **VERIFIED** |
| 6 | Class balancing | **VERIFIED** |
| 7 | Tokenization + chunking | **VERIFIED** |
| 8 | Dataset + collate | **VERIFIED** |
| 9 | Model verification | **VERIFIED** |
| 10 | Memory probe | **VERIFIED** |
| 11 | Overfit test | **VERIFIED** |
| 12 | Training smoke test | **VERIFIED** |
| 13 | Hyperparameter tuning | **RUNNING** — 4 trials done, 1 in flight |
| 14 | Final training | NOT RUN |
| 15 | Threshold optimisation | NOT RUN (tool built, tested) |
| 16 | Calibration | NOT RUN (tool built, tested) |
| 17 | Final test evaluation | NOT RUN — **test untouched** |
| 18 | Baselines | **PARTIAL** — non-ML VERIFIED; GATv2 **NOT IMPLEMENTED** |
| 19 | This report | in progress |

---

## 3. The dataset, as it actually is

| Split | Records | Vulnerable | Ratio | Invalid | Empty code |
|---|---|---|---|---|---|
| train | 184,427 | 5,574 (3.02%) | 32.1:1 | 0 | 0 |
| valid | 25,430 | 699 (2.75%) | 35.4:1 | 0 | 0 |
| test | 25,911 | 695 (2.68%) | 36.3:1 | 0 | 0 |

After cleaning: **174,432 train** (5,550 positive, 3.18%), `pos_weight` **30.4292**
measured from the cleaned split.

Chunking: 298,192 train chunks, mean **1.71** per function, 74% single-chunk.

---

## 4. Findings that change how the numbers should be read

### 4.1 PrimeVul ships with leakage its own de-duplication missed

`exact_code` and PrimeVul's own `hash` field are clean across every split pair.
`normalized_code` is not:

| Pair | exact | **normalized** | hash | idx | commit_id | project |
|---|---|---|---|---|---|---|
| train↔valid | 0 | **1,310** | 0 | 0 | 12 | 163 (concern) |
| train↔test | 0 | **796** | 0 | 0 | 12 | 149 (concern) |
| valid↔test | 0 | **14** | 0 | 0 | 0 | 84 (concern) |

**796 test functions (3.07%) and 1,310 validation functions are
whitespace-variants of training functions.** PrimeVul deduplicated on exact
content only. The repository's original checker also hashed exact content and
would have reported this as CLEAN.

Remediation removed 2,324 train records (**1.3%**). After cleaning,
train↔valid and train↔test are clean on all seven dimensions.

**Residual, accepted:** valid↔test share 14 functions (0.054% of test). Not
train-on-test leakage. Removing them would mean editing a held-out split, which
is a worse error than the 0.05% coupling it fixes.

**Project overlap is reported as a concern, never as leakage.** Treating it as
leakage would require deleting 135,245 train records (73%).

### 4.2 The 8-chunk cap is biased against the positive class

Eight chunks at stride 382 cover the first 3,184 tokens.

| Metric | Value |
|---|---|
| Functions truncated | 2.57% |
| **Corpus tokens discarded** | **22.5%** |
| **Vulnerable truncated** | **13.33%** (median 812 tokens) |
| Benign truncated | 2.25% (median 206 tokens) |

Vulnerable functions are ~4× longer and truncate **5.9× more often**. Any
vulnerability past token 3,184 is invisible to the model — a hard ceiling on
achievable recall, concentrated on the class that matters.

### 4.3 ROC-AUC reads 0.83 for a model that flags nothing

Smoke run, epoch 1 on 6,000 functions:

| Threshold | Precision | Recall | F1 | Predicted positive |
|---|---|---|---|---|
| 0.5 | 0.0000 | 0.0000 | 0.0000 | **0** |
| 0.05 (tuned on valid) | 0.2353 | 0.0488 | 0.0808 | 17 |

Threshold-free: **PR-AUC 0.1399** (chance = 0.0273), **ROC-AUC 0.8277**.

ROC-AUC says "0.83, looks good" about a model that predicts zero positives.
This is the precise failure mode PR-AUC was made the headline metric to avoid,
now observed on real data rather than argued in the abstract.

### 4.4 The existing rule engine is barely better than chance

Full test split, 25,911 functions, 0 analyzer errors:

| Baseline | Precision | Recall | F1 | PR-AUC | Flagged |
|---|---|---|---|---|---|
| rules_only | 0.1213 | 0.0950 | **0.1065** | 0.0378 | 544 |
| taint_only | 0.0000 | 0.0000 | 0.0000 | 0.0268 | 4 |
| rules_taint | 0.1211 | 0.0950 | 0.1065 | 0.0377 | 545 |

Chance PR-AUC = 0.0268. `rules_only` is **1.41× chance**. `taint_only` flags 4
functions out of 25,911 and catches **zero** vulnerabilities — no ranking
signal at all, and `rules_taint` is indistinguishable from `rules_only`,
confirming taint contributes nothing here.

**F1 0.1065 is the bar the learned model must clear.**

### 4.5 Train and test are distributionally different

| Split | Top CWEs | Top projects |
|---|---|---|
| train | CWE-119, CWE-20, CWE-264 | linux, Chrome, qemu |
| valid | CWE-617, CWE-190, CWE-787 | linux, server, hhvm |
| test | CWE-617, CWE-20, CWE-476 | linux, gpac, vim |

Valid resembles test; neither resembles train. PrimeVul's splits are not i.i.d.
draws from one pool — test measures generalisation under distribution shift.
Expect test metrics well below training-fit intuition.

---

## 5. Engineering: 12 defects fixed

Four would have produced quietly wrong results rather than visible errors.

| # | Defect | Consequence |
|---|---|---|
| 1 | Checkpoints recorded module config, not the model's architecture | A non-default pooling variant was saved mislabelled and silently reloaded as the baseline — any architecture comparison would have compared a model to itself |
| 2 | `fit()` never restored best weights | After early stopping the live model was the *last*, worse epoch |
| 3 | Field detection gated label lookup behind code lookup | One malformed record could make an entire split read as invalid labels |
| 4 | `--limit-train` took the head of the file | Could yield a single-class subset → `pos_weight` 1.0, overfit test vacuous |
| 5 | Early stopping had no `min_delta` | Float noise reset patience; early stopping never fired |
| 6 | Grad-accum boundary keyed to loader offset | Resumed epochs took a different number of optimizer steps |
| 7 | RNG state not checkpointed | Resume was close but not exact |
| 8 | ROC-AUC recomputed at all 91 threshold grid points | Threshold-independent work done 91× |
| 9 | Memory docstring pointed at the wrong knob | Would send an OOM debugger to the one parameter that cannot help |
| 10 | Pinned-memory allocator crash across tuning trials | Killed a 2-hour sweep at trial 3 |
| 11 | Non-strict checkpoint loading, dropout ignored | Silent architecture drift on reload |
| 12 | Two reporting defects (`best_epoch = -1`, "F1" label) | Misleading run summaries |

Also fixed: repo-wide test collection (missing `tree_sitter`, `fastapi`,
`pydantic_settings`, plus a root `conftest.py` reconciling two import roots).

**Test suite: 134 passing**, up from 58 at project start.

---

## 6. Code delivered

**14 new modules (~4,400 lines)** plus 13 files modified (1,042 insertions):

| Area | Modules |
|---|---|
| Leakage | `audit_splits.py` (7 dimensions), `clean_split_duplicates.py`, `undersample_chunked.py` |
| Evaluation | `calibration.py`, `select_threshold.py`, `baselines.py`, `compare_baselines.py` |
| Training | `overfit_test.py`, `memory_probe.py` |
| Tuning | `tuning/tune.py`, `tuning/space.py` |
| Infrastructure | `utils/experiment.py`, `test_ml_stages.py` (801 lines), `conftest.py` |

Methodological guarantees built into the tooling:

- Tuning objective is **PR-AUC (threshold-free)**, so the threshold cannot be
  co-optimised and inflate the reported score.
- Threshold and calibration are frozen to `decision_policy.json` **before** test
  is read; `test_model.py` refuses a policy not marked `selected_on: validation`.
- An append-only **test-evaluation ledger** makes repeat evaluations visible.
- Hybrids are computed from *saved* predictions, so test is scored exactly once.

---

## 7. Measured performance facts

| Quantity | Value | Note |
|---|---|---|
| Best config | **batch 8 + gradient checkpointing** | 10.5 fn/s, 2.5 GB peak |
| Worst config | batch 4, no checkpointing | **2.96 fn/s** — spills past 4 GB |
| batch 1 + checkpointing | 5.73 fn/s | the "safe" choice is 1.8× slower |
| Full epoch (174k) | ~5.2 h | |
| Undersampled epoch (61k) | **~1.8 h** | 10:1, all 5,550 positives kept |

Gradient checkpointing is what *enables* the large batch; it is not a
throughput cost here.

---

## 8. Stage 13 — tuning results so far

Objective: validation PR-AUC. `batch_size`/`grad_accum` pinned to the Stage 10
measured optimum so trials spend budget on learning rate.

| Round | Trial | Learning rate | Dropout | PR-AUC |
|---|---|---|---|---|
| 1 | 2 | 1.84e-06 | 0.162 | 0.1110 |
| 1 | 1 | 4.33e-06 | 0.480 | 0.1259 |
| 2 | 2 | 1.285e-05 | 0.162 | 0.1275 |
| 2 | **1** | **1.827e-05** | **0.480** | **0.1387** |

Round 1 crashed at trial 3 (defect #10) before reaching the 1e-5–5e-5 region
where CodeBERT fine-tuning normally lives. After fixing the allocator crash, a
focused round covered it. **PR-AUC increases monotonically with learning rate
across all four trials**, and the best so far (1.83e-5) sits exactly in the
standard range.

---

## 9. What remains

1. **Stage 13** finish (trial 5 in flight)
2. **Stage 14** final training — 61,050 functions × 3 epochs, ~6 h, checkpointed per epoch
3. **Stages 15–16** freeze threshold + calibration on validation
4. **Stage 17** the single test evaluation
5. **Stage 18** ML/hybrid comparison against the rules baseline
6. **Stage 19** complete this report with final numbers

---

## 10. Honest limitations

1. **No test performance number exists yet.** Everything above is training,
   validation or baseline data.
2. **The 22.5% token loss is unaddressed** and biased toward positives. Raising
   `MAX_CHUNKS` is the obvious lever; 4 GB of VRAM is the obstacle.
3. **Undersampling to 10:1 is a compute-budget decision**, declared as such. It
   shifts predicted probabilities (a calibration error, not a ranking one —
   PR-AUC and ROC-AUC depend only on score ordering). Stages 15–16 correct it on
   validation, which retains the true base rate.
4. **Tuning is 5 trials, 1 epoch each, on a 9,000-function subsample.** That is
   a smoke-scale search, not a thorough one; it is what the hardware allows.
5. **CodeBERT + GATv2 is NOT IMPLEMENTED.** `backend/ml/graphs/` and
   `backend/ml/embeddings/` contain only empty `__init__.py` files. Reported as
   unavailable rather than estimated.
6. **Project-level overlap is substantial**, so held-out metrics partly measure
   within-project generalisation and should be described that way.
7. **Nothing is committed.** 30 changed/new files; HEAD remains `c919c38`.
