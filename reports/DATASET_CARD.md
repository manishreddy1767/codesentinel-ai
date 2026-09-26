# Dataset Card — PrimeVul (as used by CodeSentinel-AI)

**Last updated:** 19 Sep 2026
**Every number below was re-verified from repository artifacts on that date.**
Sources: `data/raw_checksums.json`, `artifacts/reports/leakage_raw.json`,
`artifacts/reports/leakage_clean.json`, `artifacts/reports/EXPERIMENT_LOG.md`.

---

## 1. Source and provenance

| Field | Value |
|---|---|
| Name | PrimeVul |
| Upstream | https://github.com/DLVulDet/PrimeVul |
| Obtained from | Google Drive folder `19iLaNDS0z99N8kB_jBRTmDLehwZBolMY` (the "original release" linked in the upstream README) |
| Retrieved | 18 Sep 2026, via `gdown` |
| Upstream licence | Repository displays an MIT licence badge |
| Files taken | The three unpaired splits only |

The `v0.1` folder was **not** used: it additionally ships a `file_contents/`
directory of thousands of `.txt` files this pipeline does not consume.

### Integrity

| File | SHA-256 | Bytes |
|---|---|---|
| `primevul_train.jsonl` | `9fea452f1b7c7ffafb28d6131789f722ad820c1032d3bcd90b7fc17da3d9b117` | 343,487,568 |
| `primevul_valid.jsonl` | `56b91474fb7d75b313013766e0f5d1d8150c98e70961cf2b26df93875e87fb27` | 52,185,805 |
| `primevul_test.jsonl` | `7f0db8408bdd614f7eb8b4df145d7bc7dbfda0243f7a29c7caa6f89b9a756245` | 54,720,055 |

Files are set **read-only** and re-verified after every pipeline stage.
Nothing in the codebase writes to `data/raw/`.

---

## 2. Schema

Auto-detected, not hardcoded (`config.CODE_FIELD_CANDIDATES` /
`LABEL_FIELD_CANDIDATES`).

| Field | Presence (test split) | Use |
|---|---|---|
| `func` | 100% | source text — the model input |
| `target` | 100% | label: 1 = vulnerable, 0 = benign |
| `project` | 100% | leakage audit (context dimension) |
| `commit_id` | 100% | leakage audit (identity) |
| `idx` | 100% | leakage audit (identity) |
| `hash` | 100% | upstream content hash; leakage audit (identity) |
| `cwe` | 11.8% | reporting only |
| `big_vul_idx` | 2.0% | leakage audit (identity), sparse |

Labels are function-level. No sub-function localisation labels exist, so the
model is trained and evaluated **per function**.

---

## 3. Splits as published

| Split | Records | Vulnerable | Benign | Neg:Pos |
|---|---|---|---|---|
| train | 184,427 | 5,574 (3.02%) | 178,853 | 32.1 : 1 |
| valid | 25,430 | 699 (2.75%) | 24,731 | 35.4 : 1 |
| test | 25,911 | 695 (2.68%) | 25,216 | 36.3 : 1 |

Zero records had missing/empty code or an invalid label.

Code length, test split (characters): min 18 · p50 486 · p90 2,693 ·
p99 13,261 · max 137,677.

---

## 4. Leakage audit

**Method.** Seven dimensions compared pairwise across all three splits:

| Dimension | Kind | Treated as |
|---|---|---|
| `exact_code` | content | leakage |
| `normalized_code` (whitespace-collapsed SHA-256) | content | leakage |
| `hash`, `idx`, `big_vul_idx`, `commit_id` | identity | leakage |
| `project` | context | **concern, not leakage** |

Tool: `python -m backend.ml.preprocessing.audit_splits`.

### Result on the raw data — leakage present

| Pair | exact | **normalized** | hash | commit_id | project |
|---|---|---|---|---|---|
| train↔valid | 0 | **1,310** | 0 | 12 | 163 |
| train↔test | 0 | **796** | 0 | 12 | 149 |
| valid↔test | 0 | **14** | 0 | 0 | 84 |

**Key finding.** `exact_code` and PrimeVul's own `hash` field are clean on every
pair, but `normalized_code` is not. PrimeVul deduplicated on *exact* content, so
**796 test records (3.07%) and 1,310 validation records are whitespace-variants
of training functions** and are invisible to exact-content hashing.

Within-split duplicates (raw, normalized_code): train 7,632 keys / 7,671
redundant records; valid 3; test 54.

### Remediation — train-only

```bash
python -m backend.ml.preprocessing.clean_split_duplicates \
    --in-dir data/processed --out-dir data/processed_clean \
    --dimensions exact_code normalized_code hash idx big_vul_idx commit_id
```

Removed **2,324 train records (1.3148%)**: normalized_code 2,106, exact_code
2,015, commit_id 371 (a record can match several). Validation and test were
copied through byte-for-byte — **no held-out record was ever deleted**.

Project overlap was deliberately *not* used as a removal criterion: it would
have required deleting 135,245 train records (73%).

### Re-audit after cleaning — verified

| Pair | exact | normalized | hash | idx | big_vul_idx | commit_id |
|---|---|---|---|---|---|---|
| train↔valid | 0 | 0 | 0 | 0 | 0 | 0 |
| train↔test | 0 | 0 | 0 | 0 | 0 | 0 |
| valid↔test | **7** | **14** | 0 | 0 | 0 | 0 |

### Residual issue: valid↔test overlap — NOT REMEDIATED

14 normalized (7 exact) functions are shared between validation and test —
**0.054% of the test split**.

This is **not** train-on-test leakage: no training data is involved, so the
model has not seen these functions. The effect is that threshold and
calibration, fitted on validation, are fitted on 14 functions that also appear
in test — a small optimistic coupling.

It was not remediated because doing so requires editing a held-out split, which
changes what the model is measured against and is a worse methodological error
than the 0.054% coupling it would fix.

**Sensitivity analysis is feasible and is recommended**: recompute test metrics
with those 14 records excluded and report both numbers. It requires only the
saved per-sample predictions plus a normalized-hash join against the validation
split — no model re-run, so it does not consume the one-shot test budget. **Not
yet performed.**

*(The 7 `exact_code` overlaps appear only after preprocessing, because
whitespace normalisation makes 7 previously-distinct functions byte-identical.)*

---

## 5. Cleaned splits actually used

| Split | Functions | Vulnerable | Neg:Pos |
|---|---|---|---|
| train (cleaned) | 174,432 | 5,550 (3.18%) | 30.43 : 1 |
| valid | 25,430 | 699 (2.75%) | 35.38 : 1 |
| test | 25,911 | 695 (2.68%) | 36.28 : 1 |

`pos_weight = 30.4292`, computed from the cleaned training split, below the
configured cap of 50 (so no capping applied).

### Optional training-only undersampling

For compute budget, a 10:1 variant is used for final training:

```
61,050 functions = all 5,550 positives + 55,500 sampled negatives (seed 42)
implied pos_weight = 10.0
```

**Validation and test retain the true class balance.** The consequence is that
predicted probabilities are shifted upward relative to the deployment base
rate — a *calibration* effect, not a ranking one (PR-AUC and ROC-AUC depend
only on score order). It is corrected downstream by validation-fitted threshold
selection and calibration.

---

## 6. Distribution shift — measured

| Split | Distinct projects | Distinct CWEs | Top projects | Top CWEs |
|---|---|---|---|---|
| train | 633 | 90 | linux 45,678 · Chrome 17,540 · qemu 5,595 | CWE-119 14,304 · CWE-20 8,684 · CWE-264 6,647 |
| valid | 218 | 23 | linux 7,120 · server 920 · hhvm 911 | CWE-617 544 · CWE-190 213 · CWE-787 125 |
| test | 222 | 21 | linux 4,428 · gpac 1,890 · vim 1,672 | CWE-617 909 · CWE-20 305 · CWE-476 164 |

Train is dominated by classic memory-safety CWEs; test by CWE-617 (reachable
assertion). Validation resembles test far more than train. **The splits are not
i.i.d. draws from one pool**, so the test split measures generalisation under
distribution shift. Test metrics should be read accordingly.

Project overlap remains after cleaning (149 shared projects, 21,287 test
records), so held-out metrics partly measure *within-project* generalisation.

---

## 7. Chunking and truncation bias

CodeBERT tokenizer, `max_length=512`, `overlap=128`, `max_chunks=8`. Eight
chunks at stride 382 cover the first **3,184 tokens**.

| Split | Functions | Chunks | Mean/fn | Truncated |
|---|---|---|---|---|
| train | 174,432 | 298,192 | 1.710 | 5,174 (2.97%) |
| valid | 25,430 | 42,623 | 1.676 | 626 (2.46%) |
| test | 25,911 | 44,440 | 1.715 | 751 (2.90%) |


### Measured on the FULL splits (not a sample)

| Split | Functions | Truncated | Truncated % | **Vulnerable truncated %** | Benign truncated % | **Ratio** |
|---|---|---|---|---|---|---|
| train | 174,432 | 5,174 | 2.97% | **19.24%** | 2.43% | **7.9x** |
| valid | 25,430 | 626 | 2.46% | **16.45%** | 2.07% | **8.0x** |
| test | 25,911 | 751 | 2.90% | **20.86%** | 2.40% | **8.7x** |

> **Correction.** An earlier revision of this card quoted 13.33% / 2.25% /
> 5.9x for train. Those came from a 3% random sample; the table above is a
> full-population count over every record in each split and supersedes them.
> The effect is **stronger** than the sample suggested, and strongest on the
> test split, where **one vulnerable function in five is truncated**.

### Token-level loss (5,176-function sample of cleaned train)



| Metric | Value |
|---|---|
| Functions exceeding 3,184 tokens | 2.57% |
| **Corpus tokens discarded** | **22.49%** (724,675 / 3,222,748) |
| Tokens dropped per truncated function | median 1,823 · p90 9,459 · max 137,576 |
| Median tokens, truncated vulnerable functions | 812 |
| Median tokens, truncated benign functions | 206 |

**A vulnerable function is 7.9-8.7x more likely to be truncated than a benign
one, depending on the split (full-population counts above).**
Any vulnerability located past token 3,184 in those functions is invisible to
the model, which places a hard ceiling on achievable recall concentrated on the
positive class. This is a live limitation, not a resolved one.

---

## 7b. Record ordering — the files are sorted by label

**Verified 19 Sep 2026** by scanning `data/processed_clean/*.jsonl` in file order.

| Split | Positives | Positive index range | File length |
|---|---|---|---|
| train | 5,550 | 0 .. 82,711 | 174,432 |
| valid | 699 | 0 .. 1,076 | 25,430 |
| test | 695 | 0 .. 1,204 | 25,911 |

**100% of validation and test positives lie in the first ~4.5% of their files.**
In train, 100% of positives lie in the first half.

### Why this matters

Any code that takes a *prefix* of these files gets a wildly unrepresentative
class balance. Measured on the validation split at n=4,000:

| Selection method | Positives | Rate |
|---|---|---|
| Stratified sample | 110 | **2.75%** (matches the true rate) |
| First 4,000 records | 699 | **17.47%** — 6.4x inflated |

This was a live defect: `_Subset` originally took the head of the file. It was
fixed to stratified sampling *before* the hyperparameter search ran, which is
the only reason the search is valid — PR-AUC is not comparable across
different base rates, so scoring trials against a 17.47%-positive subset while
reporting them as validation performance would have invalidated the entire
search and the selected configuration.

### Why the rest of the pipeline is unaffected

Verified rather than assumed:

| Path | Protection |
|---|---|
| Training order | `ResumableSampler(shuffle=True)` — order randomised per epoch |
| Validation / test evaluation | `shuffle=False` but **the whole split is consumed and predictions concatenated**, so batch order cannot change any metric |
| Batch composition effects | RoBERTa uses **LayerNorm** (per-sample), not BatchNorm — a batch of all-positives cannot shift another sample's activations |
| Subsets (`--limit-*`) | Stratified and seeded |
| Prediction unit | One prediction per function (`batch_size=function_count`) |

The ordering is therefore a hazard the implementation is immune to *by
construction*, not by luck — but it is recorded here because any new code that
slices these files must stratify.

---

## 8. Known limitations

1. Normalized-hash de-duplication catches copies, not semantic clones — the
   796 figure is a **lower bound**.
2. Residual valid↔test overlap of 14 functions (§4).
3. Distribution shift between train and test (§6).
4. Class-asymmetric truncation (§7).
5. Function-level labels only — no line-level ground truth, so localisation
   cannot be evaluated against labels.
6. C/C++ dominated; the pipeline supports 5 languages but only C/C++ is
   exercised at scale.
7. `cwe` present for only 11.8% of records, so per-CWE analysis covers a
   minority of the data.
8. **Files are sorted by label** (§7b). Any prefix-based subsetting produces a
   6.4x-inflated positive rate on validation/test. All current code paths
   stratify or consume the full split, but this constrains future changes.
