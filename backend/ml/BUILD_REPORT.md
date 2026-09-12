# CodeSentinel ML Module — Build & Verification Report

**Repo:** codesentinel-ai · **Branch:** main · **Module:** `backend/ml` · **Date:** 12 Sep 2026

Build, repair and verification report for the hierarchical CodeBERT vulnerability
classifier — what was found broken, what was decided, and what was actually run.

| Metric | Value |
|---|---|
| Lines of ML code | 3,590 |
| Files created | 19 |
| Defects fixed | 6 |
| Tests pass / skip | 58 / 1 |
| Datasets present | 0 |

---

## BLOCKING — needs your action

### The PrimeVul datasets are not on this machine

`data/` did not exist in the clone, and a full `C:\` scan found no `primevul_*.jsonl`
anywhere on the system. The files previously lived in `data/primevul_dataset/`, which
`.gitignore` excludes — so they were never carried across by the clone.

Per your instruction I did **not** download them. The entire pipeline was built against
`data/raw/` as specified and verified end-to-end using a synthetic fixture with
PrimeVul's schema and imbalance. Drop your three real files into `data/raw/` and every
command in section 07 runs unchanged.

---

## 01 — Starting state

The repository had a working FastAPI analysis backend (AST / CFG / DFG / CPG, taint
analysis, rule-based detection) and a partial ML preprocessing layer from the most recent
commit, `76bb58b`. The learning half did not exist.

| Area | Found | Assessment |
|---|---|---|
| `backend/ml/preprocessing` | 9 scripts — loading, stats, balancing, chunking, length analysis | Partly usable, several defects |
| `backend/ml/training` | Empty `__init__.py` only | No trainer, optimizer, loop or checkpointing |
| `backend/ml/evaluation` | Empty `__init__.py` only | No metrics or evaluation |
| `backend/ml/models` | Did not exist | No model, dataset or collate |
| `data/` | Did not exist | No raw, processed, chunked or checkpoint dirs |
| Environment | Python 3.14, no venv, no torch or transformers | Nothing ML-related could run |

---

## 02 — Defects found and fixed

Six real defects, found by reading the existing code and confirmed against the real
CodeBERT tokenizer. The first would have quietly degraded every multi-chunk training
sample.

### D1 — Chunking destroyed special tokens  `CRITICAL`

`chunk_dataset.py` called `encode(code, add_special_tokens=True)` and *then* sliced the
result into 512-token blocks. Only chunk 1 received `<s>` and only the last received
`</s>` — every middle chunk was a bare token run in a format CodeBERT never saw during
pre-training.

Rewritten: tokenize with `add_special_tokens=False`, body capped at `max_length - 2` =
510, and each chunk wrapped independently. Guarded by
`test_chunking_multi_chunk_has_special_tokens_on_every_chunk`.

### D2 — `transformers` missing from requirements  `BUILD`

Five scripts imported it; `backend/requirements.txt` never listed it. A clean install
could not run any ML script. Added `transformers`, `tokenizers`, `safetensors` and
`regex`.

### D3 — `.gitignore` would have swallowed the model package  `BUILD`

A bare `models/` pattern matches at any depth, so the new `backend/ml/models/` source
package was invisible to git — the model code would never have been committed. Anchored
to `/models/` and `/checkpoints/`.

### D4 — Dead `test_*.py` scripts inside the package  `HYGIENE`

`test_chunking.py` and `test_tokenizer.py` were ad-hoc demo scripts that pytest would
collect as tests. `test_chunking.py` also held the *correct* chunking algorithm the
pipeline never called — and returned **zero chunks** for empty input, which would crash
collation. Removed and superseded by a real suite.

### D5 — CWD-relative paths and inconsistent data roots  `CORRECTNESS`

`dataset_stats.py` and `inspect_local_primevul.py` used `Path("data/primevul_dataset")`,
breaking outside the repo root, while other scripts used a `PROJECT_ROOT` anchor. All
paths now come from one env-overridable `config.py`.

### D6 — No leakage or duplicate detection  `VALIDITY`

Nothing checked whether a function appeared in more than one split. A single shared
function between train and test silently inflates every reported metric.
`validate_dataset.py` now hashes whitespace-normalised code and reports cross-split
overlap plus label-conflicting duplicates.

---

## 03 — The pipeline

Eight ordered stages. Each is a standalone module with its own CLI, so any stage can be
re-run without repeating the ones before it. `data/raw/` is opened read-only at every
stage and is never written to.

| # | Stage | What it does | I/O |
|---|---|---|---|
| 01 | `inspect_dataset` | Reports fields, labels, counts, imbalance, empty values, malformed lines and length percentiles — no schema assumed | `data/raw/` → stdout |
| 02 | `validate_dataset` | Blocks on unusable data; reports within-split duplicates and cross-split leakage. `--strict` makes leakage fatal | `data/raw/` → exit code |
| 03 | `preprocess_primevul` | Normalises code and field names, drops empty/invalid records, de-duplicates the train split only | `data/raw/` → `data/processed/` |
| 04 | `balance_dataset` | Reports class distribution and the pos_weight the trainer will derive. Undersampling is opt-in only | `data/processed/` → report |
| 05 | `chunk_dataset` | Tokenizes and splits into valid CodeBERT windows with 128-token overlap, capped at 8 chunks per function | `data/processed/` → `data/chunked/` |
| 06 | `training.train` | Trains with per-epoch validation, F1 early stopping, threshold search and periodic checkpointing. Never opens the test split | `data/chunked/` → `data/checkpoints/` |
| 07 | `evaluation.test_model` | Loads the best checkpoint and its validation-selected threshold, evaluates the untouched test split once | `best_codebert.pt` → final metrics |
| 08 | `pytest test_ml_pipeline` | 59-test suite over synthetic fixtures. Needs no dataset, no GPU and no model download | synthetic → 58 pass, 1 skip |

---

## 04 — Design decisions

### Class imbalance — one strategy, not a stack

**Chosen: `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`**, computed by the trainer from
the training file actually in use and capped at 50.

| Strategy | Verdict |
|---|---|
| **pos_weight — SELECTED** | Uses every sample. Discards nothing, duplicates nothing. One mechanism, one interpretable number. Leaves validation and test untouched by construction. |
| Majority undersampling | Throws away most benign functions — exactly the hard negatives that drive precision. Kept as an opt-in compute-budget tool via `--undersample`, off by default. |
| Oversampling positives | Rejected. With roughly 6k unique positives the model memorises repeats and validation F1 becomes optimistic. |
| WeightedRandomSampler | Similar effect to pos_weight but stochastic, and it makes epoch order non-deterministic — which conflicts directly with the exact-resume sampler. |

> The one place two mechanisms interact is deliberate: if you opt into undersampling,
> pos_weight is recomputed from the resulting file so it still describes the distribution
> the model actually trains on. That is one coherent reweighting of the real prior, not
> two stacked heuristics.

**Enforced guarantees:** only the train split is ever resampled or de-duplicated;
positives are never duplicated; validation and test are never balanced or augmented. All
three are covered by tests that compare file bytes before and after.

### Model architecture

```
source function
   → tokenization                        chunk_dataset.py
   → chunks                    [N, L]    N = sum of chunk counts in batch
   → CodeBERT                  [N, L, H]
   → chunk embeddings          [N, H]    CLS token (or masked mean)
   → chunk aggregation                   scatter-mean by function_index
   → function embedding        [B, H]
   → dropout + Linear(H, 1)
   → vulnerability logit       [B]       logits.shape == targets.shape
```

A batch of `B` functions expands to `N` chunks. Rather than padding to a rectangular
`[B, max_chunks, L]` tensor and spending encoder compute on padding chunks, collation
flattens to `[N, L]` and emits a `function_index` vector; the model scatters embeddings
back per function with `index_add_`. `forward` returns raw logits — sigmoid is applied
only after the loss, never inside the model.

### Memory management

| Technique | Where |
|---|---|
| Mixed precision, auto-disabled on CPU | `Trainer` |
| Gradient checkpointing, `use_reentrant=False` | `HierarchicalCodeBERTClassifier` |
| Chunk micro-batching (default 16) | `_encode_chunks` |
| Gradient accumulation + configurable batch size | `Trainer` |
| Clipping after `scaler.unscale_` | `Trainer.train_epoch` |
| `zero_grad(set_to_none=True)` | `Trainer.train_epoch` |
| Cap of 8 chunks per function | `chunk_dataset.py` |
| Lazy offset-indexed dataset reads | `ChunkedFunctionDataset` |

`use_reentrant=False` is not cosmetic. On the reentrant autograd path, gradient
checkpointing produces **no encoder gradients at all** when the block's inputs are token
ids that do not require grad — training would appear to run while the encoder never
learned. `test_gradient_checkpointing_still_produces_gradients` guards it.

Loss is accumulated as a Python float, never as a tensor; retaining the tensor keeps its
autograd graph alive for the whole epoch, which is the classic "memory grows every batch"
bug. `empty_cache()` is not called per batch.

Separately, the dataset was switched to lazy offset-indexed reads: holding the full
184k-function training split as Python lists of ints costs 28 bytes per token and runs to
several GB before training even starts.

### Checkpointing and exact mid-epoch resume

A plain `DataLoader(shuffle=True)` *cannot* be resumed correctly: the shuffle comes from a
global RNG whose state at the start of the interrupted epoch is unrecoverable, so a naive
resume silently reshuffles and replays samples the model already saw.

`ResumableSampler` derives each epoch's permutation from a pure function of
`(seed, epoch)`. On resume the identical permutation is regenerated and the first
`batch_in_epoch × batch_size` indices are skipped — no replay, no loss. `maybe_resume()`
returns `True` only if state was genuinely restored, and prints an explicit "starting from
scratch" line otherwise.

Two checkpoint kinds, both written atomically (temp file + replace):

- `data/checkpoints/latest_checkpoint.pt` — epoch, batch-in-epoch, global step, model,
  optimizer, scheduler, GradScaler, best F1, best threshold, early-stopping counter,
  architecture config. ~1.49 GB.
- `data/checkpoints/best_codebert.pt` — best-F1 weights + the validation-selected
  threshold + the metrics that justified it. ~498 MB (weights only).

### Threshold selection

Every epoch reports metrics at **0.5** and at the F1-maximising threshold from a 91-point
grid over [0.05, 0.95] — searched on **validation only**. The selected threshold is stored
inside `best_codebert.pt`. `test_model` never re-tunes it; a `--tune-on-test` flag is
deliberately absent, because optimising a threshold against test predictions turns the
reported F1 into an optimistic upper bound rather than a held-out estimate.

---

## 05 — File inventory

### Created — 19 files

| Path | Role | Lines |
|---|---|---|
| `backend/ml/config.py` | Central config, every value env-overridable | 201 |
| `backend/ml/preprocessing/inspect_dataset.py` | Stage 1 — schema-agnostic raw inspection | 212 |
| `backend/ml/preprocessing/validate_dataset.py` | Stage 2 — validation and leakage check | 277 |
| `backend/ml/models/dataset.py` | Lazy chunked-function dataset | 144 |
| `backend/ml/models/collate.py` | Variable-chunk flattening collate | 76 |
| `backend/ml/models/codebert_classifier.py` | Hierarchical CodeBERT classifier | 224 |
| `backend/ml/training/metrics.py` | Metrics and threshold search | 172 |
| `backend/ml/training/checkpoint.py` | Atomic save/load, ResumableSampler | 175 |
| `backend/ml/training/trainer.py` | Training loop | 568 |
| `backend/ml/training/train.py` | Training CLI | 210 |
| `backend/ml/evaluation/evaluator.py` | Shared inference | 113 |
| `backend/ml/evaluation/test_model.py` | Final test CLI | 170 |
| `backend/ml/utils/seed.py` | Reproducible seeding | 40 |
| `backend/ml/utils/io.py` | JSONL helpers, field detection | 70 |
| `backend/ml/utils/gpu.py` | Memory reporting, device resolution | 52 |
| `backend/tests/test_ml_pipeline.py` | Verification suite | 1221 |
| `backend/ml/README.md` | Full module documentation | 452 |
| `data/README.md` | Where to place the datasets | 44 |
| `backend/ml/models/__init__.py`, `backend/ml/utils/__init__.py` | Package markers | 0 |

### Modified — 8 files

| Path | Change |
|---|---|
| `backend/ml/preprocessing/chunk_dataset.py` | Rewritten — correct per-chunk special tokens, overlap, chunk cap, early break, invariant assertions |
| `backend/ml/preprocessing/preprocess_primevul.py` | Reads `data/raw/`, auto-detects fields, train-only de-duplication, drop accounting |
| `backend/ml/preprocessing/balance_dataset.py` | Reports first; undersampling now opt-in and train-only |
| `backend/ml/preprocessing/analyze_code_lengths.py` | Config-driven paths, argparse |
| `backend/ml/preprocessing/analyze_token_lengths.py` | Config-driven paths, now projects chunk counts for the current settings |
| `backend/requirements.txt` | Added transformers, tokenizers, safetensors, regex |
| `.gitignore` | Anchored `/models/`; `data/*` with `!data/README.md` |
| `README.md` | Layout, ML pointer, corrected test commands |

### Removed — 5 files

| Path | Reason |
|---|---|
| `inspect_local_primevul.py` | Superseded by `inspect_dataset.py` |
| `dataset_stats.py` | Superseded by `inspect_dataset.py` |
| `test_chunking.py` | Dead script pytest would collect; see D4 |
| `test_tokenizer.py` | Dead script pytest would collect |
| `loader.py` | Dead — duplicated `utils/io.py`, no importers |

---

## 06 — Verification actually performed

Beyond the unit suite, the real pipeline was run end-to-end with the actual
`microsoft/codebert-base` weights on a synthetic 640-function dataset: inspect → validate
→ preprocess → distribution → chunk → 2 epochs of genuine training → resume → final test
evaluation.

### Chunking, checked against the real tokenizer

```
bos 0  eos 2
chunk count distribution: {1: 175, 5: 77, 8: 148}
chunk len min/max: 119 512
VIOLATIONS: NONE

middle chunk decodes cleanly -> '<s>, 15); process(a3_15);\n    int a3_16 = lookup(...'
```

Every chunk carries BOS and EOS, none exceeds 512, none is empty, and no function
produces zero chunks. The middle chunk starting with `<s>` is the direct proof that D1 is
fixed.

### High-loss diagnostics behaved as specified

```
[high-loss] epoch 1 batch 19 (1 sample(s) over 1.5)
  label=1  p(vuln)=0.1030  logit=-2.1647  loss=9.0933  chunks=8
```

Exactly the case in the specification: true label 1 with a predicted vulnerable
probability of 0.10 — a confidently wrong prediction that legitimately carries high loss.
Training continued. Only non-finite loss aborts the run, which is covered by
`test_trainer_aborts_on_non_finite_loss`.

### Best checkpoint contents

```
best_f1        : 0.3333
best_threshold : 0.05          <- selected on validation only
extra          : {'pos_weight': 4.0}   <- derived from 32 neg / 8 pos
arch config    : {'model_name': 'microsoft/codebert-base', 'max_length': 512,
                  'chunk_overlap': 128, 'max_chunks': 8,
                  'chunk_pooling': 'cls', 'function_pooling': 'mean', 'seed': 42}
optimizer_state: None           <- weights only: 498 MB vs 1.49 GB for latest
```

### Resume restored genuine state

```
RESUMED from checkpoint:
  epoch=2  batch_in_epoch=0  global_step=40  best_f1=0.3333
  best_threshold=0.0500  epochs_without_improvement=1
  Skipping the first 0 batches of epoch 3 using the (seed, epoch) permutation.
```

### Why accuracy is not the headline metric

The final test run produced the clearest possible illustration of the point. These are
**plumbing evidence, not model quality** — two epochs on 40 random synthetic samples. They
say the pipeline works end-to-end; they say nothing about detection performance.

**Test @ threshold 0.5** — accuracy 0.85 · precision 0.00 · recall 0.00 · F1 0.00

|        | Pred 0 | Pred 1 |
|--------|-------:|-------:|
| True 0 |    102 |      0 |
| True 1 |     18 |      0 |

85% accurate while detecting nothing.

**Test @ validation threshold 0.05** — accuracy 0.15 · precision 0.15 · recall 1.00 · F1 0.26

|        | Pred 0 | Pred 1 |
|--------|-------:|-------:|
| True 0 |      0 |    102 |
| True 1 |      0 |     18 |

15% accurate, but F1 is six times higher.

The metrics printer emits an explicit warning whenever the model collapses to a single
class. Both warnings fired here, as they should have.

### The 27-point checklist

| # | Item | Status |
|---|---|---|
| 1 | Raw dataset loading | PASS |
| 2 | Dataset inspection | PASS |
| 3 | Dataset validation | PASS |
| 4 | Labels valid | PASS |
| 5 | Class distribution correct | PASS |
| 6 | Balancing strategy correct | PASS |
| 7 | Valid / test byte-identical | PASS |
| 8 | Chunking works | PASS |
| 9 | Long functions chunked | PASS |
| 10 | Single-chunk functions | PASS |
| 11 | Multi-chunk functions | PASS |
| 12 | Batch size 1 | PASS |
| 13 | Larger batch sizes | PASS |
| 14 | Output shape == target shape | PASS |
| 15 | Forward pass | PASS |
| 16 | Loss calculation | PASS |
| 17 | Backward pass | PASS |
| 18 | GPU training | N/A — no CUDA on this machine |
| 19 | CPU fallback | PASS |
| 20 | GPU memory stability | N/A — test written, skipped without CUDA |
| 21 | Checkpoint saving | PASS |
| 22 | Checkpoint loading | PASS |
| 23 | Resume training | PASS |
| 24 | Validation | PASS |
| 25 | Metrics | PASS |
| 26 | Best model saving | PASS |
| 27 | Final test evaluation | PASS |

---

## 07 — What to run next

Place `primevul_train.jsonl`, `primevul_valid.jsonl` and `primevul_test.jsonl` in
`data/raw/`, then run the stages in order. A `.venv` already exists in the repo with CPU
torch installed.

```bash
.venv/Scripts/python -m backend.ml.preprocessing.inspect_dataset
.venv/Scripts/python -m backend.ml.preprocessing.validate_dataset
.venv/Scripts/python -m backend.ml.preprocessing.preprocess_primevul
.venv/Scripts/python -m backend.ml.preprocessing.balance_dataset
.venv/Scripts/python -m backend.ml.preprocessing.chunk_dataset
.venv/Scripts/python -m backend.ml.training.train
.venv/Scripts/python -m backend.ml.evaluation.test_model
```

Before real training, install the CUDA build:

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Read `inspect_dataset`'s output before going further — it prints the real field names, the
actual class ratio and the pos_weight the trainer will use. If `validate_dataset` reports
cross-split leakage, resolve that before trusting any test number.

---

## 08 — Limitations and open items

**No datasets, so no real results** `BLOCKING`
Everything was verified on synthetic data. No claim is made about detection accuracy —
that requires training on real PrimeVul, which has not happened.

**Torch here is CPU-only** `SETUP`
`2.14.0+cpu` in a new `.venv`. Enough for the test suite and smoke runs; a real training
run needs the CUDA wheel.

**Functions over 8 chunks are truncated** `DESIGN`
Roughly 3.8k tokens. Raise `CODESENTINEL_MAX_CHUNKS` if your token-length profile
justifies the extra memory.

**Mean pooling dilutes a single vulnerable chunk** `DESIGN`
Inside a long benign function, one bad chunk averages away. Attention-based aggregation is
the natural next experiment; `FUNCTION_POOLING=max` is a cheap first comparison and is
already implemented.

**Only exact duplicates are detected** `VALIDITY`
PrimeVul ships paired vulnerable/fixed functions that are near-identical.
Whitespace-normalised hashing catches copies, not semantic clones, so some near-duplicate
leakage may survive.

**GPU runs are not bit-for-bit reproducible** `KNOWN`
Non-deterministic cuDNN kernels and atomics in reductions make runs statistically, not
exactly, reproducible. Export `CUBLAS_WORKSPACE_CONFIG=:4096:8` for stricter behaviour.

**Four service tests fail to import** `PRE-EXISTING`
`No module named 'app'` in `backend/tests/` — they need `cd backend` first. Confirmed by
stashing that this predates this work; left untouched as out of scope.

---

*CodeSentinel · `backend/ml` · microsoft/codebert-base · 58 passed, 1 skipped ·
Nothing committed — all changes are in the working tree.*
