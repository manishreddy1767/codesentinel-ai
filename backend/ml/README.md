# CodeSentinel ML Module

Hierarchical CodeBERT classifier that predicts whether a source-code function
is vulnerable.

```
0 = benign / non-vulnerable
1 = vulnerable
```

Backbone: [`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base).

---

## 1. Pipeline overview

```
data/raw/                      (read-only, never modified)
    |
    v  inspect_dataset      -> what is actually in the files
    v  validate_dataset     -> labels, quality, CROSS-SPLIT LEAKAGE
    v  preprocess_primevul  -> data/processed/
    v  balance_dataset      -> class distribution + pos_weight (+ optional undersample)
    v  chunk_dataset        -> data/chunked/
    |
    v  training.train       -> data/checkpoints/{latest_checkpoint,best_codebert}.pt
    |                          (per-epoch validation, F1 early stopping,
    |                           validation-only threshold selection)
    v  evaluation.test_model -> final held-out test metrics
```

Directory layout created by the pipeline:

```
data/
├── raw/          <- you provide this; nothing here is ever written to
├── processed/    <- normalised JSONL
├── chunked/      <- tokenized, chunked JSONL
└── checkpoints/  <- latest_checkpoint.pt, best_codebert.pt
```

---

## 2. Dataset

Place the PrimeVul splits here:

```
data/raw/primevul_train.jsonl
data/raw/primevul_valid.jsonl
data/raw/primevul_test.jsonl
```

**`data/raw/` is opened read-only by every stage.** No script in this module
writes to it.

### Schema

Field names are **auto-detected**, not assumed. The code field is the first of
`func`, `code`, `function`, `source`, `func_before` present in a record; the
label field is the first of `target`, `label`, `vul`, `is_vulnerable`,
`vulnerable`. `inspect_dataset` prints which ones it found. Metadata
(`project`, `commit_id`, `cwe`, `cve`, `idx`, `big_vul_idx`, `hash`) is carried
through preprocessing when present.

Run `inspect_dataset` first — it reports the real field list, label values,
record counts, class distribution, empty/missing values, malformed JSON lines
and code-length percentiles for your actual files.

---

## 3. Class imbalance strategy

**Chosen strategy: `BCEWithLogitsLoss(pos_weight = n_negative / n_positive)`.**

`pos_weight` is computed by the trainer from the training split it is actually
given, and capped at `CODESENTINEL_MAX_POS_WEIGHT` (default 50) so a
pathological split cannot produce an explosive weight.

### Why this one

| Strategy | Verdict |
|---|---|
| **`pos_weight` (chosen)** | Uses every sample; discards nothing; duplicates nothing; a single mechanism with one interpretable number; leaves validation/test untouched by construction. |
| Undersampling the majority | Throws away most benign functions — exactly the hard negatives that drive precision. Kept as an **opt-in compute-budget tool**, not the default. |
| Oversampling / duplicating positives | Rejected. With ~6k unique positives the model memorises repeats and validation F1 becomes optimistic. |
| `WeightedRandomSampler` | Equivalent effect to `pos_weight` but stochastic, and it makes epochs non-deterministic, which conflicts with the exact-resume sampler. |

Only **one** strategy is active at a time. The single place the two interact is
deliberate and documented: if you opt into undersampling, `pos_weight` is
recomputed from the resulting file, so it still describes the distribution the
model actually trains on. That is one coherent reweighting of the real prior,
not two stacked heuristics.

### Hard guarantees

- Only the **train** split is ever resampled.
- Validation and test are **never** balanced, deduplicated, or augmented — this
  is enforced by `preprocess_primevul` (dedupe is train-only) and covered by
  tests.
- Positives are never duplicated.

---

## 4. Chunking strategy

CodeBERT accepts 512 positions; many PrimeVul functions are longer.

- Tokenize with `add_special_tokens=False`.
- Chunk body = `max_length - 2` = **510** tokens, leaving exact room for
  `<s>` and `</s>`.
- **Every chunk is independently wrapped** as `[<s>] + body + [</s>]`.
- Consecutive chunks overlap by `CHUNK_OVERLAP` (default 128) tokens, so a
  pattern straddling a boundary is still wholly inside one chunk.
- Every function yields **at least one** chunk; an empty function yields the
  degenerate `[<s>, </s>]` rather than an empty list.
- **No chunk is ever empty**, and no chunk can exceed `max_length`.
- Functions needing more than `MAX_CHUNKS_PER_FUNCTION` (default 8) chunks are
  truncated to that many, and the record is flagged `"truncated": true`.

> **Bug fixed here.** The previous implementation called
> `tokenizer.encode(code, add_special_tokens=True)` and *then* sliced the
> result into 512-token blocks. Only the first chunk received `<s>` and only
> the last received `</s>`; every middle chunk was a bare token run that
> CodeBERT was never trained to see. `build_chunks` now attaches special tokens
> per chunk, and `test_chunking_multi_chunk_has_special_tokens_on_every_chunk`
> guards against a regression.

---

## 5. Model architecture

```
source function
    -> tokenization                       (chunk_dataset.py)
    -> chunks                    [N, L]
    -> CodeBERT                  [N, L, H]
    -> chunk embeddings          [N, H]    (CLS token, or masked mean)
    -> chunk aggregation                   (scatter-mean by function_index)
    -> function embedding        [B, H]
    -> dropout + Linear(H, 1)
    -> vulnerability logit       [B]
```

A batch of `B` functions expands to `N` chunks, where `N` is the *sum* of the
functions' chunk counts. Rather than padding to a rectangular
`[B, max_chunks, L]` tensor and wasting encoder compute on padding chunks, the
collate function flattens chunks to `[N, L]` and emits a `function_index`
vector of length `N`. The model scatters chunk embeddings back per function
with `index_add_`.

**The model returns exactly one raw logit per function**, shape `[B]`, so
`logits.shape == targets.shape` holds and `BCEWithLogitsLoss` applies directly.
Sigmoid is never applied inside `forward` — only in `predict_proba` and in
evaluation, after the loss.

Configurable: `CHUNK_POOLING` (`cls` | `mean`), `FUNCTION_POOLING`
(`mean` | `max`). Defaults are `cls` + `mean`, the stable choice.

---

## 6. Memory management

| Technique | Where |
|---|---|
| Mixed precision (`torch.amp`, auto-disabled on CPU) | `Trainer` |
| Gradient checkpointing (`use_reentrant=False`) | `HierarchicalCodeBERTClassifier` |
| Chunk micro-batching (`CHUNK_MICRO_BATCH`, default 16) | `_encode_chunks` |
| Configurable batch size + gradient accumulation | `Trainer` |
| Gradient clipping after `scaler.unscale_` | `Trainer.train_epoch` |
| `optimizer.zero_grad(set_to_none=True)` | `Trainer.train_epoch` |
| Cap of 8 chunks per function | `chunk_dataset.py` |

`use_reentrant=False` is not cosmetic: with the reentrant autograd path,
gradient checkpointing silently produces **no encoder gradients** when the
block's inputs (token ids) don't require grad.
`test_gradient_checkpointing_still_produces_gradients` guards this.

Loss is accumulated as a Python float, never as a tensor — retaining the tensor
would keep its autograd graph alive for the whole epoch, which is the classic
"memory grows every batch" bug.

GPU memory is logged every `GPU_LOG_EVERY` batches:

```
GPU memory | Allocated:    4821 MiB | Reserved:    6144 MiB | Peak:    5903 MiB
```

`torch.cuda.empty_cache()` is **not** called per batch — that only returns
memory to the driver and slows training. Peak stats are reset per epoch.

---

## 7. Loss and high-loss diagnostics

```python
logits = model(...)                 # raw, shape [B]
loss   = BCEWithLogitsLoss(pos_weight=w)(logits, targets)   # targets float32 [B]
probs  = torch.sigmoid(logits)      # for metrics only, after the loss
```

The criterion uses `reduction="none"` so per-sample losses are available; the
mean is taken explicitly.

When a sample's loss exceeds `HIGH_LOSS_THRESHOLD` (default 2.0) the trainer
prints epoch, batch, label, predicted probability, logit, loss and chunk count:

```
  [high-loss] epoch 1 batch 120 (2 sample(s) over 2.0)
    label=1  p(vuln)=0.1003  logit=-2.1950  loss=5.4103  chunks=3
```

This is **informational only**. A confidently wrong prediction *should* have a
high loss — `label=1, p=0.10` and `label=0, p=0.95` are both legitimately
high-loss and not evidence of a broken run. Training aborts **only** on
non-finite (NaN/Inf) loss or a genuine runtime failure.

---

## 8. Training

AdamW (no weight decay on biases/LayerNorm), linear schedule with warmup,
gradient clipping, mixed precision, reproducible seeding.

Everything is configurable by CLI flag or `CODESENTINEL_*` environment
variable — see `backend/ml/config.py`.

| Setting | Default | Env var |
|---|---|---|
| epochs | 5 | `CODESENTINEL_EPOCHS` |
| batch size (functions) | 8 | `CODESENTINEL_BATCH_SIZE` |
| chunk micro-batch | 16 | `CODESENTINEL_CHUNK_MICRO_BATCH` |
| learning rate | 2e-5 | `CODESENTINEL_LR` |
| weight decay | 0.01 | `CODESENTINEL_WEIGHT_DECAY` |
| warmup ratio | 0.06 | `CODESENTINEL_WARMUP_RATIO` |
| grad clip | 1.0 | `CODESENTINEL_MAX_GRAD_NORM` |
| checkpoint every | 500 steps | `CODESENTINEL_CHECKPOINT_EVERY` |
| early-stopping patience | 2 epochs | `CODESENTINEL_PATIENCE` |
| seed | 42 | `CODESENTINEL_SEED` |

---

## 9. Validation

Runs after **every** epoch, under `model.eval()` and `torch.no_grad()`, over
the original validation split — no balancing, no augmentation, no gradient
updates.

Reported: accuracy, precision, recall, F1, ROC-AUC, and the full confusion
matrix (TN, FP, FN, TP).

**Accuracy is deliberately de-emphasised.** On a ~97% benign split, a model
that always answers "benign" scores 97% accuracy and is worthless. Judge this
model on **F1, precision, recall and ROC-AUC**. The metrics printer emits an
explicit warning if the model collapses to a single class.

---

## 10. Threshold strategy

Both are reported every epoch:

1. Metrics at the default threshold **0.5**.
2. Metrics at the threshold that maximises **validation** F1, found by a
   91-point grid search over `[0.05, 0.95]`.

The selected threshold is saved inside `best_codebert.pt` and is what
`test_model` uses.

**The threshold is selected on validation data only.** `test_model` never
re-tunes it; a `--tune-on-test` flag is deliberately not provided, because
optimising a threshold on test predictions leaks the test set into model
selection and turns the reported F1 into an optimistic upper bound. (A
`--threshold` override exists for diagnostics and prints a loud warning.)

---

## 11. Checkpoints and resume

| File | Contents |
|---|---|
| `data/checkpoints/latest_checkpoint.pt` | epoch, batch-in-epoch, global step, model, optimizer, scheduler, GradScaler, best F1, best threshold, early-stopping counter, architecture config |
| `data/checkpoints/best_codebert.pt` | best-F1 weights + the validation-selected threshold + the metrics that justified it |

Checkpoints are written atomically (temp file + `replace`), so an interrupted
save cannot corrupt the only good checkpoint on disk.

`latest_checkpoint.pt` is ~1.5 GB: 125M fp32 parameters plus AdamW's two
moment tensors. That is expected, but it means a very small
`CHECKPOINT_EVERY` makes disk IO, not the GPU, the bottleneck. The default of
500 steps is a reasonable trade-off; `best_codebert.pt` is ~0.5 GB because it
stores weights only.

### Exact mid-epoch resume

A plain `DataLoader(shuffle=True)` **cannot** be resumed exactly: the shuffle
comes from a global RNG whose state at the start of the interrupted epoch is
unrecoverable, so a naive "resume" silently reshuffles and replays samples the
model already saw.

`ResumableSampler` solves this by deriving each epoch's permutation from a pure
function of `(seed, epoch)`. On resume the identical permutation is regenerated
and the first `batch_in_epoch * batch_size` indices are skipped, so training
continues on exactly the samples it had not yet reached — no replay, no loss.
Verified by `test_resumable_sampler_skip_resumes_exactly`.

`maybe_resume()` returns `True` only if state was genuinely restored, and
prints `No checkpoint restored - starting from scratch.` otherwise. It never
claims a resume that did not happen.

**Remaining limitation:** with `num_workers > 0` and non-deterministic CUDA
kernels, resumed runs are statistically but not bit-for-bit identical to an
uninterrupted run. The *sample order* is exact; the floating-point trajectory
may differ slightly.

---

## 12. Final test evaluation

The test split is untouched during training, hyperparameter choice, threshold
selection and early stopping. `train.py` never even opens it.

`test_model` loads `best_codebert.pt`, reads the validation-selected threshold
from it, and evaluates once.

---

## 13. Reproducibility

`set_seed` seeds Python `random`, `PYTHONHASHSEED`, NumPy, PyTorch CPU and all
CUDA devices, and sets `cudnn.deterministic=True` / `benchmark=False`.

**Complete bit-for-bit reproducibility is not guaranteed.** Several
cuDNN/cuBLAS kernels are non-deterministic by design, atomics in reduction
kernels change summation order between runs, and mixed precision makes that
ordering visible in the low bits. For stricter determinism, export
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting the process and consider
`torch.use_deterministic_algorithms(True)` (slower, and it will raise on ops
that have no deterministic kernel). Treat runs as statistically, not exactly,
reproducible.

---

## 14. Commands

Install (CPU wheels shown; use the CUDA index URL for GPU training):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt
```

Run the pipeline in order:

```bash
# 1. What is actually in data/raw/
python -m backend.ml.preprocessing.inspect_dataset

# 2. Validate labels, quality and cross-split leakage
python -m backend.ml.preprocessing.validate_dataset
python -m backend.ml.preprocessing.validate_dataset --strict   # fail on leakage

# 3. Normalise into data/processed/ (train-only dedupe)
python -m backend.ml.preprocessing.preprocess_primevul

# 4. Class distribution + the pos_weight the trainer will use
python -m backend.ml.preprocessing.balance_dataset

# 4b. OPTIONAL: undersample the majority class (compute budget only)
python -m backend.ml.preprocessing.balance_dataset --undersample --ratio 5

# 5. (optional) length profiling before chunking
python -m backend.ml.preprocessing.analyze_code_lengths
python -m backend.ml.preprocessing.analyze_token_lengths

# 6. Tokenize + chunk into data/chunked/
python -m backend.ml.preprocessing.chunk_dataset
python -m backend.ml.preprocessing.chunk_dataset --use-balanced-train

# 7. Train
python -m backend.ml.training.train
python -m backend.ml.training.train --epochs 3 --batch-size 4
python -m backend.ml.training.train --resume

# 8. Final held-out test evaluation
python -m backend.ml.evaluation.test_model
python -m backend.ml.evaluation.test_model --save-report data/test_report.json
```

Smoke run on a few hundred samples (no GPU needed):

```bash
python -m backend.ml.training.train \
    --epochs 1 --batch-size 2 --limit-train 40 --limit-valid 40 --cpu
```

Tests:

```bash
python -m pytest backend/tests/test_ml_pipeline.py -v
```

---

## 15. Module layout

```
backend/ml/
├── config.py                       central config, all env-overridable
├── preprocessing/
│   ├── inspect_dataset.py          stage 1  raw inspection
│   ├── validate_dataset.py         stage 2  validation + leakage check
│   ├── preprocess_primevul.py      stage 3  normalise -> data/processed/
│   ├── balance_dataset.py          stage 4  distribution + optional undersample
│   ├── chunk_dataset.py            stage 5  tokenize + chunk -> data/chunked/
│   ├── analyze_code_lengths.py     diagnostic
│   ├── analyze_token_lengths.py    diagnostic
│   └── __init__.py
├── models/
│   ├── dataset.py                  ChunkedFunctionDataset
│   ├── collate.py                  variable-chunk collate
│   └── codebert_classifier.py      HierarchicalCodeBERTClassifier
├── training/
│   ├── metrics.py                  metrics + threshold search
│   ├── checkpoint.py               save/load + ResumableSampler
│   ├── trainer.py                  training loop
│   └── train.py                    CLI entrypoint
├── evaluation/
│   ├── evaluator.py                shared inference
│   └── test_model.py               final test CLI
└── utils/
    ├── seed.py                     reproducible seeding
    ├── io.py                       JSONL helpers
    └── gpu.py                      memory reporting / device resolution
```

---

## 16. Limitations

- **No accuracy claim is made.** This module is built to be correct,
  reproducible and leak-free; the quality of the resulting detector depends
  entirely on training runs that have not been performed here.
- Functions longer than 8 chunks (~3.8k tokens) are truncated. Raise
  `CODESENTINEL_MAX_CHUNKS` if your length profile justifies the memory.
- Mean-pooling over chunks dilutes a single vulnerable chunk inside a long
  benign function. Attention-based chunk aggregation is the natural next
  experiment; `FUNCTION_POOLING=max` is a cheap first comparison.
- PrimeVul ships paired vulnerable/fixed functions that are near-identical.
  `validate_dataset` reports exact-duplicate overlap, but near-duplicate
  (semantic clone) leakage across splits is not detected.
- Bit-for-bit reproducibility on GPU is not guaranteed (§13).
