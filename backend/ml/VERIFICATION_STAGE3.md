# CodeSentinel ML — Pipeline Verification and Hardening, Round 3

**Date:** 17 Sep 2026
**Machine:** Windows 11 · `C:\Users\rinki\codesentinel-ai` · RTX 3050 Laptop (4 GB)
**Branch:** `main` @ `c919c38`
**Scope:** Stages A–R — implementation correctness, leakage auditing, tuning,
threshold selection, calibration, experiment tracking.

> Supersedes `VERIFICATION_STAGE1.md` and `VERIFICATION_STAGE2.md` for
> everything except their dataset-availability finding, which still stands.

---

## 1. Headline

**The implementation is now verified correct on real CodeBERT. Model
performance on PrimeVul remains unmeasured, because the dataset is not on this
machine.**

Those are two separate claims and they should not be blurred:

| Claim | Status |
|---|---|
| The pipeline does what it says: gradients flow, labels align, chunks reassemble, resume is exact, no test leakage by construction | **Verified** |
| The model detects vulnerabilities at some specific precision/recall | **Unknown, and nothing in this report says otherwise** |

No accuracy number in this document is a PrimeVul result. Every number here
comes from synthetic fixtures built to exercise code paths, and the synthetic
task is trivially separable by design (`strcpy` vs `strncpy`), so its metrics
are all 1.0. That is evidence the plumbing is connected, and evidence of
nothing else.

Per the brief: **no claim is made that the model is "100% correct."** Machine
learning models cannot be guaranteed correct. What has been established is that
the *implementation* is correct, and that the measurement apparatus around it
is now honest enough to produce a trustworthy number once data exists.

---

## 2. Stage results

| Stage | Name | Result | Basis |
|---|---|---|---|
| A | Environment verification | **PASS** | CUDA swap completed and verified with a real CUDA matmul |
| B | Dataset inspection | **BLOCKED** | No PrimeVul files on this machine |
| C | Data leakage / split audit | **PASS (implementation)** · BLOCKED (real data) | New multi-dimension audit, verified against planted leaks |
| D | Preprocessing verification | **PASS** | 240/80/80 records in and out, zero silent drops |
| E | Class imbalance analysis | **PASS (implementation)** · BLOCKED (real numbers) | `pos_weight` proven to come from train only |
| F | Chunking verification | **PASS** | Real CodeBERT tokenizer, all 8 edge cases |
| G | Dataset + collate verification | **PASS** | Unit tests plus a real training run |
| H | Model architecture / fwd / bwd | **PASS** | Forward, loss, backward, encoder gradients under checkpointing |
| I | GPU / memory verification | **PASS** | Measured on the real RTX 3050; safe defaults derived, not assumed |
| J | Small overfit test | **PASS** | Real CodeBERT: loss 0.645 → 0.030, train F1 1.000 |
| K | Training smoke test | **PASS** | 2 epochs, validation, metrics, checkpoints |
| L | Checkpoint / resume | **PASS** | Full state incl. RNG; resume continues correctly |
| M | Hyperparameter tuning | **PASS (implementation)** · BLOCKED (real search) | New tuning package, smoke-tested |
| N | Threshold optimisation | **PASS (implementation)** | Four objectives, frozen to a policy file |
| O | Probability calibration | **PASS (implementation)** | Platt + isotonic, with an adopt/reject rule |
| P | Final training | **BLOCKED** | Requires data and GPU |
| Q | Final test evaluation | **BLOCKED** | Requires a trained model |
| R | Final report | **This document** | — |

**Tests: 93 passed, 0 skipped.** Up from 58 passed / 1 skipped before this
round. The previously-skipped CUDA memory test now runs and passes.

---

## 3. Blockers

### 3.1 The PrimeVul dataset is not on this machine

`data/raw/` is empty. Searched, and found nothing, in: the repository tree,
`Downloads`, `Desktop`, `Documents`, `OneDrive`, and `C:\Users\rinki`. No
`*primevul*` and no `*.jsonl` other than an unrelated log.

This is the third consecutive round to report this. The brief states the
datasets are already downloaded locally; on **this** machine they are not.
Stages B, E (real numbers), P and Q cannot run until that is resolved, and no
amount of implementation work substitutes for it.

To resolve, either place the three files in `data/raw/`, or point the pipeline
at them without copying:

```bash
export CODESENTINEL_RAW_DIR="/c/path/to/primevul"
```

### 3.2 PyTorch could not use the GPU — **RESOLVED**

At the start of this round:

```
torch 2.14.0+cpu   torch.version.cuda = None   cuda.is_available() = False
```

`torch 2.14.0+cu130` was installed (approved during the session) and verified
with a real CUDA operation, not just a capability flag:

```
torch        : 2.14.0+cu130
cuda build   : 13.0
is_available : True
device       : NVIDIA GeForce RTX 3050 Laptop GPU
VRAM         : 4.0 GiB      capability 8.6      bf16 supported: True
REAL CUDA matmul: 20x 2048^3 in 0.267 s
```

> The earlier reports recommended `cu121`. That advice is stale: there is no
> `cu121` build for Python 3.14. `cu130` is the correct wheel and matches the
> installed torch version exactly.

**One incident worth recording.** The first install attempt failed at the
uninstall step with `WinError 32` — a tuning smoke test was running in the
background and held `torch/_functorch/utils.py` open, leaving torch
half-uninstalled and unimportable. Repaired by stopping the process and
reinstalling from the cached wheel. The lesson is procedural: do not run
torch-dependent work while pip is replacing torch.

`bf16` support (capability 8.6) is worth noting for training: on Ampere,
bf16 autocast avoids the gradient-scaling failure modes of fp16 at no
throughput cost, and is the better default here.

---

## 4. Defects found and fixed

Twelve issues. Four of them would have produced quietly wrong results rather
than visible errors, which is the category worth the most attention.

### 4.1 Checkpoints recorded the wrong architecture — *silent corruption*

`save_checkpoint` read the architecture from the **module-level config** rather
than from the model instance:

```python
"chunk_pooling":    config.CHUNK_POOLING,      # the environment's value
"function_pooling": config.FUNCTION_POOLING,   # NOT the model's value
```

A model built with `function_pooling="attention"` while the environment still
said `"mean"` was recorded as `"mean"`. `load_model_from_checkpoint` then
rebuilt it as a mean-pooling model and loaded the weights into it.

This matters specifically because the brief asks for an architecture
comparison. Every non-default variant in such a comparison would have been
saved mislabelled and reloaded as the baseline — silently producing a
comparison of a model against itself. Now the instance reports its own
configuration via `arch_config()`, and `load_state_dict(..., strict=True)`
makes any residual mismatch fail loudly.

### 4.2 The best model was never restored after training

`fit()` saved the best checkpoint to disk but left `self.model` holding the
**last** epoch's weights. After early stopping those are by definition worse
than the best epoch's. Any caller evaluating the live model — rather than
knowing to reload `best_codebert.pt` by hand — measured the wrong weights.

Fixed: `fit()` reloads the best checkpoint and reports which weights are in
memory. The brief explicitly requires "restore best model behavior."

### 4.3 Early stopping had no minimum improvement

`improved = tuned_metrics.f1 > self.best_f1` — any improvement at all, down to
1e-9, reset the patience counter. On a noisy validation metric, early stopping
effectively never fires. Now gated by `min_delta` (default 1e-4).

### 4.4 Gradient accumulation broke on mid-epoch resume

```python
is_step_boundary = (offset + 1) % self.grad_accum_steps == 0
```

`offset` is the position within the *resumed* loader, not within the epoch.
Resuming at batch 37 with `grad_accum=4` put the optimizer-step boundaries on a
different phase than the uninterrupted run, so a resumed epoch took a different
number of optimizer steps. Now keyed to the absolute epoch position.

### 4.5 RNG state was not checkpointed

`ResumableSampler` made the *data order* a pure function of `(seed, epoch)`,
which is good work — but dropout draws from the global torch RNG, which was not
saved. Resume was therefore close but not exact. Now captures and restores
Python, NumPy, torch and CUDA RNG state, and reports whether it was restored.

### 4.6 Schema detection could miscount an entire split — *silent corruption*

```python
if code_field is None:
    code_field  = detect_field(record, CODE_FIELD_CANDIDATES)
    label_field = detect_field(record, LABEL_FIELD_CANDIDATES)   # same guard
```

Label detection was gated behind the *code* field being missing. A first record
carrying code but no label pinned `label_field` to `None` permanently, after
which `record.get(None)` returned `None` for every subsequent record and the
whole split was counted as having invalid labels. Each field is now detected
independently and retried until found. Covered by a regression test.

### 4.7 Subset selection could be single-class — *silent corruption*

`--limit-train` took the **head** of the file. PrimeVul-derived splits are
grouped by project and by paired vulnerable/fixed functions, so a head slice
can contain few positives or none. With zero positives, `pos_weight` silently
becomes 1.0 and the mandated overfit test proves nothing.

Now stratified and seeded: it preserves the class ratio, guarantees at least
one of each class, and returns the identical subset for a given seed so tuning
trials stay comparable.

### 4.8 Threshold search recomputed threshold-independent metrics 91 times

`find_best_threshold` called `compute_metrics` at every grid point, and each
call recomputed `roc_auc_score` — an O(n log n) quantity that does not depend
on the threshold. Replaced with a vectorised sweep: sort once, then binary
search per threshold. **Verified to agree exactly** with the reference
implementation across 546 grid points, including deliberate ties at the
boundary.

### 4.9 A memory docstring that pointed at the wrong knob

`_encode_chunks` claimed micro-batching bounds peak activation memory "rather
than by however many chunks the batch happened to expand into". Measurement
shows that holds for inference and not for training (section 6). Corrected,
because the false version would send someone debugging an OOM to the one
parameter that will not move it.

### 4.10–4.12 Smaller fixes

- `load_model_from_checkpoint` ignored the saved dropout and loaded
  non-strictly; both fixed.
- The early-stopping message said "F1" regardless of the monitored metric.
- `best_epoch` was absent from `latest_checkpoint.pt`, so a resumed run
  reported `epoch -1`.

---

## 5. What was added

### 5.1 Multi-dimensional leakage audit — `preprocessing/audit_splits.py`

Round 2 identified that leakage checking used code content only. That gap is
now closed. Seven dimensions, checked and reported **separately**, because they
do not mean the same thing:

| Dimension | Kind | Verdict on overlap |
|---|---|---|
| `exact_code`, `normalized_code` | code | **LEAKAGE** |
| `hash`, `idx`, `big_vul_idx`, `commit_id` | identity | **LEAKAGE** |
| `project` | context | **CONCERN — not leakage** |

Identity checking is the substantive addition. Content hashing cannot see a
function that was lightly edited between splits but is the *same function*
upstream — and PrimeVul ships paired vulnerable/fixed functions, so that is a
realistic risk, not a theoretical one.

Project overlap is deliberately **never** blocking. Sharing a repository across
splits is a property of the dataset's split design. It does mean held-out
metrics partly measure *within-project* generalisation, which is worth stating
next to the headline numbers — but treating it as leakage would mean deleting
most of the training set for no defensible reason.

Label-conflicting duplicates are reported separately at every level: the same
key with opposite labels is worse than a plain duplicate, because it makes the
pair unlearnable as well as leaked.

**Verified** against splits with one planted leak of each kind:

```
  train vs valid
    exact_code      code         1 shared   1 conflicting   LEAKAGE
    hash            identity     0 shared                   CLEAN
  train vs test
    exact_code      code         0 shared                   CLEAN      <- content-blind case
    hash            identity     1 shared   0 conflicting   LEAKAGE
    commit_id       identity     1 shared                   LEAKAGE
    project         context      1 shared                   CONCERN (not leakage)
```

The `train vs test` row is the point: identical upstream identity, completely
different function body. The previous implementation reported that pair as
clean.

### 5.2 Leakage remediation — `preprocessing/clean_split_duplicates.py`

Removes offending records from **training only**. Validation and test are
copied through byte-for-byte.

Removing them from evaluation instead would change what the model is measured
against, make results incomparable to published PrimeVul numbers, and bias the
held-out sets toward whatever is easy. Writing output over the input directory
is refused outright. Verified: detect → clean (2 records) → re-audit clean,
with the project *concern* correctly surviving, since cleaning must not delete
training data to make a non-problem disappear.

### 5.3 Metrics — PR-AUC, Brier, MCC

PR-AUC was missing and is now the headline ranking metric. The reason is
specific: ROC-AUC is computed from TPR and FPR, and FPR carries the negative
count in its denominator. On a 30:1 split that denominator is enormous, so a
large absolute number of false positives barely moves FPR — ROC-AUC stays
flatteringly high on exactly the models that would drown a reviewer in false
alarms. Precision uses *predicted* positives, so PR-AUC degrades honestly.

Both are reported. PR-AUC is what gets optimised. The always-positive baseline
is printed next to it, because a PR-AUC must beat the positive rate, not 0.5.

### 5.4 Hyperparameter tuning — `backend/ml/tuning/`

Optuna (TPE + median pruning) when installed, otherwise a seeded random search
over an identical space. Optuna is not currently installed, so the fallback is
what runs; nothing is blocked by that.

**The objective is validation PR-AUC, and the threshold is deliberately not
searched.** Tuning a thresholded metric jointly with the hyperparameters makes
the reported best score a maximum over two nested searches, and therefore
biased upward. PR-AUC is threshold-free, so it scores the ranking and cannot be
gamed by moving a cutoff. The threshold is chosen once, afterwards, in Stage N.

The test split is never loaded by this module. Trials record parameters,
metrics, duration, peak memory, best epoch and full history; results are
written after every trial, so an interrupted search is still analysable. OOM is
caught, scored worst, and recorded rather than crashing the sweep.

### 5.5 Threshold selection — `evaluation/select_threshold.py`

0.5 is not treated as meaningful. It is the threshold at which the odds are
even, which is only the right boundary when the classes are balanced and the
two error types cost the same — and here neither holds, not least because
`pos_weight` deliberately inflates positive scores.

Four objectives are computed and compared on validation:

| Objective | When it is the right choice |
|---|---|
| `f1` | Costs genuinely unknown (default) |
| `fbeta` (β=2) | A missed vulnerability costs much more than a false alarm |
| `mcc` | Robust under heavy imbalance |
| `recall_at_precision` | Reviewer attention is the binding constraint |

That last one deserves the emphasis. Below some precision a security tool gets
muted, and **a muted tool has zero recall in practice** — so maximising recall
without a precision floor can make things worse in the field than a more
conservative threshold would.

The choice is a policy decision, not a mathematical one. The module reports all
four, selects the one requested, and freezes it to `decision_policy.json`
alongside the objective used, so the final numbers can be read in context.

### 5.6 Calibration — `evaluation/calibration.py`

Brier, ECE, MCE and reliability curves, plus Platt and isotonic calibrators.
Quantile binning by default, because with a 30:1 split equal-width bins leave
the upper bins nearly empty and ECE becomes noise.

Whether calibration helps is judged by **stratified K-fold cross-validation
inside the validation set**, so the improvement estimate comes from data the
calibrator never saw. The final calibrator is refit on all of validation and
frozen. The test set is never involved.

**An important subtlety, found by a test that initially failed.** My first
adoption rule accepted calibration on a model that was perfectly calibrated by
construction. ECE is a biased estimator: each bin's observed rate is a binomial
mean that deviates from the predicted rate by pure sampling noise, and the
absolute value in the ECE sum turns that noise into a positive number that
never cancels. Any two-parameter calibrator reduces measured ECE simply by
shrinking predictions toward the base rate.

So adoption now requires the measured ECE to exceed an **analytic noise floor**
— the ECE a perfectly calibrated model of that sample size would still show,
`sqrt(2/π)·sqrt(p(1-p)/n)` averaged over bins — by a factor of two, *and* an
out-of-fold relative gain, *and* no Brier regression. Verified in both
directions: adopted on a deliberately overconfident model (ECE 0.041 → 0.006),
declined on a well-calibrated one.

Calibrators serialise to plain JSON (round-trip verified exact) rather than
pickled estimators, so a frozen policy stays loadable across library versions.

### 5.7 Experiment tracking — `utils/experiment.py`

Per-run directories under `artifacts/` with environment snapshot (versions,
GPU, git commit **and dirty-tree flag**, `CODESENTINEL_*` overrides), config,
metrics and an append-only event log. Dependency-free by choice: no server to
stand up, and plain JSON outlives the tooling that wrote it.

The dirty-tree flag matters — a dirty tree means the commit hash does not fully
describe the code that produced the numbers, which is worth recording rather
than hiding.

### 5.8 Overfit test — `training/overfit_test.py`

Stage J as a first-class tool. A model that cannot fit 32 examples it sees
repeatedly has a bug — a detached graph, a frozen encoder, misaligned labels, a
learning rate too small to move anything. Each of those also produces a
plausible-looking curve on the full dataset, where slow progress is
indistinguishable from no learning. On 32 samples it is unambiguous.

It runs *before* any search, because searching over a broken implementation
just finds the configuration that hides the bug best.

### 5.9 Test-evaluation ledger

The test split is meant to be read once, after every choice is frozen. Nothing
can technically prevent a second run, but an append-only ledger makes it
visible: if it lists several different checkpoints, the "held-out" number was
selected on. Verified to warn on a repeat with a different checkpoint.

### 5.10 Attention pooling

Added as a **third option** alongside the existing `mean` and `max`. Mean
remains the default and the baseline is always a candidate in any comparison.
Uses a numerically-stable segment softmax — the per-function max is subtracted
before `exp()`, so it cannot overflow in fp16. Verified that a uniform
attention head reduces exactly to the mean, which would fail if the softmax
normalised across the batch instead of within each function.

---

## 6. Evidence

### Stage J — overfit test, real CodeBERT

```
subset: 32 functions (8 vulnerable / 24 benign, stratified)

  epoch    loss     grad_norm   train_f1   train_acc
      1   0.6453     10.8243     0.0000      0.7500
      4   0.4356      2.6565     0.0000      0.7500
      6   0.1877      4.5604     1.0000      1.0000
      8   0.0300      0.1511     1.0000      1.0000

  first-epoch loss : 0.6453
  final loss       : 0.0300  (95.3% reduction)
  PASS
```

Gradients flow, the optimizer updates weights, and labels line up with the
functions they are attached to. This is the single most informative result in
the report.

### Stage F — chunking, real CodeBERT tokenizer

```
empty           chunks=1  lens=[2]              valid=True
tiny            chunks=1  lens=[8]              valid=True
exactly_body    n=510     chunks=1  lens=[512]           trunc=False
body_plus_1     n=511     chunks=2  lens=[512, 131]      trunc=False
body_times_2    n=1020    chunks=3  lens=[512, 512, 258] trunc=False
very_long       chunks=8                                 trunc=True

overlap: step=382  expected=128  actual=128  contiguous=True
```

Every chunk ≤ 512, every chunk starts with `<s>` and ends with `</s>`, no empty
chunks, empty input still yields one valid chunk, truncation is recorded.

### Stage I — GPU memory, measured on the real device

Worst case (every function at the 8-chunk cap, 512 tokens), AMP and gradient
checkpointing on, 15% headroom reserved:

```
 batch  micro  chunks   peak MiB  reserved  verdict
     1      4       8       2181      2548  fits
     2      4      16       3647      4108  over budget
     4      4      32       6585      7250  over budget
     8      4      64          -         -  OOM

RECOMMENDATION:  --batch-size 1 --chunk-micro-batch 4 --grad-accum 8
measured peak :  2181 MiB of 4096 MiB       effective batch: 8
```

**This contradicts the earlier reports' estimate** of
`--batch-size 2 --chunk-micro-batch 4 --grad-accum 4`. That configuration needs
~3.6 GB against a 4 GB card and does not survive a worst-case batch. The
estimate was reasonable arithmetic; it was simply wrong, which is the argument
for measuring.

#### `chunk_micro_batch` does not do what its docstring claimed

The probe showed peak training memory barely moving across micro-batch sizes.
Isolating it (32 chunks x 512 tokens, peak MiB above baseline):

```
micro_batch      1     2     4     8    16    32
inference       81    79   108   168   288   528     <- 6.7x range
training      1260  1213  1167  1225  1290  1619     <- ~1.4x range
```

Under `no_grad` each micro-batch's activations are freed as soon as its
embeddings are taken, so micro-batching is the dominant lever. Under grad,
autograd must retain what backward needs across *every* micro-batch, so peak
memory tracks the number of chunks in the batch no matter how they were fed
through. The same measurement without gradient checkpointing showed the same
flatness, confirming the cause is autograd retention rather than checkpointing.

The docstring asserted that peak activation memory is "bounded by
`chunk_micro_batch` rather than by however many chunks the batch happened to
expand into". That is true for evaluation and false for training. It has been
corrected in both `codebert_classifier.py` and the probe, because it would have
sent someone hitting an OOM to tune the one knob that will not help. The
effective training levers are **batch size** and **MAX_CHUNKS_PER_FUNCTION**,
which reduce the chunk count itself.

### Stage L — checkpoint contents

```
model_state                  201 tensors
optimizer_state              present
scheduler_state              present
scaler_state                 present
epoch / global_step / batch_in_epoch
best_f1 / best_threshold / epochs_without_improvement
rng_state                    keys=['numpy', 'python', 'torch']      <- new
config    {..., 'function_pooling': 'mean', 'dropout': 0.1, ...}    <- from the instance
extra     {'pos_weight': 3.0, 'monitor': 'pr_auc', 'best_epoch': 1}
```

Resume verified: `RNG state: restored`, continued at the correct epoch, early
stopping fired correctly.

---

## 7. Honest limitations

1. **No PrimeVul result exists.** Everything above is synthetic-data evidence
   of implementation correctness.
2. **The synthetic task is trivially separable**, so its 1.0 metrics are
   meaningless as performance and are reported only as plumbing checks.
3. **The GPU path is verified, but only at the probe level.** A real CUDA
   matmul, a real forward/backward, and the memory grid all ran. A long
   training run on real data has not, so fragmentation over many hours and the
   eval-pass peak alongside training state remain unproven.
4. **4 GB VRAM is tight, and measurably tighter than previously estimated.**
   Only `--batch-size 1` survives a worst-case batch. With `--grad-accum 8`
   that still gives an effective batch of 8, but it means roughly 8x more
   optimizer steps per epoch than a batch of 8 would, and correspondingly
   longer wall-clock.
5. **Full training on this laptop will be slow.** ~180k training functions
   averaging multiple 512-token chunks, at an effective batch of 8 forced into
   batch-size-1 steps on a 4 GB laptop GPU, is plausibly days per run. That
   makes the tuning budget a real constraint, not a formality: prefer few
   trials on a subsample over many trials on everything, and consider whether
   this model needs to be trained here at all.
6. **Optuna is not installed**, so tuning falls back to seeded random search.
   Reproducible and a strong baseline; it simply lacks pruning. Given the
   wall-clock constraint above, installing it for median pruning is probably
   worth it — a pruned bad trial is hours saved, not minutes.
7. **Project-level overlap is expected to be substantial** in PrimeVul. When
   real numbers arrive, they should be described as partly within-project
   generalisation rather than as clean cross-project results.
8. **The chunking strategy caps functions at 8 chunks** (~3,184 tokens).
   Functions longer than that are truncated, and a vulnerability past the cut
   is invisible to the model. The truncation rate on real PrimeVul is unknown
   and should be measured at Stage B.

---

## 8. What to do next

**Order matters.** Steps 1 and 2 are independent of each other; everything
after step 3 depends on both.

1. **Provide the dataset** — copy into `data/raw/`, or set
   `CODESENTINEL_RAW_DIR`. This is now the *only* remaining blocker.
2. **Inspect and audit** (Stages B, C):
   ```bash
   python -m backend.ml.preprocessing.inspect_dataset
   python -m backend.ml.preprocessing.audit_splits --dir data/raw --strict \
       --json artifacts/reports/leakage_raw.json
   ```
   **If identity or code leakage is found, stop and clean before training.**
   Project-only overlap is expected; note it and continue.
3. **Preprocess, then re-audit, then chunk** (Stages D, F). Record the
   truncation rate at the 8-chunk cap while you are there.
4. **Re-run the memory probe** (Stage I) against *real* chunk-count
   statistics, since this round used the synthetic worst case:
   ```bash
   python -m backend.ml.training.memory_probe
   ```
5. **Overfit test** (Stage J) — must pass before anything expensive:
   ```bash
   python -m backend.ml.training.overfit_test --samples 32
   ```
6. **Smoke search, then the real search** (Stage M) — a 3-trial smoke run
   before committing compute. Consider `pip install optuna` first for pruning.
7. **Train once** with the winning configuration (Stage P).
8. **Freeze the decision policy** (Stages N, O) on validation.
9. **Evaluate the test set exactly once** (Stage Q) with `--policy`.

Report the Stage Q numbers with the threshold objective, the calibration
decision, and the project-overlap caveat attached. A precision/recall pair
without its operating point is not a result.

---

## 9. Compliance with the working rules

- `data/raw/` was never written to. It is empty, and nothing was placed in it.
- No dataset was balanced, overwritten or deleted. The cleaning tool refuses to
  write over its input.
- No architecture was replaced. Attention pooling was *added*; `mean` remains
  the default and the baseline stays available.
- `pos_weight` is computed from the training split only, and a test asserts it.
- The test split was not used for any selection. Tuning, thresholding and
  calibration all read validation only.
- No uncontrolled architecture search was run.
- Existing imports and file names were preserved; `collate.py` was not renamed.
- No large artifacts are committed — `artifacts/` is git-ignored.
- **No claim is made that the model is correct or accurate.** The
  implementation is verified; model performance is unknown.
