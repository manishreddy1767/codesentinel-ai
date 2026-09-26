# CodeSentinel ML — Verification Report, Round 2

**Date:** 12 Sep 2026
**Machine:** Windows 11 · `C:\Users\rinki\codesentinel-ai`
**Branch:** `main` @ `c919c38` (synced with `origin/main`)
**Scope:** post-merge re-verification — sync, Stage 1, Stage 2 audit, Stages 8–9

> Supersedes `VERIFICATION_STAGE1.md`, which was written before the ML branch
> was merged and described the push as rejected. That is now resolved.

---

## Summary

| Outcome | Detail |
|---|---|
| Repository sync | **PASS** — local `main` fast-forwarded to `origin/main` |
| Stage 1 — Dataset verification | **FAIL** — raw datasets absent from this machine |
| Stage 2 — Data leakage check | **BLOCKED** — cannot run; implementation audited, one gap found |
| Stage 8 — Test suite | **PASS** — 58 passed, 1 skipped |
| Stage 9 — GPU / environment | **FAIL** — CPU-only PyTorch, GPU unusable |
| **Full training ready** | **NO** |

Two blockers remain, unchanged from the previous round:

1. The PrimeVul datasets are not on this machine.
2. PyTorch is a CPU-only build and cannot use the RTX 3050 that is present.

---

## Task 0 — Repository synchronization

### What was checked

Current branch, working-tree status, `git fetch --all --prune`, and a diff of
local history against the remote.

### Result: PASS

```
origin/main advanced:  76bb58b..c919c38

  c919c38  Merge pull request #1 from manishreddy1767/ml/codebert-vulnerability-classifier
  d9a297c  Build complete CodeBERT vulnerability classifier ML module
```

Local `main` was fast-forwarded to `c919c38` and now matches `origin/main`
exactly. The working tree is clean apart from untracked report files under
`backend/ml/`.

Remote branches present:

```
origin/HEAD -> origin/main
origin/main
origin/ml/codebert-vulnerability-classifier
```

### Note on provenance

The work described as "pushed from another laptop" is commit `d9a297c`, which
was authored on **this** machine and has returned via PR #1. The 34 files and
5,628 insertions on `origin/main` are therefore identical to what was already
present locally. Nothing was overwritten, and there is no unfamiliar code to
reconcile.

### Changes made

None to source. `git checkout main` and a fast-forward merge only.

---

## Module inventory

| Expected path | Status | Lines |
|---|---|---|
| `backend/ml/config.py` | PRESENT | 201 |
| `backend/ml/models/codebert_classifier.py` | PRESENT | 224 |
| `backend/ml/models/training_collate.py` | **ABSENT** | — |
| `backend/ml/models/collate.py` | PRESENT | 76 |
| `backend/ml/models/dataset.py` | PRESENT | 144 |
| `backend/ml/training/train.py` | PRESENT | 210 |
| `backend/ml/training/trainer.py` | PRESENT | 568 |
| `backend/ml/training/metrics.py` | PRESENT | 172 |
| `backend/ml/training/checkpoint.py` | PRESENT | 175 |
| `backend/ml/evaluation/evaluator.py` | PRESENT | 113 |
| `backend/ml/evaluation/test_model.py` | PRESENT | 170 |
| `backend/tests/test_ml_pipeline.py` | PRESENT | 1221 |

`backend/ml/preprocessing/`:

```
__init__.py
analyze_code_lengths.py
analyze_token_lengths.py
balance_dataset.py
chunk_dataset.py
inspect_dataset.py
preprocess_primevul.py
validate_dataset.py
```

### Naming discrepancy — no action taken

The verification request names `backend/ml/models/training_collate.py`. The
actual file is `backend/ml/models/collate.py`.

Treating the repository as source of truth, **the file has not been renamed.**
The trainer, evaluator and test suite all import
`backend.ml.models.collate`; renaming without updating every importer would
break the pipeline, and renaming *with* updates is churn with no functional
benefit. If the `training_collate.py` name is required, it can be done properly
across all importers on request.

---

## Stage 1 — Dataset verification

### What was checked

Existence of the three raw files, plus two independent system-wide scans in
case the data was stored elsewhere or under a different name.

### Result: FAIL

```
$ ls -la data/raw/
total 0
(empty directory)

data/raw/primevul_train.jsonl  -- MISSING
data/raw/primevul_valid.jsonl  -- MISSING
data/raw/primevul_test.jsonl   -- MISSING
```

`inspect_dataset` confirms:

```
Looking in: C:\Users\rinki\codesentinel-ai\data\raw

  SPLIT: TRAIN  -> MISSING: file does not exist.
  SPLIT: VALID  -> MISSING: file does not exist.
  SPLIT: TEST   -> MISSING: file does not exist.
```

System scans:

| Scan | Result |
|---|---|
| All `*.jsonl` on `C:\`, excluding tool/cache/test noise | **NONE** |
| Any file > 20 MB written in the last 12 h under `C:\Users\rinki` | Only Chrome cache, pip cache, VS Code extension binaries |

### Problem found

The datasets are not on this machine — not in `data/raw/`, not anywhere on
`C:\`, under any name. The repository sync brought the **code** across, but
`data/` is git-ignored by design; datasets never travel through GitHub. If they
were downloaded on the other laptop, they remain only on that laptop.

### Changes made

None. Nothing under `data/` was created, modified, or deleted.

### Safe to continue?

**No.** Stages 2–7 and 10–15 all require real data.

---

## Stage 2 — Data leakage check

### Status: BLOCKED — cannot execute without data

No leakage result is reported here. Fabricating one would be worse than
reporting none.

### Implementation audit

Since the check could not be run, its implementation was read instead, so the
method is known in advance. From `backend/ml/preprocessing/validate_dataset.py`:

**Method.** Each function's code is whitespace-normalised
(`" ".join(code.split())`), then hashed with SHA-256. Hash sets are compared
across all three split pairs.

```python
def _normalize_for_hash(code: str) -> str:
    # Collapse whitespace so that reformatting alone does not hide a duplicate.
    # Deliberately conservative: it does not strip comments or rename
    # identifiers, so it detects copies rather than semantic clones.
    return " ".join(code.split())
```

**Pairs compared:** `train ↔ valid`, `train ↔ test`, `valid ↔ test`.

**Reported per pair:** `CLEAN` or `LEAKAGE`, the count of shared functions, and
the count of **label-conflicting** shared functions — the same code appearing
on both sides with opposite labels, which is worse than a plain duplicate
because it makes that pair unlearnable as well as leaked.

**Also reported:** duplicate functions *within* each split, counted separately
and treated as a warning rather than a failure.

**Failure behaviour:** leakage is reported as a finding and the run exits 0 by
default; `--strict` promotes leakage to a blocking failure with exit 1.

Expected output shape:

```
CROSS-SPLIT LEAKAGE CHECK
  train  vs valid : CLEAN  shared=0  label-conflicting=0
  train  vs test  : CLEAN  shared=0  label-conflicting=0
  valid  vs test  : CLEAN  shared=0  label-conflicting=0
```

### Gap found — identifier-based overlap is not checked

The Stage 2 requirement asks for overlap checked against `hash`,
function/code content, `project`, and commit identifiers. A grep confirms the
current implementation uses **code content only**:

```
$ grep -n "hash\"|'hash'|project|commit_id" backend/ml/preprocessing/validate_dataset.py
  NO -- code-content hashing only
```

So today the check does **not** use:

- PrimeVul's own `hash` field
- `project`
- `commit_id`
- `idx` / `big_vul_idx`

**Why this matters.** Content hashing catches byte-identical and
reformatting-only copies. It will miss a function that was lightly edited
between splits but is the *same* function by upstream identity — and PrimeVul
ships paired vulnerable/fixed functions, so identity-based overlap is a
realistic risk. It also cannot report *project-level* overlap, where the same
codebase appears across splits; that is not strictly leakage, but it inflates
generalisation estimates and is worth measuring.

**Proposed minimal fix** (not applied — awaiting approval, and the real data is
needed to validate it):

1. Extend `validate_split` to collect the metadata identifiers present in the
   records, alongside the existing content hashes.
2. Extend `check_cross_split_leakage` to report overlap per identifier type
   separately, so content overlap and identity overlap are distinguishable
   rather than conflated.
3. Keep `project` overlap as an informational statistic, not a failure —
   project-level sharing is a property of PrimeVul's split design, not
   necessarily a defect.
4. Add tests covering identifier overlap with and without content overlap.

This is additive. It does not change existing behaviour or any other stage.

### Safe to continue?

**No** — the stage has not run.

---

## Stage 8 — Test suite

### What was checked

Full ML test suite with skip reasons reported.

### Result: PASS

```
$ python -m pytest backend/tests/test_ml_pipeline.py -q -rs

58 passed, 1 skipped in 25.45s

SKIPPED [1] backend/tests/test_ml_pipeline.py:760: CUDA not available
```

No failures, no errors. The single skip is
`test_gpu_memory_does_not_grow_across_batches`, which executes automatically
once CUDA is available.

Coverage: chunking invariants, per-chunk special tokens, dataset and collate
shapes, attention masks, chunk counts, model output shape vs target shape, loss
computation, backward pass, encoder gradients under gradient checkpointing,
checkpoint round-trip, exact mid-epoch resume, metrics, threshold search.

### Caveat

These tests run against **synthetic fixtures**. They establish that the
implementation is internally correct; they do not establish that it behaves
correctly on real PrimeVul data. Stages 2–7 and 10–15 exist precisely to close
that gap.

---

## Stage 9 — GPU and environment verification

### What was checked

Python, PyTorch, CUDA availability, a live CUDA tensor operation, GPU identity,
and supporting library versions.

### Result: FAIL

```
Python              : 3.14.3 | Windows-11-10.0.26200-SP0
torch               : 2.14.0+cpu
torch.version.cuda  : None   <-- CPU-only wheel
cuda.is_available() : False
device count        : 0
GPU name            : n/a (torch cannot see any device)
CUDA tensor op      : SKIPPED
transformers        : 5.17.0
scikit-learn        : 1.9.1
numpy               : 2.5.3
```

Physical hardware:

```
NVIDIA GeForce RTX 3050 Laptop GPU
driver 592.82
4096 MiB total, 0 MiB used
```

### Problem found

The GPU and driver are healthy and idle, but the installed PyTorch is the
CPU-only wheel and cannot address it. Mixed precision is therefore unavailable
— the trainer auto-disables AMP when running on CPU. Training must not begin in
this state.

### Fix (Windows Git Bash) — not applied, ~2.5 GB download

```bash
cd /c/Users/rinki/codesentinel-ai
.venv/Scripts/python.exe -m pip uninstall -y torch
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected afterwards: a `+cu121` version string, a CUDA version, and `True`.

### Capacity planning — 4 GB VRAM

Approximate fp32 budget:

| Component | Size |
|---|---|
| Weights | ~0.50 GB |
| Gradients | ~0.50 GB |
| AdamW m + v | ~1.00 GB |
| **Steady state** | **~2.00 GB** |
| Remaining for activations | ~1.8 GB |

Expected starting configuration, to be **measured** at Stage 11 rather than
assumed:

```bash
--batch-size 2 --chunk-micro-batch 4 --grad-accum 4
```

Effective batch 8, with AMP and gradient checkpointing both enabled. If long
multi-chunk functions trigger OOM, the next lever is reducing
`CODESENTINEL_MAX_CHUNKS` from its default of 8.

---

## Stages not run

| Stage | Name | Reason |
|---|---|---|
| 2 | Data leakage check | No data |
| 3 | Preprocessing verification | No data |
| 4 | Class imbalance analysis | No data |
| 5 | Chunking verification | No data |
| 6 | Collate function verification (on real data) | No data |
| 7 | Model architecture verification (on real data) | No data |
| 10 | Model forward pass | No data |
| 11 | GPU memory test | No data, no CUDA |
| 12 | Training smoke test | No data, no CUDA |
| 13 | Checkpoint and resume verification | No data, no CUDA |
| 14 | Validation pipeline | No data |
| 15 | Threshold selection | No data |

None were skipped for convenience. Each has a hard dependency on real data, a
CUDA-capable PyTorch build, or both.

---

## Stage 16 — Go / No-Go

```
DATA:                  FAIL      files absent from this machine
DATA LEAKAGE:          BLOCKED   implementation audited; gap found
PREPROCESSING:         BLOCKED
CLASS IMBALANCE:       BLOCKED
CHUNKING:              BLOCKED
COLLATE FUNCTION:      BLOCKED   unit-tested, not verified on real data
MODEL FORWARD PASS:    BLOCKED
ENCODER GRADIENTS:     BLOCKED   unit-tested, not verified on real data
LOSS FUNCTION:         BLOCKED
GPU:                   FAIL      CPU-only torch; RTX 3050 unused
GPU MEMORY:            BLOCKED
CHECKPOINTING:         BLOCKED
RESUME TRAINING:       BLOCKED
VALIDATION PIPELINE:   BLOCKED
THRESHOLD SELECTION:   BLOCKED
TEST SUITE:            PASS      58 passed, 1 skipped

FULL TRAINING READY:   NO
```

---

## Action required

### A. Datasets — choose one

**A1 — Copy them onto this machine.** Place the three files in `data/raw/`, then
confirm:

```bash
ls -la /c/Users/rinki/codesentinel-ai/data/raw/
```

All three `.jsonl` files should be listed. Verification then resumes at Stage 2.

**A2 — They are elsewhere on this disk.** Point the pipeline at them in place,
without copying:

```bash
export CODESENTINEL_RAW_DIR="/c/path/to/primevul"
```

**A3 — They exist only on the other laptop.** Either run this verification
there, or transfer the files. They are large, so a USB drive or network share
is preferable to re-downloading.

### B. GPU

Approve the CUDA reinstall in Stage 9, or run it manually.

### C. Optional — Stage 2 leakage extension

Decide whether the identifier-based overlap check described above should be
added before the first full training run, or deferred. It is additive and does
not affect any other stage.

---

## Working-rule compliance

- `data/raw/` was never modified — it is empty, and nothing was written to it.
- No dataset was balanced, overwritten, or deleted.
- No source file was rewritten during this round.
- No hardcoded schema assumptions were introduced; field detection remains
  automatic.
- `pos_weight` remains computed from the training split at runtime, never
  hardcoded.
- No training of any kind was started.
- No claim is made that the model is correct or accurate. The **implementation**
  is verified against synthetic fixtures; real-data behaviour and model
  performance are both unknown until the blockers are cleared.
