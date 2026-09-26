# Reproducibility — CodeSentinel-AI

**Last updated:** 19 Sep 2026

Everything here was run on the machine described in §1. Commands are the exact
ones executed, not idealised versions.

---

## 1. Environment

| Component | Version |
|---|---|
| OS | Windows 11 Home Single Language (build 26200) |
| Shell | MINGW64 / Git Bash, plus PowerShell for host queries |
| CPU | AMD Ryzen 7 5800H — 8 cores / 16 threads |
| RAM | 15.3 GB |
| GPU | NVIDIA RTX 3050 Laptop — 4.0 GiB, capability 8.6, 16 SMs |
| GPU driver | 592.82 · VBIOS 94.07.56.40.18 · PCIe gen3 ×8 |
| Python | 3.14.3 (`.venv`) |
| PyTorch | 2.14.0+cu130 (CUDA build 13.0, cuDNN 92400) |
| transformers | 5.17.0 |
| scikit-learn | 1.9.1 · numpy 2.5.3 · scipy 1.18.1 |
| optuna | 5.0.0 |
| tree-sitter | 0.26.0 (+ c 0.24.2, cpp 0.23.4, python 0.25.0, java 0.23.5, javascript 0.25.0) |
| fastapi | 0.141.1 · pydantic-settings 2.15.0 · uvicorn 0.53.0 · python-multipart 0.0.32 |
| gdown | 6.4.0 |
| **torch_geometric** | **not installed** — no graph model exists |

**Power state matters on this machine.** On battery the GPU is clamped to
`enforced.power.limit = 30 W` (default 60 W) and reports
`SwPowerCap + SwThermalSlowdown` continuously at ~1,290 MHz. On AC it runs at
60 W and ~1,800 MHz. Three CUDA crashes occurred while on battery. **Run
training on AC power.**

### Determinism

`set_seed(42)` seeds Python, NumPy, torch and CUDA, and sets
`cudnn.deterministic = True`, `cudnn.benchmark = False`.

Full bit-for-bit reproducibility is **not** claimed: several cuDNN/cuBLAS
kernels are non-deterministic by design, atomics change summation order, and
AMP makes that visible in the low bits. Runs are statistically, not exactly,
reproducible. Data order *is* exactly reproducible — `ResumableSampler` derives
each epoch's permutation from `(seed, epoch)`, and RNG state is checkpointed.

---

## 2. Setup

```bash
git clone <repo> && cd codesentinel-ai
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt

# CUDA build matching this driver (do NOT use cu121 — no Python 3.14 wheel):
.venv/Scripts/python -m pip install "torch==2.14.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130

# Added during this project:
.venv/Scripts/python -m pip install optuna gdown uvicorn python-multipart \
    fastapi pydantic-settings httpx tree_sitter tree_sitter_c tree_sitter_cpp \
    tree_sitter_python tree_sitter_java tree_sitter_javascript

# Verify CUDA with a real operation, not just the availability flag:
.venv/Scripts/python -c "import torch; a=torch.randn(1024,1024,device='cuda'); \
    torch.cuda.synchronize(); print(torch.__version__, float((a@a).sum()))"
```

---

## 3. Data pipeline — exact commands

```bash
# 1. Acquire (official Google Drive folder 19iLaNDS0z99N8kB_jBRTmDLehwZBolMY)
#    file ids: train 1qRO_Qdy7KXcZbJJAu5J3VZWkRVvT4Kbu
#              valid 1CMQ185Ww_bsBWGbJe4sZW0vzceWnmNE7
#              test  1fVybeCHhFfBMOHQ5rGSA-mzu9Egri38H
#    -> data/raw/, then set read-only. Checksums in data/raw_checksums.json.

# 2. Inspect
python -m backend.ml.preprocessing.inspect_dataset

# 3. Leakage audit on RAW
python -m backend.ml.preprocessing.audit_splits --dir data/raw \
    --json artifacts/reports/leakage_raw.json

# 4. Preprocess (train-only dedupe; raw untouched)
python -m backend.ml.preprocessing.preprocess_primevul

# 5. Remove leaked records from TRAIN only
python -m backend.ml.preprocessing.clean_split_duplicates \
    --in-dir data/processed --out-dir data/processed_clean \
    --dimensions exact_code normalized_code hash idx big_vul_idx commit_id

# 6. Re-audit (exits 1 on residual valid<->test overlap — expected, documented)
python -m backend.ml.preprocessing.audit_splits --dir data/processed_clean \
    --strict --json artifacts/reports/leakage_clean.json

# 7. Class distribution / pos_weight
python -m backend.ml.preprocessing.balance_dataset --dir data/processed_clean

# 8. Tokenize + chunk
python -m backend.ml.preprocessing.chunk_dataset \
    --in-dir data/processed_clean --out-dir data/chunked

# 9. Optional: 10:1 undersample, TRAIN ONLY (compute budget)
python -m backend.ml.preprocessing.undersample_chunked --ratio 10
```

---

## 4. Verification gates (run before spending GPU time)

```bash
python -m backend.ml.training.memory_probe                    # measured, not assumed
python -m backend.ml.training.overfit_test --samples 48 --balanced
```

The overfit test is a **debugging gate**, not a performance claim: it asks
whether the optimiser can fit 48 examples at all.

---

## 5. Tuning, training, evaluation

```bash
# Tuning — validation PR-AUC objective, test split never loaded
python -m backend.ml.tuning.tune --name primevul_search --trials 4 --epochs 1 \
    --train-file data/chunked/primevul_train_chunked_under10.jsonl \
    --limit-train 9000 --limit-valid 4000 --backend optuna \
    --fix batch_size=8 grad_accum_steps=1 --monitor pr_auc

# Final training, crash-resilient (see §6)
python scripts/train_supervised.py \
    --checkpoint-dir data/checkpoints --log /tmp/final_train3.log \
    --max-attempts 20 --cooldown 45 -- \
    --train-file data/chunked/primevul_train_chunked_under10.jsonl \
    --chunked-dir data/chunked --checkpoint-dir data/checkpoints \
    --epochs 3 --batch-size 8 --grad-accum 1 --chunk-micro-batch 16 \
    --eval-batch-size 16 --lr 1.827e-05 --dropout 0.480 \
    --weight-decay 0.0732 --warmup-ratio 0.090 --monitor pr_auc \
    --patience 2 --seed 42 --checkpoint-every 250 --high-loss-threshold 0

# Checkpoint integrity (fresh process, CPU-only — safe during training)
python scripts/verify_checkpoint.py --checkpoint data/checkpoints/latest_checkpoint.pt

# Threshold + calibration — VALIDATION ONLY, writes decision_policy.json
python -m backend.ml.evaluation.select_threshold \
    --checkpoint data/checkpoints/best_codebert.pt --chunked-dir data/chunked

# THE single test evaluation
python -m backend.ml.evaluation.test_model \
    --checkpoint data/checkpoints/best_codebert.pt --chunked-dir data/chunked \
    --policy data/checkpoints/decision_policy.json \
    --save-report artifacts/reports/test_report.json --save-predictions

# Baselines + hybrids (hybrids reuse saved predictions — test scored once)
python -m backend.ml.evaluation.baselines --dir data/processed_clean --split test
python -m backend.ml.evaluation.compare_baselines \
    --ml artifacts/reports/test_report.json \
    --baselines artifacts/reports/baselines_test.json

# Post-hoc analyses. Both read the saved predictions and never run the model,
# so they can be re-run freely without touching the one-shot test budget.
python scripts/error_analysis.py \
    --ml artifacts/reports/test_report.json --split test \
    --baselines artifacts/reports/baselines_test.json
python scripts/sensitivity_valid_test_overlap.py \
    --ml artifacts/reports/test_report.json
```

**Order matters at one point only.** `select_threshold.py` must be re-run
against whichever checkpoint training finally leaves in `best_codebert.pt`. The
threshold and the isotonic knots are fitted to a specific score distribution; a
policy carried over from an earlier epoch's checkpoint puts the operating point
somewhere nobody chose. `test_model.py` cannot detect this, because a stale
policy is still a structurally valid one.

Everything after `test_model.py` is pure arithmetic on the saved predictions, so
the test split is *scored* exactly once regardless of how many times the
analyses are repeated. `artifacts/test_evaluations.jsonl` is the append-only
record of how many times it was actually scored.


---

## 6. Reliability tooling

```bash
python scripts/gpu_monitor.py --out artifacts/reports/gpu_telemetry.jsonl \
    --checkpoint-dir data/checkpoints --interval 30
```

`train_supervised.py` restarts after a crash **only while forward progress
continues**; two failures at the same `global_step` abort it, so a
deterministic bug cannot be hidden by retrying. Crash count and timestamps are
written to `data/checkpoints/supervisor_summary.json`.

---

## 7. Tests and services

```bash
python -m pytest backend/tests/ -q          # 154 passing as of 19 Sep 2026

cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# serve frontend/ on port 5500 (CORS is configured for it) and open
# pages/analyze-code.html
```

---

## 8. Artifacts and git policy

Git-ignored (regenerable, large): `data/`, `artifacts/`, `*.pt`.
Committed: source, tests, docs, `data/README.md`.

Preserved for provenance:
`data/raw_checksums.json` · `artifacts/reports/leakage_raw.json` ·
`leakage_clean.json` · `baselines_test.json` · `gpu_telemetry.jsonl` ·
`EXPERIMENT_LOG.md` · `artifacts/tuning/*/trials.json` ·
`artifacts/test_evaluations.jsonl` (the one-shot ledger).

---

## 9. What cannot be reproduced from this repository

- **GraphCodeBERT** and **GATv2** results — neither is implemented.
- Bit-exact training runs (§1).
- The three CUDA crashes — hardware/driver dependent, and the machine was on
  battery at a 30 W cap at the time.
