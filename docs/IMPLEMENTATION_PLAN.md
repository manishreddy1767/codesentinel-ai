# CodeSentinel-AI — Implementation Plan

**Last updated:** 19 Sep 2026
**Baseline:** `PROJECT_STATUS.md` (Phase 1 inspection, same date)

---

## 1. The governing constraint

Everything below is shaped by one fact: **a 4 GB laptop GPU**.

Measured, not assumed:

| Configuration | Throughput | Peak VRAM |
|---|---|---|
| batch 8 + gradient checkpointing | **10.5 fn/s** | 2.5 GB |
| batch 2, no checkpointing | 10.6 fn/s | 3.6 GB |
| batch 1 + checkpointing | 5.7 fn/s | 2.4 GB |
| batch 4, **no** checkpointing | **2.96 fn/s** | 4.5 GB — spills to host RAM |

Throughput collapses once peak exceeds ~4 GB, because Windows WDDM spills to
system memory. Gradient checkpointing is what *enables* the large batch; it is
not a cost here.

Derived budgets: full epoch (174,432 fn) ≈ **5.2 h**; undersampled 10:1
(61,050 fn) ≈ **1.8 h**.

A second constraint, observed: the CUDA context wedged after ~4 h of sustained
load in the previous session. Long runs must checkpoint and be resumable. Both
are verified working.

---

## 2. Priority order

Ordered by *value per hour of compute*, not by phase number.

### P0 — Finish the ML result (blocks Phases 12, 15, 16, 17, 23)

| Step | Action | Est. | Risk |
|---|---|---|---|
| P0.1 | Final training: 61,050 undersampled fn × 3 epochs, lr 1.827e-05, dropout 0.480, batch 8 + grad-ckpt | ~6 h | CUDA wedge; mitigated by per-epoch checkpoints + verified resume |
| P0.2 | Freeze threshold + calibration on **validation only** → `decision_policy.json` | ~20 min | none |
| P0.3 | **One** test evaluation with `--save-predictions` | ~20 min | irreversible: burns the one-shot guarantee |
| P0.4 | Hybrid + ablation comparison from saved predictions | ~5 min | none |

P0.3 is the only irreversible step in the project. It must not run until the
model, threshold, calibration and preprocessing are all frozen.

### P1 — Frontend integration (largest user-visible gap)

The pages exist and look finished; they simply do nothing. This is cheap to fix
and removes the most misleading part of the project — pages currently display
invented vulnerabilities.

| Step | Action | Est. |
|---|---|---|
| P1.1 | Add `/supported-languages`; extend `/analyze` response if needed | 1 h |
| P1.2 | `frontend/assets/js/api.js` — typed client, error + timeout handling | 1 h |
| P1.3 | Wire `analyze-code.html`: language select, editor, submit, loading, error, results | 2 h |
| P1.4 | Wire `analysis-results.html` / `vulnerability-details.html` to real payloads | 2 h |
| P1.5 | Replace hardcoded findings everywhere with empty states | 1 h |
| P1.6 | Safe rendering (`textContent`, never `innerHTML`, for code and messages) | 0.5 h |
| P1.7 | CORS for the dev origin; frontend integration tests | 1 h |

No bundler needed: there is no `package.json`, so plain ES modules served
statically are the right choice.

### P2 — Backend API completeness (Phase 3)

| Step | Action | Est. |
|---|---|---|
| P2.1 | `GET /supported-languages` | 0.5 h |
| P2.2 | `POST /analyze/file` — extension allow-list, size cap, never execute | 1.5 h |
| P2.3 | `GET /analysis/{id}` + persistence (SQLite) | 2 h |
| P2.4 | `POST /analyze/repository` — or formally descope | 3 h |
| P2.5 | Timeouts, request limits, structured logging without secrets | 1 h |

### P3 — GATv2 (Phase 10) — the largest open decision

Nothing exists: `backend/ml/graphs/` and `backend/ml/embeddings/` are empty, and
`torch_geometric` is not installed. Honest options:

**Option A — implement it (~12–16 h).** Graph extraction from the existing CPG
→ node features → `torch_geometric` GATv2Conv → pooling → classifier →
training → evaluation. Delivers a genuine second model and makes the
graph-ablation experiments (F–I) possible.

*Risk:* the CPG is built per-request for single functions; converting 174k
functions to graphs is a new offline pipeline of its own, and a 4 GB GPU must
then train a second model. Realistically this is a multi-day path, not an
afternoon.

**Option B — formally descope (~1 h).** Document GATv2 as NOT IMPLEMENTED with
the reason, and present the project as a CodeBERT + rules + taint hybrid. The
paper claim narrows honestly.

**Recommendation: A only if the deadline allows a multi-day slip; otherwise B.**
What must *not* happen is claiming GATv2 exists. This decision is yours.

### P4 — Documentation and audit (Phases 20, 23)

`DATASET_CARD.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`,
`RESEARCH_CONTRIBUTION.md`, `COMPLETION_AUDIT.md`. ~4 h, best written after P0
so the numbers are real.

---

## 3. Open decisions needed from you

| # | Decision | Recommendation |
|---|---|---|
| 1 | **GATv2**: implement or descope? | Descope unless a multi-day slip is acceptable |
| 2 | **GraphCodeBERT vs CodeBERT** — the brief says GraphCodeBERT; CodeBERT is implemented and tuned | Keep CodeBERT, document the deviation. Switching discards the tuning and costs another ~6 h; GraphCodeBERT's advantage comes from data-flow pretraining inputs this pipeline does not currently supply |
| 3 | **Repository analysis** endpoint | Descope; it adds traversal and safety surface for little research value |
| 4 | Train on full 174k (5.2 h/epoch) or undersampled 61k (1.8 h/epoch) | Undersampled, already prepared and justified as a compute-budget decision |

---

## 4. Sequencing

```
NOW ──► P0.1 final training (~6 h, background, resumable)
          │
          ├── in parallel (CPU-only, no GPU contention):
          │     P1 frontend integration
          │     P2 backend endpoints
          │     P4 dataset card + protocol docs
          │
          ▼
        P0.2 freeze threshold + calibration      (validation only)
          ▼
        P0.3 ONE test evaluation                 ← irreversible
          ▼
        P0.4 hybrid + ablations
          ▼
        P3 decision (GATv2 implement / descope)
          ▼
        P4 final report + completion audit
```

Frontend and backend work is CPU-only and can proceed while the GPU trains.
That parallelism is the main reason to start training first.

---

## 5. What will *not* be claimed

- No test metric until P0.3 actually runs.
- No GATv2 result unless P3 Option A completes and is evaluated.
- No novelty claim for combining AST + CFG + DFG — that is established practice.
- No IEEE-acceptance claim.

The nearest defensible contribution is empirical, and two findings already
support it: PrimeVul contains **normalized-code leakage its own de-duplication
missed** (796 test functions), and the 8-chunk cap discards **22.5% of tokens
with a 7.9-8.7x bias against vulnerable functions**. Both are measured, both affect
how published numbers on this dataset should be read, and neither depends on
GATv2 existing.
