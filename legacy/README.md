# Legacy: parallel ML implementation from the diverged `main`

These files came from commit `81a2c12` on `origin/main`, which contained an
earlier, parallel implementation of the ML module developed alongside the one
now in `backend/ml/`.

They were moved here (not deleted) on 26 Sep 2026 while reconciling a rebase
that had merged both implementations into a single non-working tree.

## Why they were moved rather than kept in place

They target a superseded API and fail against the current one:

| Legacy file | Superseded by | Difference |
|---|---|---|
| `training_collate.py` | `backend/ml/models/collate.py` | Current version returns `function_index` for scatter-aggregation and dynamic per-batch padding |
| `loader.py` | `backend/ml/utils/io.py` | Same role (`read_jsonl`) |
| `test_metrics.py` | `backend/tests/test_ml_stages.py` | Tests `calculate_metrics`; the current module exports `compute_metrics`, `sweep_thresholds` and adds PR-AUC, Brier and MCC |
| `test_evaluator.py`, `test_model.py`, `test_dataset.py`, `test_training_collate.py`, `test_trainer.py`, `test_training_pipeline.py`, `test_overfit.py` | `backend/tests/test_ml_pipeline.py`, `backend/tests/test_ml_stages.py` | The current suite covers the same ground (93 ML tests) against the current API |

Nothing is lost: the full original state is tagged as
`backup/remote-parallel-impl` (`81a2c12`).

## Kept from this implementation

`backend/ml/preprocessing/inspect_split_overlap.py` was **retained** in the
main tree. It reports per-function detail for exact cross-split duplicates,
which complements `audit_splits.py` (that tool reports aggregate counts across
seven dimensions but not per-record detail).
