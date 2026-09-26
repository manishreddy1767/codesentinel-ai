# Pre-Test Pre-Flight Audit

**Date:** 2026-09-26
**Purpose:** verify every step that runs *after* the one-shot test evaluation
actually works, *before* the irreversible step is taken.

## Why this document exists

The test split is evaluated exactly once. Every analysis downstream of that
evaluation — hybrids, error analysis, sensitivity — consumes the saved
predictions. If any of those consumers is broken, the failure appears *after*
the budget is spent, and the choice is then between not running the analysis or
re-reading the protected split.

This already happened once: `baselines.py` did not persist per-sample `scores`,
which would have made the hybrid comparison impossible after the fact. It was
caught and fixed. This audit is the systematic version of that catch.

Nothing here reads the test split's labels. The one live model run below is on
**validation**, on CPU, over 300 records.

---

## 1. Prediction/record join — VERIFIED BY MEASUREMENT

`error_analysis.py` and `sensitivity_valid_test_overlap.py` both join saved
predictions to dataset records **by position**. Predictions come from a
`DataLoader(shuffle=False)` over `ChunkedFunctionDataset`; the analyses read
`data/processed_clean/primevul_<split>.jsonl`. Three links had to hold, and each
was measured rather than assumed:

| Link | test | valid |
|---|---|---|
| processed record count | 25,911 | 25,430 |
| chunked record count | 25,911 | 25,430 |
| `idx` sequence identical, element-by-element | yes | yes |
| `target` sequence identical, element-by-element | yes | yes |
| `len(ChunkedFunctionDataset)` matches file | yes | yes |
| dataset order == file order (8 probed positions incl. both ends) | yes | yes |

Positive rate: test 695/25,911 = **2.6823%**; valid 699/25,430 = **2.7487%**.

A misaligned join would not have raised — it would have produced a plausible
but wrong error analysis. This is the highest-value check in the document.

## 2. Script entrypoints — DEFECT FOUND AND FIXED

`scripts/error_analysis.py` and `scripts/sensitivity_valid_test_overlap.py`
both documented `python scripts/<name>.py ...` in their own docstrings, and both
died with `ModuleNotFoundError: No module named 'backend'`. Running a script by
path puts the script's directory on `sys.path`, not the repository root; they
had only ever worked with `PYTHONPATH` set.

Both are post-test consumers, so this would have surfaced immediately after the
irreversible step. Fixed with a `__file__`-anchored bootstrap, and covered by
`backend/tests/test_script_entrypoints.py`, which launches every script in
`scripts/` as a real subprocess with `PYTHONPATH` stripped — a plain import test
cannot catch this, because pytest already has the root on the path. The test was
confirmed to fail when the bootstrap is removed and pass when restored.

`-m` invocation of the four `backend.ml.evaluation` entrypoints was already fine.

## 3. One-shot script exercised end to end — ON VALIDATION, ON CPU

`test_model.py --policy --save-predictions` was run against a 300-record
validation slice on CPU (`--cpu`, so no contention with the live training run).
It completed: checkpoint load → policy load → frozen threshold 0.1100 →
calibration applied → metrics → held-out calibration report → JSON written.

This does **not** touch the evaluation ledger: `_record_test_evaluation` is
gated on `split == "test"`.

The slice's own metrics are not meaningful (12 positives, none above the
threshold) and are not reported anywhere. The point was the code path.

## 4. Consumer key contracts — ALL SATISFIED

Keys written: `probabilities`, `raw_probabilities`, `targets`, `threshold`,
`split`, `checkpoint`, `policy`, `calibration_applied`, `calibration_measured`,
`n_samples`, `n_positive`, `n_negative`, `metrics_at_0.5`,
`metrics_at_selected_threshold`, `validation_selected_threshold`,
`best_validation_f1`.

| Consumer | Requires | Status |
|---|---|---|
| `compare_baselines.py` | `probabilities`, `targets`, `threshold` | OK |
| `error_analysis.py` | `probabilities`, `targets`, `threshold` | OK |
| `sensitivity_valid_test_overlap.py` | `probabilities`, `targets`, `threshold` | OK |

All three arrays are the same length; `probabilities != raw_probabilities`,
confirming calibration is genuinely applied rather than silently skipped.

## 5. Calibration/threshold ordering — VERIFIED CORRECT

This was the subtlest hazard: a threshold selected on **raw** scores but applied
to **calibrated** scores would put the operating point somewhere nobody chose.

`select_threshold.py` calls `find_best_threshold` once on raw scores (for the
objective comparison table), then, *if calibration is adopted*, re-fits the final
calibrator and **re-derives the threshold on the calibrated scores**. That
re-derived value is what is frozen. `test_model.py` applies the same serialised
calibrator and then that threshold. The orders match.

The frozen policy corroborates it numerically: the selected threshold carries
Brier 0.02533 (calibrated) while `objective_comparison.f1` carries 0.03444
(raw, = `metrics_at_0.5`). Calibrated 0.1100 maps back through the stored
isotonic knots to raw ≈0.2018, matching the raw f1 pick of 0.20 — and both give
identical **tp = 228, fn = 471**, as a monotone map must.

## 6. Calibrator serialisation — ALREADY COVERED

`test_calibrator_serialisation_round_trips` is parametrised over `platt` and
`isotonic`, asserts `calibrator_from_dict(c.to_dict()).predict(x) == c.predict(x)`
and that the dict survives a JSON round trip. Lossy serialisation would have
given the test split a different calibrator than validation fitted.

## 7. Test split integrity

`artifacts/test_evaluations.jsonl` **does not exist**, so the test split has not
been evaluated. This is the check that the "evaluate once" claim rests on.

---

## Status

The chain from the one-shot evaluation to the final numbers is verified. Test
suite: **183 passing**.

One item is deliberately *not* pre-flighted: the frozen policy in
`data/checkpoints/decision_policy.json` is fitted to the **epoch-1** checkpoint.
If training produces a better one, the threshold and calibration must be
re-derived on it before the test evaluation, or the operating point is stale.
That is tracked as step 2 of §10 in `FINAL_EVALUATION.md`.
