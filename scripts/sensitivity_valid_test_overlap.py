"""
Sensitivity analysis for the residual validation/test overlap.

    python scripts/sensitivity_valid_test_overlap.py \
        --ml artifacts/reports/test_report.json

14 functions (7 byte-identical, 14 whitespace-identical) appear in both the
validation and test splits after cleaning - 0.054% of test. They were NOT
removed, because deleting records from a held-out split changes what the model
is measured against, which is a worse error than the coupling it fixes.

This is not train-on-test leakage: the model never trained on those functions.
The concern is narrower - the threshold and calibration were fitted on
validation, which contains 14 functions that also sit in test, so the operating
point is very slightly tuned toward them.

This script quantifies that by recomputing the test metrics with those records
excluded. It reads the per-sample predictions saved by
`test_model.py --save-predictions`, so it does **not** re-run the model and
does **not** consume the one-shot test budget.

Order assumption (checked, not assumed silently): predictions are saved in the
order the chunked test file is read, which is the order `processed_clean/
primevul_test.jsonl` was written. The script asserts the record counts match
before joining.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.ml.training.metrics import compute_metrics
from backend.ml.utils.io import read_jsonl


def normalized_hash(code: str) -> str:
    """Whitespace-collapsed SHA-256, identical to audit_splits."""

    return hashlib.sha256(
        " ".join(code.split()).encode("utf-8", errors="replace")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", type=Path, required=True)
    parser.add_argument("--dir", type=Path, default=Path("data/processed_clean"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.ml.read_text(encoding="utf-8"))

    if "probabilities" not in report:
        raise SystemExit(
            "ERROR: the report has no stored predictions. Re-run test_model.py "
            "with --save-predictions."
        )

    probabilities = np.asarray(report["probabilities"], dtype=np.float64)
    targets = np.asarray(report["targets"], dtype=np.int64)
    threshold = float(report["threshold"])

    # ---- build the validation hash set ----------------------------------
    valid_hashes = {
        normalized_hash(record["func"])
        for _, record in read_jsonl(args.dir / "primevul_valid.jsonl", skip_invalid=True)
        if isinstance(record.get("func"), str)
    }

    # ---- walk test in file order, flag overlaps --------------------------
    test_hashes = [
        normalized_hash(record["func"])
        for _, record in read_jsonl(args.dir / "primevul_test.jsonl", skip_invalid=True)
    ]

    if len(test_hashes) != probabilities.size:
        raise SystemExit(
            f"ERROR: {len(test_hashes)} test records but {probabilities.size} "
            f"predictions. The join would be misaligned - aborting rather than "
            f"reporting a wrong number."
        )

    overlap_mask = np.array([h in valid_hashes for h in test_hashes], dtype=bool)
    n_overlap = int(overlap_mask.sum())

    print("=" * 74)
    print("SENSITIVITY: TEST METRICS EXCLUDING VALIDATION-OVERLAPPING RECORDS")
    print("=" * 74)
    print(f"  test records          : {probabilities.size:,}")
    print(f"  overlapping with valid: {n_overlap} ({n_overlap / probabilities.size:.4%})")
    print(f"  of which positive     : {int(targets[overlap_mask].sum())}")
    print(f"  frozen threshold      : {threshold:.4f}")

    full = compute_metrics(probabilities, targets, threshold)
    kept = compute_metrics(
        probabilities[~overlap_mask], targets[~overlap_mask], threshold
    )

    print()
    print(f"  {'metric':<12}{'full test':>12}{'excluded':>12}{'delta':>10}")
    print("  " + "-" * 46)

    rows = []
    for name in ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "brier"):
        a, b = getattr(full, name), getattr(kept, name)
        delta = b - a
        rows.append((name, a, b, delta))
        print(f"  {name:<12}{a:>12.4f}{b:>12.4f}{delta:>+10.4f}")

    print()
    print(f"  support: {full.support_positive} pos / {full.support_negative} neg"
          f"  ->  {kept.support_positive} pos / {kept.support_negative} neg")

    largest = max(rows, key=lambda r: abs(r[3]))
    print()
    print("=" * 74)
    print(
        f"  Largest change: {largest[0]} {largest[3]:+.4f}\n"
        f"  Interpretation: if every delta is small relative to the metric, the\n"
        f"  residual overlap does not materially affect the reported result.\n"
        f"  This is a robustness check, not a correction - the headline number\n"
        f"  remains the full-test-split one."
    )

    payload = {
        "n_test": int(probabilities.size),
        "n_overlap": n_overlap,
        "overlap_fraction": float(n_overlap / probabilities.size),
        "threshold": threshold,
        "metrics_full_test": full.to_dict(),
        "metrics_excluding_overlap": kept.to_dict(),
        "deltas": {name: float(d) for name, _, _, d in rows},
    }

    out = args.out or Path("artifacts/reports/sensitivity_valid_test_overlap.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
