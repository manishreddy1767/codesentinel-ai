"""
Stage 18 (part 2) - ML, HYBRID AND ABLATION COMPARISON.

    python -m backend.ml.evaluation.compare_baselines \
        --ml artifacts/reports/test_report.json \
        --baselines artifacts/reports/baselines_test.json

Combines the frozen model's test predictions (written by `test_model.py
--save-predictions`) with the rule/taint baseline scores (written by
`baselines.py`) and reports every comparison side by side.

Why this is a separate module
-----------------------------
The test split must be *scored* exactly once. Running the model again to build
a hybrid would be a second evaluation, and would make the "evaluated once"
guarantee a fiction. So this reads the predictions that were already saved and
does all the combining arithmetically - no model, no GPU, no second pass.

Hybrids
-------
ml_or_rules    flag if EITHER fires. Maximises recall; the union of two
               detectors can only find more, and can only be less precise.
ml_and_rules   flag only if BOTH fire. The precision-first policy.
ml_plus_rules  a blended score, ML probability boosted where rules agree.
               Reported with PR-AUC, so it is judged on ranking rather than on
               one operating point.

Every comparison uses the SAME frozen threshold for the ML component that
Stage 15 selected on validation. Re-tuning per hybrid would be selection on
test.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.ml import config
from backend.ml.training.metrics import compute_metrics


def _load(path: Path, what: str) -> dict:
    if not path.exists():
        raise SystemExit(f"ERROR: {what} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--boost",
        type=float,
        default=0.15,
        help="Probability added (pre-clip) where the rules also fire, for ml_plus_rules",
    )
    args = parser.parse_args()

    ml = _load(args.ml, "ML test report")
    base = _load(args.baselines, "baseline report")

    if not ml.get("probabilities"):
        raise SystemExit(
            "ERROR: the ML report has no stored probabilities. Re-run "
            "test_model.py with --save-predictions so the hybrid can be built "
            "without evaluating the test split a second time."
        )

    ml_probabilities = np.asarray(ml["probabilities"], dtype=np.float64)
    targets = np.asarray(ml["targets"], dtype=np.int64)
    threshold = float(ml["threshold"])

    if base["functions"] != targets.size:
        raise SystemExit(
            f"ERROR: baseline covered {base['functions']} functions but the ML "
            f"report has {targets.size}. They must be the same split, unsampled, "
            f"in the same order. Re-run baselines.py without --limit."
        )

    rows = {}

    ml_metrics = compute_metrics(ml_probabilities, targets, threshold)
    rows["ml_only"] = ml_metrics

    ml_flag = ml_probabilities >= threshold

    for name, result in base["results"].items():
        scores = np.asarray(result["scores"], dtype=np.float64)
        rows[name] = compute_metrics(scores, targets, 1e-9)

        rule_flag = scores > 0

        union = np.where(ml_flag | rule_flag, 1.0, 0.0)
        intersection = np.where(ml_flag & rule_flag, 1.0, 0.0)
        blended = np.clip(ml_probabilities + args.boost * (scores > 0), 0.0, 1.0)

        rows[f"ml_or_{name}"] = compute_metrics(union, targets, 0.5)
        rows[f"ml_and_{name}"] = compute_metrics(intersection, targets, 0.5)
        rows[f"ml_plus_{name}"] = compute_metrics(blended, targets, threshold)

    positive_rate = float(targets.mean())

    print("=" * 96)
    print("STAGE 18 - COMPARISON ON THE TEST SPLIT")
    print("=" * 96)
    print(f"  functions={targets.size}  positives={int(targets.sum())} ({positive_rate:.2%})")
    print(f"  ML threshold (frozen on validation): {threshold:.4f}")
    print()
    print(f"  {'variant':<24}{'precision':>11}{'recall':>9}{'F1':>8}{'PR-AUC':>9}{'TP':>7}{'FP':>8}{'FN':>7}")
    print("  " + "-" * 88)

    for name, m in rows.items():
        print(
            f"  {name:<24}{m.precision:>11.4f}{m.recall:>9.4f}{m.f1:>8.4f}"
            f"{m.pr_auc:>9.4f}{m.tp:>7}{m.fp:>8}{m.fn:>7}"
        )

    print(f"\n  always-positive PR-AUC baseline: {positive_rate:.4f}")
    print(
        "\n  Note: PR-AUC for the binary union/intersection rows is computed over a\n"
        "  0/1 score and is therefore not a ranking measure - read F1 and the\n"
        "  confusion counts for those, and PR-AUC for the probability-valued rows."
    )

    print("\n  NOT IMPLEMENTED (reported, not estimated):")
    print("    CodeBERT + GATv2 : backend/ml/graphs/ is an empty package")

    payload = {
        "threshold": threshold,
        "functions": int(targets.size),
        "positives": int(targets.sum()),
        "positive_rate": positive_rate,
        "variants": {name: m.to_dict() for name, m in rows.items()},
        "not_implemented": ["codebert_gatv2"],
    }

    out_path = args.out or (config.REPORTS_DIR / "comparison_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
