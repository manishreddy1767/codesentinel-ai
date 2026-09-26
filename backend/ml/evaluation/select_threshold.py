"""
Stage N/O - FREEZE THE DECISION POLICY (threshold + calibration).

    python -m backend.ml.evaluation.select_threshold
    python -m backend.ml.evaluation.select_threshold --objective fbeta --beta 2
    python -m backend.ml.evaluation.select_threshold --min-precision 0.6 \
        --objective recall_at_precision

Runs the best checkpoint over the VALIDATION split and writes
``decision_policy.json``: the operating threshold, the objective that chose it,
and the calibration decision. ``test_model.py`` then reads that file, so by the
time the test split is touched there is nothing left to choose.

THE TEST SPLIT IS NEVER LOADED BY THIS MODULE.

Why 0.5 is not the answer
-------------------------
0.5 is the threshold at which the *odds* are even. It is only the right
decision boundary when the classes are balanced and the two error types cost
the same. Here neither holds: the split is heavily benign, and the model is
trained with a pos_weight that deliberately inflates positive scores, so the
sigmoid output is not a calibrated probability to begin with. Reading anything
into 0.5 specifically is a habit, not an argument.

Choosing the objective is a policy decision, not a maths one
------------------------------------------------------------
    f1                   Balanced. The right default when the costs are
                         genuinely unknown.
    fbeta (beta=2)       Recall weighted 4x precision. Appropriate when a
                         missed vulnerability is much worse than a false alarm,
                         e.g. a gate on security-critical code.
    recall_at_precision  Maximise recall subject to a precision floor. The
                         honest choice when reviewer attention is the binding
                         constraint: below some precision the tool gets muted,
                         and a muted tool has zero recall in practice.

There is no universally correct pick. This module reports all of them, selects
the one asked for, and records which was used so the final numbers can be read
in the right context.
"""

import argparse
import json
from pathlib import Path

from backend.ml import config
from backend.ml.evaluation.calibration import (
    calibration_report,
    cross_validated_calibration,
    decide,
    format_calibration,
)
from backend.ml.evaluation.evaluator import load_model_from_checkpoint, predict
from backend.ml.models.collate import make_collate_fn
from backend.ml.models.dataset import ChunkedFunctionDataset
from backend.ml.training.metrics import (
    compute_metrics,
    find_best_threshold,
)
from backend.ml.utils.gpu import resolve_device
from backend.ml.utils.seed import set_seed

OBJECTIVES = ("f1", "fbeta", "mcc", "recall_at_precision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.BEST_CHECKPOINT)
    parser.add_argument("--chunked-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write decision_policy.json (default: next to the checkpoint)",
    )
    parser.add_argument(
        "--objective", default=config.THRESHOLD_OBJECTIVE, choices=OBJECTIVES
    )
    parser.add_argument("--beta", type=float, default=config.THRESHOLD_BETA)
    parser.add_argument(
        "--min-precision", type=float, default=config.THRESHOLD_MIN_PRECISION
    )
    parser.add_argument(
        "--calibration",
        default="auto",
        choices=("auto", "platt", "isotonic", "none"),
        help="auto = evaluate both and adopt only if it genuinely helps",
    )
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("CODESENTINEL - DECISION POLICY SELECTION (VALIDATION ONLY)")
    print("=" * 74)

    set_seed(config.SEED)

    if not args.checkpoint.exists():
        print(f"\nERROR: checkpoint not found: {args.checkpoint}")
        raise SystemExit(1)

    device = resolve_device(prefer_cuda=not args.cpu)
    model, payload = load_model_from_checkpoint(args.checkpoint, device)

    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(f"  objective  : {args.objective}")

    # ---- validation predictions -----------------------------------------
    path = args.chunked_dir / "primevul_valid_chunked.jsonl"
    print(f"\nLoading VALIDATION split: {path}")

    dataset = ChunkedFunctionDataset(path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        payload.get("config", {}).get("model_name", config.MODEL_NAME)
    )
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    print(f"  functions: {len(dataset)}")
    print("\nRunning inference on validation...")

    probabilities, targets = predict(
        model, dataset, collate_fn, device, batch_size=args.batch_size
    )

    positives = int(targets.sum())
    print(f"  positives: {positives} / {targets.size} ({positives / targets.size:.2%})")

    # ---- threshold comparison across objectives --------------------------
    print("\n" + "=" * 74)
    print("THRESHOLD OBJECTIVES COMPARED (all on validation)")
    print("=" * 74)
    print(f"  {'objective':<22}{'threshold':>10}{'precision':>11}{'recall':>9}{'F1':>8}{'FP':>8}{'FN':>8}")

    comparison = {}

    for objective in OBJECTIVES:
        threshold, metrics, _ = find_best_threshold(
            probabilities,
            targets,
            objective=objective,
            beta=args.beta,
            min_precision=args.min_precision,
        )
        comparison[objective] = {
            "threshold": threshold,
            "metrics": metrics.to_dict(),
        }
        print(
            f"  {objective:<22}{threshold:>10.3f}{metrics.precision:>11.4f}"
            f"{metrics.recall:>9.4f}{metrics.f1:>8.4f}{metrics.fp:>8}{metrics.fn:>8}"
        )

    default_metrics = compute_metrics(probabilities, targets, config.DEFAULT_THRESHOLD)
    print(
        f"  {'(default 0.5)':<22}{0.5:>10.3f}{default_metrics.precision:>11.4f}"
        f"{default_metrics.recall:>9.4f}{default_metrics.f1:>8.4f}"
        f"{default_metrics.fp:>8}{default_metrics.fn:>8}"
    )

    selected_threshold = comparison[args.objective]["threshold"]
    selected_metrics = compute_metrics(probabilities, targets, selected_threshold)

    print("\n" + selected_metrics.format(
        f"SELECTED: {args.objective} @ threshold {selected_threshold:.4f}"
    ))

    # ---- calibration ------------------------------------------------------
    print("\n" + "=" * 74)
    print("CALIBRATION (fit and judged inside validation only)")
    print("=" * 74)

    calibration_decision = {"adopt": False, "method": None, "reason": "not evaluated"}
    before = calibration_report(probabilities, targets)

    print("\n" + format_calibration(before, "UNCALIBRATED (validation)"))

    if args.calibration == "none":
        calibration_decision["reason"] = "disabled with --calibration none"
        print("\n  Calibration disabled.")
    else:
        candidates = (
            ["platt", "isotonic"] if args.calibration == "auto" else [args.calibration]
        )
        evaluated = {}

        for method in candidates:
            try:
                _, cv_before, cv_after = cross_validated_calibration(
                    probabilities,
                    targets,
                    method=method,
                    folds=args.calibration_folds,
                    seed=config.SEED,
                )
            except ValueError as error:
                print(f"\n  {method}: skipped ({error})")
                continue

            verdict = decide(cv_before, cv_after)
            evaluated[method] = verdict

            print(
                f"\n  {method:<9} out-of-fold ECE {verdict['ece_before']:.5f}"
                f" -> {verdict['ece_after']:.5f}"
                f"   Brier {verdict['brier_before']:.6f} -> {verdict['brier_after']:.6f}"
            )
            print(f"            {'ADOPT' if verdict['adopt'] else 'REJECT'}: {verdict['reason']}")

        adopted = {m: v for m, v in evaluated.items() if v["adopt"]}

        if adopted:
            # Best genuine ECE reduction wins.
            method = min(adopted, key=lambda m: adopted[m]["ece_after"])
            calibration_decision = {
                "adopt": True,
                "method": method,
                **adopted[method],
            }
            print(f"\n  DECISION: adopt {method} calibration.")
        else:
            calibration_decision = {
                "adopt": False,
                "method": None,
                "reason": "no method cleared the improvement bar",
                "evaluated": evaluated,
            }
            print(
                "\n  DECISION: no calibration. The model is close enough to "
                "calibrated that fitting a corrector would add a moving part "
                "without buying probability quality."
            )

    # NOTE: when calibration is adopted, the threshold must be re-derived on the
    # calibrated scores, because the calibrator moves the probability scale.
    if calibration_decision.get("adopt"):
        from backend.ml.evaluation.calibration import fit_final_calibrator

        calibrator = fit_final_calibrator(
            probabilities, targets, method=calibration_decision["method"]
        )
        calibrated = calibrator.predict(probabilities)

        # Serialised into the policy so the final test run applies exactly this
        # calibrator, without needing validation predictions again.
        calibration_decision["parameters"] = calibrator.to_dict()

        selected_threshold, selected_metrics, _ = find_best_threshold(
            calibrated,
            targets,
            objective=args.objective,
            beta=args.beta,
            min_precision=args.min_precision,
        )

        print(
            f"\n  Threshold re-derived on calibrated scores: "
            f"{selected_threshold:.4f}"
        )
        print("\n" + selected_metrics.format("SELECTED (calibrated, validation)"))

    # ---- freeze ------------------------------------------------------------
    out_path = args.out or args.checkpoint.parent / "decision_policy.json"

    policy = {
        "checkpoint": str(args.checkpoint),
        "selected_on": "validation",
        "objective": args.objective,
        "beta": args.beta,
        "min_precision": args.min_precision,
        "threshold": float(selected_threshold),
        "calibration": calibration_decision,
        "validation_metrics_at_threshold": selected_metrics.to_dict(),
        "validation_metrics_at_0.5": default_metrics.to_dict(),
        "objective_comparison": comparison,
        "validation_positives": positives,
        "validation_size": int(targets.size),
        "frozen": True,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 74)
    print("DECISION POLICY FROZEN")
    print("=" * 74)
    print(f"  threshold   : {selected_threshold:.4f}  (objective: {args.objective})")
    print(
        f"  calibration : "
        f"{calibration_decision.get('method') or 'none'}"
    )
    print(f"  written to  : {out_path}")
    print("\n  The test split has not been read. Run the single final evaluation:")
    print(f"    python -m backend.ml.evaluation.test_model --policy {out_path}")


if __name__ == "__main__":
    main()
