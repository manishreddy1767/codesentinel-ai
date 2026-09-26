"""
FINAL TEST EVALUATION.

    python -m backend.ml.evaluation.test_model
    python -m backend.ml.evaluation.test_model --split valid

Loads the best checkpoint, reads the decision threshold that was selected on
VALIDATION data during training, and evaluates the untouched test split once.

The threshold is never re-tuned here. Re-optimising it on test predictions
would leak the test set into model selection and turn the reported F1 into an
optimistic upper bound rather than a held-out estimate. A --tune-on-test flag
is deliberately not provided.
"""

import argparse
import json
from pathlib import Path

from backend.ml import config
from backend.ml.evaluation.evaluator import evaluate_at_threshold, predict
from backend.ml.models.collate import make_collate_fn
from backend.ml.models.dataset import ChunkedFunctionDataset
from backend.ml.utils.gpu import memory_report, resolve_device
from backend.ml.utils.seed import set_seed


def _record_test_evaluation(args, metrics) -> None:
    """
    Append this test evaluation to a durable ledger and flag repeats.

    A held-out estimate is only unbiased if the split was consulted once, after
    all selection was finished. Repeated evaluation of *different* checkpoints
    is selection-on-test by another name, and it is invisible unless something
    writes it down.
    """

    import hashlib
    from datetime import datetime, timezone

    ledger_path = config.ARTIFACTS_DIR / "test_evaluations.jsonl"

    try:
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16]
    except OSError:
        digest = "unknown"

    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_16": digest,
        "threshold": metrics.threshold,
        "f1": metrics.f1,
        "pr_auc": metrics.pr_auc,
        "recall": metrics.recall,
        "precision": metrics.precision,
    }

    previous = []

    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    previous.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError as error:
        print(f"\n  WARNING: could not write the test-evaluation ledger: {error}")
        return

    distinct = {record.get("checkpoint_sha256_16") for record in previous}
    distinct.discard("unknown")

    if previous:
        print(f"\n  Test-evaluation ledger: {ledger_path}")
        print(f"    this is evaluation #{len(previous) + 1} of the test split")

        if distinct - {digest}:
            print(
                "\n  WARNING: the test split has now been evaluated with more than\n"
                "  one checkpoint. Whichever result you report was, to some degree,\n"
                "  chosen with knowledge of the test set. Treat it as an optimistic\n"
                "  estimate and say so in the write-up."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=config.BEST_CHECKPOINT)
    parser.add_argument("--chunked-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument(
        "--split",
        default="test",
        choices=list(config.SPLITS),
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help=(
            "decision_policy.json frozen by select_threshold.py. Supplies the "
            "threshold and any adopted calibration."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the checkpoint threshold (for diagnostics only)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.EVAL_BATCH_SIZE
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Write metrics as JSON to this path",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Include per-sample probabilities and targets in the saved report, "
            "so Stage 18 hybrids can be computed without a second evaluation."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CODESENTINEL - FINAL TEST EVALUATION")
    print("=" * 70)

    set_seed(config.SEED)

    if not args.checkpoint.exists():
        print(f"\nERROR: checkpoint not found: {args.checkpoint}")
        print("Train first: python -m backend.ml.training.train")
        raise SystemExit(1)

    device = resolve_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    # ---- model ---------------------------------------------------------
    print(f"\nLoading best model: {args.checkpoint}")

    from backend.ml.evaluation.evaluator import load_model_from_checkpoint

    model, payload = load_model_from_checkpoint(args.checkpoint, device)

    saved_threshold = payload.get("best_threshold", config.DEFAULT_THRESHOLD)
    saved_f1 = payload.get("best_f1", float("nan"))

    print(f"  trained to epoch      : {payload.get('epoch')}")
    print(f"  best validation F1    : {saved_f1:.4f}")
    print(f"  validation threshold  : {saved_threshold:.4f}")
    print(f"  architecture          : {payload.get('config', {})}")

    # ---- decision policy -------------------------------------------------
    # Precedence: explicit CLI override (diagnostic only) > frozen policy file
    # > the threshold the trainer selected on validation.
    policy = {}

    if args.policy:
        if not args.policy.exists():
            print(f"\nERROR: policy file not found: {args.policy}")
            raise SystemExit(1)

        policy = json.loads(args.policy.read_text(encoding="utf-8"))

        if policy.get("selected_on") != "validation":
            print(
                f"\nERROR: {args.policy} was not selected on validation "
                f"(selected_on={policy.get('selected_on')!r}). Refusing to use it."
            )
            raise SystemExit(1)

        saved_threshold = policy["threshold"]
        print(f"\n  Frozen policy: {args.policy}")
        print(f"    objective   : {policy.get('objective')}")
        print(f"    threshold   : {saved_threshold:.4f}")
        print(f"    calibration : {policy.get('calibration', {}).get('method') or 'none'}")

    threshold = args.threshold if args.threshold is not None else saved_threshold

    if args.threshold is not None:
        print(
            f"\n  WARNING: threshold overridden to {threshold:.4f} on the command\n"
            f"  line. This is a diagnostic, not a valid held-out result."
        )

    # ---- data -----------------------------------------------------------
    path = args.chunked_dir / f"primevul_{args.split}_chunked.jsonl"
    print(f"\nLoading {args.split} split: {path}")

    dataset = ChunkedFunctionDataset(path)
    negatives, positives = (
        len(dataset) - sum(dataset.targets),
        sum(dataset.targets),
    )
    print(f"  functions  : {len(dataset)}")
    print(f"  vulnerable : {positives}")
    print(f"  benign     : {negatives}")
    print("  (untouched: no balancing, no deduplication, no augmentation)")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        payload.get("config", {}).get("model_name", config.MODEL_NAME)
    )
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    # ---- inference --------------------------------------------------------
    print("\nRunning inference...")
    probabilities, targets = predict(
        model,
        dataset,
        collate_fn,
        device,
        batch_size=args.batch_size,
    )
    print(f"  predictions: {probabilities.shape}  targets: {targets.shape}")
    print("  " + memory_report())

    # ---- frozen calibration ------------------------------------------------
    raw_probabilities = probabilities
    calibration = policy.get("calibration") or {}

    if calibration.get("adopt") and calibration.get("parameters"):
        from backend.ml.evaluation.calibration import calibrator_from_dict

        calibrator = calibrator_from_dict(calibration["parameters"])
        probabilities = calibrator.predict(probabilities)

        print(
            f"\n  Applied frozen {calibration['method']} calibration "
            f"(fitted on validation, not on this split)."
        )

    # ---- metrics -----------------------------------------------------------
    baseline = evaluate_at_threshold(
        probabilities,
        targets,
        config.DEFAULT_THRESHOLD,
        f"{args.split.upper()} @ default threshold 0.5",
    )

    tuned = evaluate_at_threshold(
        probabilities,
        targets,
        threshold,
        f"{args.split.upper()} @ validation-selected threshold {threshold:.3f}",
    )

    print("\n" + "=" * 70)
    print("HEADLINE RESULT")
    print("=" * 70)
    print(
        f"  Reportable {args.split} metrics use the validation-selected\n"
        f"  threshold {threshold:.4f}:"
    )
    print(f"    precision : {tuned.precision:.4f}")
    print(f"    recall    : {tuned.recall:.4f}")
    print(f"    F1        : {tuned.f1:.4f}")
    print(f"    MCC       : {tuned.mcc:.4f}")
    print(f"    PR-AUC    : {tuned.pr_auc:.4f}  (baseline {tuned.positive_rate:.4f})")
    print(f"    ROC-AUC   : {tuned.roc_auc:.4f}  (optimistic under imbalance)")
    print(f"    Brier     : {tuned.brier:.4f}")
    print(f"    accuracy  : {tuned.accuracy:.4f}  (least informative here)")

    # The operational reading matters more than the headline float.
    print(
        f"\n  In operational terms, on {tuned.support_positive + tuned.support_negative} "
        f"functions:"
    )
    print(
        f"    {tuned.tp} of {tuned.support_positive} vulnerabilities found, "
        f"{tuned.fn} missed"
    )
    print(
        f"    {tuned.fp} false alarms raised against {tuned.support_negative} "
        f"benign functions"
    )

    if tuned.predicted_positive:
        print(
            f"    a reviewer working the alerts sees a true vulnerability "
            f"{tuned.precision:.1%} of the time"
        )

    # ---- calibration on the held-out split ---------------------------------
    from backend.ml.evaluation.calibration import calibration_report, format_calibration

    held_out_calibration = calibration_report(probabilities, targets)
    print("\n" + format_calibration(
        held_out_calibration, f"{args.split.upper()} CALIBRATION (measured, not fitted)"
    ))

    # ---- evaluation ledger --------------------------------------------------
    # The test split is meant to be evaluated once, after every choice is
    # frozen. Nothing can technically stop a second run, but an append-only
    # ledger makes it visible rather than invisible - if this file lists several
    # different checkpoints, the "held-out" number has been selected on.
    if args.split == "test":
        _record_test_evaluation(args, tuned)

    if args.save_report:
        report = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "validation_selected_threshold": threshold,
            "threshold": threshold,
            "best_validation_f1": saved_f1,
            "policy": str(args.policy) if args.policy else None,
            "calibration_applied": bool(calibration.get("adopt")),
            "n_samples": int(targets.size),
            "n_positive": int(targets.sum()),
            "n_negative": int((targets == 0).sum()),
            "metrics_at_0.5": baseline.to_dict(),
            "metrics_at_selected_threshold": tuned.to_dict(),
            "calibration_measured": held_out_calibration,
        }

        if args.save_predictions:
            # Stored so Stage 18 hybrids can be built arithmetically instead of
            # running the model over the test split a second time, which would
            # break the evaluate-once guarantee.
            report["probabilities"] = [float(p) for p in probabilities]
            report["raw_probabilities"] = [float(p) for p in raw_probabilities]
            report["targets"] = [int(t) for t in targets]

        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  report written to {args.save_report}")


if __name__ == "__main__":
    main()
