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
    print(f"    ROC-AUC   : {tuned.roc_auc:.4f}")
    print(f"    accuracy  : {tuned.accuracy:.4f}  (least informative here)")

    if args.save_report:
        report = {
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "validation_selected_threshold": threshold,
            "best_validation_f1": saved_f1,
            "metrics_at_0.5": baseline.to_dict(),
            "metrics_at_selected_threshold": tuned.to_dict(),
        }

        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  report written to {args.save_report}")


if __name__ == "__main__":
    main()
