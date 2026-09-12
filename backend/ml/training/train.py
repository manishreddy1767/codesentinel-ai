"""
Training entrypoint.

    python -m backend.ml.training.train
    python -m backend.ml.training.train --epochs 3 --batch-size 4
    python -m backend.ml.training.train --resume
    python -m backend.ml.training.train --limit-train 200 --limit-valid 100

The test split is never loaded here. Run backend.ml.evaluation.test_model
after training finishes.
"""

import argparse
from pathlib import Path

from backend.ml import config
from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier
from backend.ml.models.collate import make_collate_fn
from backend.ml.models.dataset import ChunkedFunctionDataset
from backend.ml.training.trainer import Trainer
from backend.ml.utils.gpu import resolve_device
from backend.ml.utils.seed import set_seed


class _Subset:
    """Tiny head-of-dataset view used by --limit-* for smoke runs."""

    def __init__(self, dataset, limit: int):
        self.dataset = dataset
        self.limit = min(limit, len(dataset))

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        return self.dataset[index]

    def class_counts(self):
        targets = self.dataset.targets[: self.limit]
        positives = sum(targets)
        return len(targets) - positives, positives

    def pos_weight(self, cap: float | None = None):
        negatives, positives = self.class_counts()

        if positives == 0:
            return 1.0

        return min(negatives / positives, cap or config.MAX_POS_WEIGHT)

    def total_chunks(self):
        return sum(self.dataset.chunk_counts[: self.limit])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--chunked-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    parser.add_argument("--model", default=config.MODEL_NAME)

    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument(
        "--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE
    )
    parser.add_argument(
        "--chunk-micro-batch", type=int, default=config.CHUNK_MICRO_BATCH
    )
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=config.WARMUP_RATIO)
    parser.add_argument(
        "--grad-accum", type=int, default=config.GRADIENT_ACCUMULATION_STEPS
    )
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--checkpoint-every", type=int, default=config.CHECKPOINT_EVERY
    )
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    parser.add_argument("--gpu-log-every", type=int, default=config.GPU_LOG_EVERY)
    parser.add_argument(
        "--high-loss-threshold", type=float, default=config.HIGH_LOSS_THRESHOLD
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)

    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Override the pos_weight derived from the training split",
    )

    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pt")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)

    return parser


def main() -> None:
    args = build_argparser().parse_args()

    print("=" * 70)
    print("CODESENTINEL - CODEBERT VULNERABILITY CLASSIFIER TRAINING")
    print("=" * 70)
    print(config.describe())

    set_seed(args.seed)

    device = resolve_device(prefer_cuda=not args.cpu)
    print(f"\nDevice: {device}")

    if device.type == "cpu":
        print(
            "  NOTE: running on CPU. Mixed precision is disabled and training\n"
            "  the full split will be impractically slow - use --limit-train\n"
            "  for a smoke run, or install a CUDA build of torch."
        )

    # ---- data ---------------------------------------------------------
    print("\nLoading chunked datasets...")

    train_dataset = ChunkedFunctionDataset(
        args.chunked_dir / "primevul_train_chunked.jsonl"
    )
    valid_dataset = ChunkedFunctionDataset(
        args.chunked_dir / "primevul_valid_chunked.jsonl"
    )

    print(f"  train: {len(train_dataset)} functions")
    print(f"  valid: {len(valid_dataset)} functions")

    if args.limit_train:
        train_dataset = _Subset(train_dataset, args.limit_train)
        print(f"  train limited to {len(train_dataset)} functions")

    if args.limit_valid:
        valid_dataset = _Subset(valid_dataset, args.limit_valid)
        print(f"  valid limited to {len(valid_dataset)} functions")

    # ---- tokenizer / collate -------------------------------------------
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        raise ValueError(f"{args.model} tokenizer has no pad token id")

    collate_fn = make_collate_fn(pad_token_id)

    # ---- model ----------------------------------------------------------
    print(f"\nBuilding model from {args.model}...")

    model = HierarchicalCodeBERTClassifier(
        model_name=args.model,
        gradient_checkpointing=(
            config.USE_GRADIENT_CHECKPOINTING and not args.no_grad_checkpointing
        ),
        chunk_micro_batch=args.chunk_micro_batch,
    )

    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {parameters:,} ({trainable:,} trainable)")

    # ---- trainer ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=collate_fn,
        device=device,
        pos_weight=args.pos_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum,
        use_amp=(config.USE_AMP and not args.no_amp),
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        gpu_log_every=args.gpu_log_every,
        patience=args.patience,
        seed=args.seed,
        high_loss_threshold=args.high_loss_threshold,
    )

    trainer.maybe_resume(resume=args.resume)

    result = trainer.fit()

    print("\nNext step - evaluate the held-out test split:")
    print("  python -m backend.ml.evaluation.test_model")

    return result


if __name__ == "__main__":
    main()
