"""
Stage J - SMALL OVERFIT TEST.

    python -m backend.ml.training.overfit_test --samples 32 --epochs 30
    python -m backend.ml.training.overfit_test --samples 64 --cpu

Trains on a tiny, fixed subset of the TRAINING split and evaluates on that same
subset. Deliberately measures memorisation, not generalisation.

Why this is worth a dedicated tool
----------------------------------
A model that cannot drive the loss towards zero on 32 examples it sees over and
over has a bug, not a data problem - a detached graph, a frozen encoder, an
optimizer stepping on the wrong parameters, labels misaligned with inputs, a
learning rate so small nothing moves. Every one of those failures also produces
a plausible-looking training curve on the full dataset, where a loss that drifts
down slowly is indistinguishable from a loss that is barely learning. On 32
samples the distinction is unambiguous.

This runs BEFORE any hyperparameter search, because searching over a broken
implementation just finds the configuration that hides the bug best.

Dropout is disabled and pos_weight forced to 1.0: the question is purely
"can the optimisation path fit this data", and regularisation only blurs it.
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from backend.ml import config
from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier
from backend.ml.models.collate import make_collate_fn
from backend.ml.models.dataset import ChunkedFunctionDataset
from backend.ml.training.metrics import compute_metrics
from backend.ml.training.train import _Subset
from backend.ml.utils.gpu import memory_report, resolve_device
from backend.ml.utils.seed import set_seed


class _BalancedSubset(_Subset):
    """
    A 50/50 subset of the training split, for the memorisation check only.

    Never use this for training or evaluation: it misrepresents the base rate
    by an order of magnitude. It exists because the overfit test asks a
    capacity question - can the optimiser drive this loss down - and on a 3%
    positive split a ratio-preserving sample of 64 contains two positives,
    which makes F1 a four-valued statistic and the result unreadable.
    """

    def __init__(self, dataset, limit: int, seed: int = 0):
        import random as _random

        self.dataset = dataset
        targets = list(dataset.targets)

        positives = [i for i, t in enumerate(targets) if t == 1]
        negatives = [i for i, t in enumerate(targets) if t != 1]

        rng = _random.Random(seed)
        rng.shuffle(positives)
        rng.shuffle(negatives)

        half = limit // 2
        chosen = positives[: min(half, len(positives))]
        chosen += negatives[: limit - len(chosen)]

        rng.shuffle(chosen)

        self.indices = chosen
        self.limit = len(chosen)
        self.targets = [targets[i] for i in chosen]
        self.chunk_counts = [dataset.chunk_counts[i] for i in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunked-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-micro-batch", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Higher than production on purpose: this test wants fast memorisation",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "Use a 50/50 subset instead of one matching the real class ratio. "
            "Recommended here: PrimeVul is ~3%% positive, so a ratio-preserving "
            "64-sample subset holds only ~2 vulnerable functions and F1 moves in "
            "steps of 0.25. This test asks whether the optimiser CAN fit the "
            "data, which is a question about capacity, not about class priors."
        ),
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.95,
        help=(
            "Training-set F1 counting as 'substantially overfit'. 0.95 rather "
            "than 1.0 because the run should not be called a failure over a "
            "single stubborn example; the loss-reduction check below is the "
            "real signal."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("CODESENTINEL - STAGE J: SMALL OVERFIT TEST")
    print("=" * 74)

    set_seed(args.seed)
    device = resolve_device(prefer_cuda=not args.cpu)

    path = args.chunked_dir / "primevul_train_chunked.jsonl"
    full = ChunkedFunctionDataset(path)

    if args.balanced:
        dataset = _BalancedSubset(full, args.samples, seed=args.seed)
    else:
        dataset = _Subset(full, args.samples, seed=args.seed)

    negatives, positives = dataset.class_counts()

    print(f"  device       : {device}")
    print(f"  source       : {path}")
    print(f"  subset       : {len(dataset)} functions "
          f"({positives} vulnerable / {negatives} benign, stratified)")
    print(f"  epochs       : {args.epochs}")
    print(f"  learning rate: {args.lr}")

    if positives == 0 or negatives == 0:
        print("\nFAIL: the subset is single-class; the test would be meaningless.")
        raise SystemExit(1)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    # Dropout off: this test measures capacity to fit, not regularised
    # generalisation, and dropout only adds variance to the curve.
    model = HierarchicalCodeBERTClassifier(
        model_name=args.model,
        dropout=0.0,
        gradient_checkpointing=False,
        chunk_micro_batch=args.chunk_micro_batch,
    ).to(device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # pos_weight=1: no imbalance correction, so the loss is a clean read on fit.
    criterion = nn.BCEWithLogitsLoss()

    print("\n  epoch    loss     grad_norm   train_f1   train_acc")
    print("  " + "-" * 52)

    first_loss = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        grad_norm = 0.0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = batch["targets"]

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                function_index=batch["function_index"],
                batch_size=targets.size(0),
            )

            loss = criterion(logits, targets)
            loss.backward()

            # Reported, never clipped: a vanishing gradient here is the single
            # most diagnostic symptom this test can surface.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = total_loss / max(1, batches)

        if first_loss is None:
            first_loss = average_loss

        # Evaluate on the training subset itself: memorisation is the point.
        model.eval()
        probabilities = []
        targets_all = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    function_index=batch["function_index"],
                    batch_size=batch["targets"].size(0),
                )
                probabilities.append(torch.sigmoid(logits.float()).cpu())
                targets_all.append(batch["targets"].cpu())

        probabilities = torch.cat(probabilities).numpy()
        targets_all = torch.cat(targets_all).numpy().astype(int)

        metrics = compute_metrics(probabilities, targets_all, 0.5)
        history.append(
            {"epoch": epoch + 1, "loss": average_loss, "f1": metrics.f1}
        )

        print(
            f"  {epoch + 1:>5}  {average_loss:>7.4f}  {grad_norm:>10.4f}"
            f"  {metrics.f1:>9.4f}  {metrics.accuracy:>10.4f}"
        )

        if not math.isfinite(average_loss):
            print("\nFAIL: loss became non-finite.")
            raise SystemExit(1)

        if metrics.f1 >= args.target_f1 and average_loss < 0.1:
            print(f"\n  Target reached at epoch {epoch + 1}.")
            break

    final = history[-1]
    loss_drop = (first_loss - final["loss"]) / first_loss if first_loss else 0.0

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  first-epoch loss : {first_loss:.4f}")
    print(f"  final loss       : {final['loss']:.4f}  ({loss_drop:.1%} reduction)")
    print(f"  final train F1   : {final['f1']:.4f}")
    print("  " + memory_report())

    passed = final["f1"] >= args.target_f1 and final["loss"] < first_loss * 0.5

    if passed:
        print(
            f"\n  PASS: the model memorised {len(dataset)} samples "
            f"(F1 {final['f1']:.3f}).\n"
            f"  Gradients flow, the optimizer updates weights, and labels line "
            f"up with inputs."
        )
    else:
        print(
            "\n  FAIL: the model could not fit a tiny subset. Do NOT proceed to "
            "tuning\n  or full training. Investigate, in this order:\n"
            "    1. are encoder gradients non-zero after backward()?\n"
            "    2. does the optimizer include the encoder parameters?\n"
            "    3. do targets line up with the functions they are attached to?\n"
            "    4. is the learning rate large enough to move anything?"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
