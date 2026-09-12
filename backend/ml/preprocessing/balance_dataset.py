"""
Stage 4 - CLASS DISTRIBUTION ANALYSIS (and OPTIONAL undersampling).

By default this script only *reports*: it measures the class distribution of
each split and prints the pos_weight the trainer will derive from it. That is
the primary imbalance strategy for CodeSentinel (see the ML README).

    python -m backend.ml.preprocessing.balance_dataset

Undersampling the majority class is available as an explicit opt-in, purely as
a compute-budget tool for machines that cannot afford a full epoch over the
whole training split:

    python -m backend.ml.preprocessing.balance_dataset --undersample --ratio 5

Hard rules enforced here:
  * only the TRAIN split is ever resampled
  * validation and test are read for reporting and never rewritten
  * vulnerable samples are never duplicated (no oversampling), so the model
    cannot memorise repeated positives
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import read_jsonl, write_jsonl


def load_by_class(path: Path):
    """Split a processed JSONL file into (vulnerable, benign) record lists."""

    vulnerable = []
    benign = []

    for _, record in read_jsonl(path):
        if int(record["target"]) == 1:
            vulnerable.append(record)
        else:
            benign.append(record)

    return vulnerable, benign


def report_distribution(split: str, path: Path):
    if not path.exists():
        print(f"  {split:<6} MISSING ({path})")
        return None

    counts = Counter()

    for _, record in read_jsonl(path):
        counts[int(record["target"])] += 1

    total = sum(counts.values())
    positives = counts[1]
    negatives = counts[0]

    if total == 0:
        print(f"  {split:<6} EMPTY")
        return None

    positive_share = positives / total * 100

    line = (
        f"  {split:<6} total={total:>8}"
        f"  vulnerable={positives:>7} ({positive_share:5.2f}%)"
        f"  benign={negatives:>8}"
    )

    if positives:
        line += f"  ratio={negatives / positives:6.2f}:1"

    print(line)

    return {"total": total, "positives": positives, "negatives": negatives}


def compute_pos_weight(positives: int, negatives: int) -> float:
    """
    pos_weight for BCEWithLogitsLoss.

    BCEWithLogitsLoss multiplies the positive term of the loss by pos_weight,
    so setting it to n_negative / n_positive makes the two classes contribute
    equal total gradient mass without touching the data itself.
    """

    if positives <= 0:
        return 1.0

    return min(negatives / positives, config.MAX_POS_WEIGHT)


def undersample(
    input_path: Path,
    output_path: Path,
    ratio: float,
    seed: int,
):
    rng = random.Random(seed)

    vulnerable, benign = load_by_class(input_path)

    print(f"\n  source vulnerable: {len(vulnerable)}")
    print(f"  source benign    : {len(benign)}")

    wanted_benign = int(len(vulnerable) * ratio)
    wanted_benign = min(wanted_benign, len(benign))

    selected_benign = rng.sample(benign, wanted_benign)

    records = vulnerable + selected_benign
    rng.shuffle(records)

    written = write_jsonl(output_path, records)

    print(f"\n  kept vulnerable  : {len(vulnerable)} (all, never duplicated)")
    print(f"  kept benign      : {len(selected_benign)}")
    print(f"  written          : {written} -> {output_path}")

    return {
        "positives": len(vulnerable),
        "negatives": len(selected_benign),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument(
        "--undersample",
        action="store_true",
        help="Also write an undersampled copy of the training split",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=config.BENIGN_TO_VULNERABLE_RATIO,
        help="Benign-per-vulnerable ratio when undersampling",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    print("=" * 70)
    print("CODESENTINEL - CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print("\nClass distribution per split:")

    distributions = {}
    for split in config.SPLITS:
        distributions[split] = report_distribution(
            split, args.dir / f"primevul_{split}.jsonl"
        )

    train = distributions.get("train")

    if train:
        pos_weight = compute_pos_weight(train["positives"], train["negatives"])

        print("\n" + "-" * 70)
        print("PRIMARY IMBALANCE STRATEGY: BCEWithLogitsLoss(pos_weight=...)")
        print("-" * 70)
        print(f"  pos_weight from full training split = {pos_weight:.4f}")
        print(f"  (capped at {config.MAX_POS_WEIGHT})")
        print(
            "\n  The trainer recomputes this from whichever training file it is\n"
            "  actually given, so it stays correct if you undersample below."
        )

    print("\n  Validation and test distributions above are REPORTED ONLY.")
    print("  This script never rewrites them.")

    if args.undersample:
        print("\n" + "=" * 70)
        print(f"OPTIONAL UNDERSAMPLING (train only, ratio {args.ratio}:1)")
        print("=" * 70)

        input_path = args.dir / "primevul_train.jsonl"

        if not input_path.exists():
            print(f"ERROR: {input_path} not found. Run preprocess_primevul first.")
            raise SystemExit(1)

        result = undersample(
            input_path,
            args.dir / "primevul_train_balanced.jsonl",
            args.ratio,
            args.seed,
        )

        new_weight = compute_pos_weight(result["positives"], result["negatives"])
        print(f"\n  pos_weight for the undersampled file = {new_weight:.4f}")
        print(
            "\n  Note: pos_weight is derived from the distribution actually being\n"
            "  trained on. That is one coherent reweighting of the real prior,\n"
            "  not two stacked heuristics."
        )
        print(
            "\n  To train on it:\n"
            "    python -m backend.ml.preprocessing.chunk_dataset --use-balanced-train"
        )


if __name__ == "__main__":
    main()
