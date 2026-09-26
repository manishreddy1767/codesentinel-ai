"""
Optional - NEGATIVE UNDERSAMPLING OF AN ALREADY-CHUNKED TRAINING FILE.

    python -m backend.ml.preprocessing.undersample_chunked --ratio 10
    python -m backend.ml.preprocessing.undersample_chunked --ratio 10 --dry-run

Keeps EVERY positive and a seeded random sample of negatives at the requested
negative:positive ratio. Writes a new file; the input is never modified.

TRAIN ONLY. Validation and test must keep the real class balance, or the
metrics stop describing the deployment distribution.

Why operate on the chunked file
-------------------------------
Chunking is a deterministic per-record transform, so undersampling before or
after it selects the same functions. Doing it afterwards reuses the tokenisation
that has already been paid for - re-chunking 61k functions costs minutes of GPU
-idle CPU time for no change in output.

Why undersample at all
----------------------
This is a **compute-budget** decision, not a modelling one, and it should be
declared as such. At 30:1 a full epoch over PrimeVul costs ~5.2 h on a 4 GB
laptop GPU, so a 3-epoch run is ~16 h. At 10:1 the same wall-clock buys three
passes over every positive instead of one. When positives are the scarce signal,
seeing all 5,550 of them three times is worth more than seeing 113,000 extra
negatives once.

What it costs, and how that is paid for
---------------------------------------
The training distribution no longer matches validation/test, so the model's
raw probabilities come out systematically too high. That is a **calibration**
error, not a ranking error: PR-AUC and ROC-AUC are unaffected because they
depend only on the ordering of scores. The shift is corrected downstream by
Stage N (threshold chosen on validation, which keeps the true base rate) and
Stage O (calibration fitted on validation). Both already exist, and both operate
on data with the real class balance.

`pos_weight` must be recomputed from the undersampled file - the trainer does
this automatically from whichever dataset it is handed.
"""

import argparse
import json
import random
from pathlib import Path

from backend.ml import config


def undersample(
    input_path: Path,
    output_path: Path,
    ratio: float,
    seed: int,
    dry_run: bool,
) -> dict:
    """Two passes: count and choose, then write. Never loads the file into RAM."""

    positives = []
    negatives = []

    with input_path.open("r", encoding="utf-8") as handle:
        for offset, line in enumerate(handle):
            stripped = line.strip()
            if stripped:
                target = json.loads(stripped).get("target", 0)
                (positives if int(target) == 1 else negatives).append(offset)

    wanted_negative = min(len(negatives), round(len(positives) * ratio))

    rng = random.Random(seed)
    chosen_negatives = set(rng.sample(negatives, wanted_negative))
    keep = set(positives) | chosen_negatives

    stats = {
        "input_records": len(positives) + len(negatives),
        "input_positive": len(positives),
        "input_negative": len(negatives),
        "kept_positive": len(positives),
        "kept_negative": wanted_negative,
        "kept_total": len(positives) + wanted_negative,
        "requested_ratio": ratio,
        "achieved_ratio": wanted_negative / max(1, len(positives)),
        "seed": seed,
    }

    if dry_run:
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for index, line in enumerate(source):
            if line.strip() and index in keep:
                destination.write(line if line.endswith("\n") else line + "\n")
                written += 1

    stats["written"] = written

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.CHUNKED_DIR / "primevul_train_chunked.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.CHUNKED_DIR / "primevul_train_chunked_under10.jsonl",
    )
    parser.add_argument("--ratio", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("CODESENTINEL - NEGATIVE UNDERSAMPLING (TRAIN ONLY)")
    print("=" * 74)
    print(f"  in    : {args.input}")
    print(f"  out   : {args.output}{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  ratio : {args.ratio}:1 negative:positive")
    print(f"  seed  : {args.seed}")

    if not args.input.exists():
        print(f"\nERROR: {args.input} not found.")
        raise SystemExit(1)

    if not args.dry_run and args.output.resolve() == args.input.resolve():
        print("\nERROR: refusing to overwrite the input file.")
        raise SystemExit(2)

    if "train" not in args.input.name:
        print(
            f"\nERROR: {args.input.name} does not look like a training file. "
            f"Undersampling validation or test would invalidate every metric."
        )
        raise SystemExit(2)

    stats = undersample(
        args.input, args.output, args.ratio, args.seed, args.dry_run
    )

    print("\n" + "-" * 74)
    print(f"  input      : {stats['input_records']:>8} "
          f"({stats['input_positive']} pos / {stats['input_negative']} neg, "
          f"{stats['input_negative'] / max(1, stats['input_positive']):.1f}:1)")
    print(f"  kept       : {stats['kept_total']:>8} "
          f"({stats['kept_positive']} pos / {stats['kept_negative']} neg, "
          f"{stats['achieved_ratio']:.1f}:1)")
    print(f"  dropped    : {stats['input_records'] - stats['kept_total']:>8} negatives")
    print(f"  ALL {stats['kept_positive']} positives retained")
    print(f"\n  implied pos_weight for the new file: {stats['achieved_ratio']:.4f}")

    if args.dry_run:
        print("\nDRY RUN: nothing written.")
        return

    print(f"  written    : {stats['written']} records -> {args.output}")
    print(
        "\n  Remember: this file is for TRAINING only. Validation and test keep "
        "the real\n  class balance, so threshold selection and calibration still "
        "see the true base rate."
    )


if __name__ == "__main__":
    main()
