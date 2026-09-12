"""
Character-length profile of the processed splits.

    python -m backend.ml.preprocessing.analyze_code_lengths

Diagnostic only. Useful for sanity-checking chunking settings before paying
for a tokenizer pass (see analyze_token_lengths.py for the token-level view).
"""

import argparse
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import read_jsonl


def percentile(sorted_values, fraction: float):
    if not sorted_values:
        return 0

    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def analyze_file(name: str, path: Path) -> None:
    print("\n" + "=" * 60)
    print(f"{name.upper()} SPLIT")
    print("=" * 60)

    if not path.exists():
        print(f"MISSING: {path}")
        return

    lengths = [len(record.get("func", "")) for _, record in read_jsonl(path)]

    if not lengths:
        print("No records.")
        return

    lengths.sort()

    print(f"Samples            : {len(lengths)}")
    print(f"Minimum characters : {lengths[0]}")
    print(f"Maximum characters : {lengths[-1]}")
    print(f"Average characters : {sum(lengths) / len(lengths):.2f}")

    print("\nCharacter length percentiles:")
    for fraction in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(fraction * 100):<3}: {percentile(lengths, fraction)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.PROCESSED_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("CODE FUNCTION LENGTH ANALYSIS")
    print("=" * 60)

    for split in config.SPLITS:
        analyze_file(split, args.dir / f"primevul_{split}.jsonl")


if __name__ == "__main__":
    main()
