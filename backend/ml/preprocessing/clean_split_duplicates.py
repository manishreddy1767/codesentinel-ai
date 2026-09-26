"""
Stage C (remediation) - REMOVE LEAKED RECORDS FROM THE TRAINING SPLIT.

    python -m backend.ml.preprocessing.clean_split_duplicates \
        --in-dir data/processed --out-dir data/processed_clean

    # preview only, writes nothing
    python -m backend.ml.preprocessing.clean_split_duplicates --dry-run

Only the TRAINING split is ever filtered.

Why train-only
--------------
Leakage is fixed by removing the offending records from *training*, not from
evaluation. Dropping them from validation or test would silently change what
the model is measured against, make the numbers incomparable to published
PrimeVul results, and bias the held-out sets towards whatever is easy. So
valid and test are copied through byte-for-byte, and every removal happens on
the side that is allowed to shrink.

Raw data is never touched: this reads ``--in-dir`` (processed) and writes to a
separate ``--out-dir``. Writing the output on top of the input is refused.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.ml import config
from backend.ml.preprocessing.audit_splits import (
    DIMENSIONS,
    _exact_hash,
    _normalized_hash,
)
from backend.ml.utils.io import read_jsonl

# Dimensions that justify dropping a training record. Deliberately excludes
# "project": sharing a repository is not leakage, and filtering on it would
# delete most of the training data for no defensible reason.
DEFAULT_DIMENSIONS = ("exact_code", "normalized_code", "hash", "idx", "big_vul_idx")


def _record_keys(record: dict, dimensions) -> dict:
    """The key this record occupies in each requested dimension."""

    keys = {}
    code = record.get("func")

    if isinstance(code, str) and code.strip():
        if "exact_code" in dimensions:
            keys["exact_code"] = _exact_hash(code)
        if "normalized_code" in dimensions:
            keys["normalized_code"] = _normalized_hash(code)

    for name, kind, _ in DIMENSIONS:
        if kind == "code" or name not in dimensions:
            continue

        value = record.get(name)

        if value is None or value == "":
            continue

        keys[name] = str(value)

    return keys


def build_holdout_index(paths, dimensions) -> dict:
    """Every key occupied by any record in the held-out splits."""

    index = defaultdict(set)

    for path in paths:
        if not path.exists():
            print(f"  WARNING: {path} not found; its keys cannot be excluded.")
            continue

        count = 0

        for _, record in read_jsonl(path, skip_invalid=True):
            count += 1
            for dimension, key in _record_keys(record, dimensions).items():
                index[dimension].add(key)

        print(f"  indexed {count:>8} records from {path.name}")

    return index


def clean_train(
    train_path: Path,
    output_path: Path,
    holdout_index: dict,
    dimensions,
    dry_run: bool,
) -> dict:
    """Filter the training split against the held-out index."""

    stats = Counter()
    by_dimension = Counter()

    handle = None

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8")

    try:
        for _, record in read_jsonl(train_path, skip_invalid=True):
            stats["read"] += 1

            keys = _record_keys(record, dimensions)

            hits = [
                dimension
                for dimension, key in keys.items()
                if key in holdout_index.get(dimension, ())
            ]

            if hits:
                stats["removed"] += 1
                # Counted per dimension, so the report shows which check earned
                # its keep rather than just a total.
                for dimension in hits:
                    by_dimension[dimension] += 1
                continue

            stats["kept"] += 1

            if handle is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if handle is not None:
            handle.close()

    return {"stats": stats, "by_dimension": by_dimension}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.DATA_DIR / "processed_clean",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help=f"Overlap dimensions that justify removal (default: {' '.join(DEFAULT_DIMENSIONS)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without writing anything",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("CODESENTINEL - TRAIN SPLIT LEAKAGE CLEANING")
    print("=" * 74)
    print(f"  in  : {args.in_dir}")
    print(f"  out : {args.out_dir}{'  (DRY RUN - nothing written)' if args.dry_run else ''}")
    print(f"  dims: {', '.join(args.dimensions)}")

    if not args.dry_run and args.out_dir.resolve() == args.in_dir.resolve():
        print(
            "\nERROR: --out-dir must differ from --in-dir. Refusing to overwrite "
            "the input splits in place."
        )
        raise SystemExit(2)

    train_path = args.in_dir / "primevul_train.jsonl"

    if not train_path.exists():
        print(f"\nERROR: {train_path} not found.")
        raise SystemExit(2)

    print("\nIndexing held-out splits...")
    holdout_index = build_holdout_index(
        [args.in_dir / "primevul_valid.jsonl", args.in_dir / "primevul_test.jsonl"],
        set(args.dimensions),
    )

    print("\nFiltering training split...")
    result = clean_train(
        train_path,
        args.out_dir / "primevul_train.jsonl",
        holdout_index,
        set(args.dimensions),
        args.dry_run,
    )

    stats = result["stats"]

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  train records read   : {stats['read']}")
    print(f"  removed (leaked)     : {stats['removed']}")
    print(f"  kept                 : {stats['kept']}")

    if stats["read"]:
        print(f"  removal rate         : {stats['removed'] / stats['read']:.4%}")

    if result["by_dimension"]:
        print("\n  removals by dimension (a record can match several):")
        for dimension, count in result["by_dimension"].most_common():
            print(f"    {dimension:<18} {count}")

    if args.dry_run:
        print("\nDRY RUN: no files were written.")
        return

    # valid/test are copied unchanged so the cleaned directory is a complete,
    # self-consistent dataset rather than a train-only fragment.
    for split in ("valid", "test"):
        source = args.in_dir / f"primevul_{split}.jsonl"
        destination = args.out_dir / f"primevul_{split}.jsonl"

        if not source.exists():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        print(f"  copied {split} unchanged -> {destination}")

    print(f"\nCleaned splits written to {args.out_dir}")
    print("Re-run the audit against them to confirm:")
    print(
        f"  python -m backend.ml.preprocessing.audit_splits --dir {args.out_dir} --strict"
    )


if __name__ == "__main__":
    main()
