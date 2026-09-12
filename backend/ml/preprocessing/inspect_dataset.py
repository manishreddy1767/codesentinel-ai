"""
Stage 1 - RAW DATA INSPECTION.

Reports what is actually inside data/raw/ without assuming any schema:
field names, label field and its meaning, record counts, class distribution,
code-length statistics, missing values and malformed lines.

    python -m backend.ml.preprocessing.inspect_dataset

Read-only. Never writes to data/raw/.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import detect_field


def _percentile(sorted_values, fraction: float):
    if not sorted_values:
        return 0

    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def _preview(value, limit: int = 300) -> str:
    text = str(value).replace("\n", "\n")
    return text if len(text) <= limit else text[:limit] + " ..."


def inspect_file(name: str, path: Path, show_sample: bool = True) -> dict:
    print("\n" + "=" * 70)
    print(f"SPLIT: {name.upper()}   ({path})")
    print("=" * 70)

    if not path.exists():
        print("MISSING: file does not exist.")
        return {"exists": False, "path": str(path)}

    total = 0
    invalid_json = 0
    field_presence = Counter()
    labels = Counter()
    code_lengths = []
    empty_code = 0
    missing_code_field = 0
    missing_label_field = 0

    code_field = None
    label_field = None
    first_record = None

    # Count malformed lines explicitly rather than silently skipping them.
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            if first_record is None:
                first_record = record
                code_field = detect_field(record, config.CODE_FIELD_CANDIDATES)
                label_field = detect_field(record, config.LABEL_FIELD_CANDIDATES)

            for key in record:
                field_presence[key] += 1

            if code_field and code_field in record:
                code = record[code_field]

                if isinstance(code, str):
                    code_lengths.append(len(code))

                    if not code.strip():
                        empty_code += 1
                else:
                    missing_code_field += 1
            else:
                missing_code_field += 1

            if label_field and label_field in record:
                labels[record[label_field]] += 1
            else:
                missing_label_field += 1

    valid = total - invalid_json

    print(f"\nTotal lines            : {total}")
    print(f"Valid JSON records     : {valid}")
    print(f"Invalid JSON lines     : {invalid_json}")

    print(f"\nDetected code field    : {code_field}")
    print(f"Detected label field   : {label_field}")

    print("\nField presence (field: records containing it):")
    for key, count in field_presence.most_common():
        coverage = (count / valid * 100) if valid else 0.0
        print(f"  {key:<20} {count:>8}  ({coverage:5.1f}%)")

    print("\nLabel distribution:")
    if labels:
        for value, count in sorted(labels.items(), key=lambda item: str(item[0])):
            share = (count / valid * 100) if valid else 0.0
            print(f"  label={value!r:<8} {count:>8}  ({share:5.2f}%)")

        positives = labels.get(1, 0) + labels.get("1", 0)
        negatives = labels.get(0, 0) + labels.get("0", 0)

        if positives:
            print(f"\n  negative:positive ratio = {negatives / positives:.2f} : 1")
            print(f"  suggested pos_weight    = {negatives / positives:.2f}")
        else:
            print("\n  WARNING: no positive (vulnerable) samples detected.")
    else:
        print("  none found")

    print("\nData quality:")
    print(f"  records with empty/blank code : {empty_code}")
    print(f"  records missing a code field  : {missing_code_field}")
    print(f"  records missing a label field : {missing_label_field}")

    if code_lengths:
        ordered = sorted(code_lengths)
        print("\nCode length in characters:")
        print(f"  min     : {ordered[0]}")
        print(f"  max     : {ordered[-1]}")
        print(f"  mean    : {sum(ordered) / len(ordered):.1f}")
        for fraction in (0.50, 0.75, 0.90, 0.95, 0.99):
            print(f"  p{int(fraction * 100):<6}: {_percentile(ordered, fraction)}")

    if show_sample and first_record is not None:
        print("\nFirst record (values truncated):")
        for key, value in first_record.items():
            print(f"  {key}: {_preview(value)}")

    return {
        "exists": True,
        "path": str(path),
        "total": total,
        "valid": valid,
        "invalid_json": invalid_json,
        "code_field": code_field,
        "label_field": label_field,
        "labels": {str(k): v for k, v in labels.items()},
        "empty_code": empty_code,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=config.RAW_DIR,
        help="Directory holding primevul_{train,valid,test}.jsonl",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Do not print the first record of each split",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CODESENTINEL - RAW DATASET INSPECTION")
    print("=" * 70)
    print(f"Looking in: {args.dir}")

    if not args.dir.exists():
        print(f"\nERROR: directory does not exist: {args.dir}")
        print("Place primevul_train.jsonl / _valid.jsonl / _test.jsonl there.")
        raise SystemExit(1)

    summaries = {}

    for split in config.SPLITS:
        path = args.dir / f"primevul_{split}.jsonl"
        summaries[split] = inspect_file(split, path, show_sample=not args.no_sample)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for split, summary in summaries.items():
        if not summary.get("exists"):
            print(f"  {split:<6} MISSING")
            continue

        labels = summary["labels"]
        print(
            f"  {split:<6} {summary['valid']:>8} records"
            f"   vulnerable={labels.get('1', 0):>7}"
            f"   benign={labels.get('0', 0):>7}"
            f"   invalid={summary['invalid_json']}"
        )


if __name__ == "__main__":
    main()
