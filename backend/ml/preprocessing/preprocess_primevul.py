"""
Stage 3 - TRAINING DATA PREPROCESSING.

Reads the untouched raw splits from data/raw/ and writes normalised,
schema-stable records to data/processed/.

    python -m backend.ml.preprocessing.preprocess_primevul

Guarantees about the output:
  * every record has a non-empty ``func`` string and an int ``target`` in {0, 1}
  * field names are normalised regardless of what the raw export called them
  * malformed JSON, empty code and invalid labels are dropped and counted
  * exact duplicate functions are dropped from the TRAIN split only

Why duplicates are dropped only from train: validation and test must stay a
faithful sample of the real distribution. De-duplicating them would change
the evaluation set and make metrics incomparable to published PrimeVul
numbers. Removing duplicates from train only reduces redundant gradient
signal without touching what the model is measured against.

data/raw/ is opened read-only and is never modified.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import detect_field, read_jsonl


def normalize_code(code) -> str:
    """Light normalisation that preserves source structure and indentation."""

    if not isinstance(code, str):
        return ""

    code = code.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.rstrip() for line in code.split("\n")]

    # Collapse runs of blank lines to a single blank line.
    cleaned = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank and previous_blank:
            continue

        cleaned.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned).strip()


def normalize_cwe(cwe) -> list:
    """Coerce the CWE field into a consistent list of strings."""

    if cwe is None:
        return []

    if isinstance(cwe, str):
        cwe = cwe.strip()
        return [cwe] if cwe else []

    if isinstance(cwe, (list, tuple)):
        return [str(item).strip() for item in cwe if str(item).strip()]

    return [str(cwe).strip()]


def _code_hash(code: str) -> str:
    return hashlib.sha256(
        " ".join(code.split()).encode("utf-8", errors="replace")
    ).hexdigest()


def process_split(split: str, input_path: Path, output_path: Path, dedupe: bool):
    print("\n" + "-" * 70)
    print(f"PROCESSING: {split}")
    print(f"  in : {input_path}")
    print(f"  out: {output_path}")
    print("-" * 70)

    if not input_path.exists():
        print("  SKIPPED: input file not found.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    labels = Counter()
    seen = set()

    code_field = None
    label_field = None

    with output_path.open("w", encoding="utf-8") as out:
        import json

        for _, record in read_jsonl(input_path, skip_invalid=True):
            stats["read"] += 1

            if code_field is None:
                code_field = detect_field(record, config.CODE_FIELD_CANDIDATES)
                label_field = detect_field(record, config.LABEL_FIELD_CANDIDATES)

                if code_field is None or label_field is None:
                    raise ValueError(
                        f"{input_path}: could not detect code/label fields. "
                        f"Saw keys: {sorted(record)}"
                    )

            code = normalize_code(record.get(code_field))

            if not code:
                stats["dropped_empty_code"] += 1
                continue

            try:
                target = int(record.get(label_field))
            except (TypeError, ValueError):
                stats["dropped_bad_label"] += 1
                continue

            if target not in (0, 1):
                stats["dropped_bad_label"] += 1
                continue

            if dedupe:
                digest = _code_hash(code)

                if digest in seen:
                    stats["dropped_duplicate"] += 1
                    continue

                seen.add(digest)

            processed = {
                "func": code,
                "target": target,
                "cwe": normalize_cwe(record.get("cwe")),
            }

            for field in config.METADATA_FIELDS:
                if field == "cwe":
                    continue
                if field in record:
                    processed[field] = record[field]

            out.write(json.dumps(processed, ensure_ascii=False) + "\n")

            stats["kept"] += 1
            labels[target] += 1

    print(f"  detected code field : {code_field}")
    print(f"  detected label field: {label_field}")
    print(f"  records read        : {stats['read']}")
    print(f"  kept                : {stats['kept']}")
    print(f"  dropped empty code  : {stats['dropped_empty_code']}")
    print(f"  dropped bad label   : {stats['dropped_bad_label']}")
    print(f"  dropped duplicates  : {stats['dropped_duplicate']}")
    print(f"  vulnerable (1)      : {labels[1]}")
    print(f"  benign (0)          : {labels[0]}")

    if labels[1]:
        print(f"  negative:positive   : {labels[0] / labels[1]:.2f} : 1")

    return {"stats": stats, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate functions in the training split",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CODESENTINEL - PRIMEVUL PREPROCESSING")
    print("=" * 70)

    if not args.raw_dir.exists():
        print(f"\nERROR: raw directory not found: {args.raw_dir}")
        raise SystemExit(1)

    results = {}

    for split in config.SPLITS:
        results[split] = process_split(
            split,
            args.raw_dir / f"primevul_{split}.jsonl",
            args.out_dir / f"primevul_{split}.jsonl",
            # Validation and test are never de-duplicated.
            dedupe=(split == "train" and not args.no_dedupe),
        )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    for split, result in results.items():
        if result is None:
            print(f"  {split:<6} skipped")
            continue

        labels = result["labels"]
        print(
            f"  {split:<6} kept={result['stats']['kept']:>8}"
            f"  vulnerable={labels[1]:>7}  benign={labels[0]:>7}"
        )


if __name__ == "__main__":
    main()
