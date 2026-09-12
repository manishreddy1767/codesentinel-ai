"""
Stage 2 - DATA VALIDATION and LEAKAGE CHECK.

Asserts the dataset is fit to train on, and - critically - checks for
duplicate functions *across* splits, which would leak test data into
training and inflate every reported metric.

    python -m backend.ml.preprocessing.validate_dataset
    python -m backend.ml.preprocessing.validate_dataset --dir data/processed

Read-only. Exits non-zero if a blocking problem is found.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import detect_field, read_jsonl


def _normalize_for_hash(code: str) -> str:
    """
    Collapse whitespace so that reformatting alone does not hide a duplicate.

    Deliberately conservative: it does not strip comments or rename
    identifiers, so it detects copies rather than semantic clones.
    """

    return " ".join(code.split())


def _code_hash(code: str) -> str:
    return hashlib.sha256(
        _normalize_for_hash(code).encode("utf-8", errors="replace")
    ).hexdigest()


def validate_split(name: str, path: Path) -> dict:
    print("\n" + "-" * 70)
    print(f"VALIDATING: {name}  ({path})")
    print("-" * 70)

    problems = []
    warnings = []

    if not path.exists():
        print("  MISSING FILE")
        return {
            "ok": False,
            "problems": [f"{path} does not exist"],
            "warnings": [],
            "hashes": {},
            "labels": Counter(),
        }

    total = 0
    empty_code = 0
    bad_labels = Counter()
    labels = Counter()
    hashes = {}
    internal_duplicates = 0

    code_field = None
    label_field = None

    raw_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_lines += 1

    for _, record in read_jsonl(path, skip_invalid=True):
        total += 1

        if code_field is None:
            code_field = detect_field(record, config.CODE_FIELD_CANDIDATES)
            label_field = detect_field(record, config.LABEL_FIELD_CANDIDATES)

        code = record.get(code_field) if code_field else None

        if not isinstance(code, str) or not code.strip():
            empty_code += 1
            continue

        label = record.get(label_field) if label_field else None

        # Accept ints and the string forms of 0/1; reject anything else.
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            bad_labels[repr(label)] += 1
            continue

        if label_int not in (0, 1):
            bad_labels[repr(label)] += 1
            continue

        labels[label_int] += 1

        digest = _code_hash(code)

        if digest in hashes:
            internal_duplicates += 1
        else:
            hashes[digest] = label_int

    invalid_json = raw_lines - total

    print(f"  non-blank lines        : {raw_lines}")
    print(f"  parsed records         : {total}")
    print(f"  invalid JSON lines     : {invalid_json}")
    print(f"  code field             : {code_field}")
    print(f"  label field            : {label_field}")
    print(f"  usable records         : {sum(labels.values())}")
    print(f"  vulnerable (1)         : {labels[1]}")
    print(f"  benign (0)             : {labels[0]}")
    print(f"  empty / missing code   : {empty_code}")
    print(f"  invalid label values   : {sum(bad_labels.values())}")
    print(f"  duplicates within split: {internal_duplicates}")

    if bad_labels:
        print("    offending label values:")
        for value, count in bad_labels.most_common(5):
            print(f"      {value}: {count}")

    if code_field is None:
        problems.append(f"{name}: no recognisable code field")

    if label_field is None:
        problems.append(f"{name}: no recognisable label field")

    if sum(labels.values()) == 0:
        problems.append(f"{name}: zero usable records")

    if labels[1] == 0:
        problems.append(f"{name}: contains no vulnerable samples")

    if labels[0] == 0:
        problems.append(f"{name}: contains no benign samples")

    if invalid_json:
        problems_note = f"{name}: {invalid_json} malformed JSON lines will be dropped"
        warnings.append(problems_note)

    if empty_code:
        warnings.append(
            f"{name}: {empty_code} records have empty code and will be dropped"
        )

    if internal_duplicates:
        warnings.append(
            f"{name}: {internal_duplicates} duplicate functions inside the split"
        )

    if labels[1] and labels[0]:
        ratio = labels[0] / labels[1]
        print(f"  negative:positive      : {ratio:.2f} : 1")

        if ratio > 20:
            warnings.append(
                f"{name}: severe imbalance ({ratio:.1f}:1) - pos_weight is essential"
            )

    return {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "hashes": hashes,
        "labels": labels,
    }


def check_cross_split_leakage(results: dict) -> list:
    """Duplicate functions shared between splits are training-on-test leakage."""

    print("\n" + "=" * 70)
    print("CROSS-SPLIT LEAKAGE CHECK")
    print("=" * 70)

    problems = []
    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))

    for left, right in pairs:
        left_hashes = results.get(left, {}).get("hashes", {})
        right_hashes = results.get(right, {}).get("hashes", {})

        if not left_hashes or not right_hashes:
            print(f"  {left:<6} vs {right:<6}: skipped (a split is unavailable)")
            continue

        shared = set(left_hashes) & set(right_hashes)

        # A function on both sides with opposite labels is worse than a plain
        # duplicate: it makes that pair unlearnable as well as leaked.
        conflicting = sum(
            1 for digest in shared if left_hashes[digest] != right_hashes[digest]
        )

        status = "CLEAN" if not shared else "LEAKAGE"
        print(
            f"  {left:<6} vs {right:<6}: {status}"
            f"  shared={len(shared)}  label-conflicting={conflicting}"
        )

        if shared:
            problems.append(
                f"{len(shared)} functions appear in both {left} and {right}"
            )

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat cross-split leakage as a blocking failure",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CODESENTINEL - DATASET VALIDATION")
    print("=" * 70)
    print(f"Directory: {args.dir}")

    results = {}

    for split in config.SPLITS:
        results[split] = validate_split(split, args.dir / f"primevul_{split}.jsonl")

    leakage = check_cross_split_leakage(results)

    all_problems = []
    all_warnings = []

    for result in results.values():
        all_problems.extend(result.get("problems", []))
        all_warnings.extend(result.get("warnings", []))

    print("\n" + "=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    if all_warnings:
        print("\nWarnings (non-blocking):")
        for warning in all_warnings:
            print(f"  - {warning}")

    if leakage:
        print("\nLeakage findings:")
        for item in leakage:
            print(f"  - {item}")
        print(
            "\n  PrimeVul ships paired vulnerable/fixed functions, so some overlap\n"
            "  can be legitimate, but any function shared between train and test\n"
            "  inflates test metrics. Review before trusting results."
        )

    if all_problems:
        print("\nBlocking problems:")
        for problem in all_problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    if leakage and args.strict:
        print("\nFAILED: --strict was set and leakage was detected.")
        raise SystemExit(1)

    print("\nPASSED: dataset is usable for training.")


if __name__ == "__main__":
    main()
