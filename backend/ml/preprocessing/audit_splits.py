"""
Stage C - MULTI-DIMENSIONAL SPLIT LEAKAGE AUDIT.

    python -m backend.ml.preprocessing.audit_splits
    python -m backend.ml.preprocessing.audit_splits --dir data/processed --strict
    python -m backend.ml.preprocessing.audit_splits --json artifacts/reports/leakage.json

Read-only. Never writes to the directory it audits.

Why this exists alongside validate_dataset.py
---------------------------------------------
``validate_dataset.py`` hashes whitespace-normalised code and reports a single
"shared / not shared" verdict per split pair. That catches byte-identical and
reformatting-only copies, and nothing else. It cannot see:

  * a function lightly edited between splits that is the *same* function by
    upstream identity - and PrimeVul ships paired vulnerable/fixed functions,
    so identity-level overlap is a realistic risk rather than a theoretical one;
  * project-level overlap, where the same codebase appears on both sides.

This module checks each dimension separately and reports them separately,
because they do not mean the same thing:

  EXACT / NORMALISED CODE   the same function body is in two splits.
                            Unambiguous leakage: the model can memorise it.

  UPSTREAM IDENTITY         `hash`, `idx`, `big_vul_idx`, `commit_id` agree
                            across splits. Leakage, and it survives edits that
                            defeat content hashing.

  PROJECT                   the same repository appears in two splits. NOT
                            automatically leakage - it is a property of how
                            PrimeVul was split - but it inflates generalisation
                            estimates, because a model can learn project-specific
                            idiom rather than vulnerability. Reported as a
                            concern, never as a blocking failure.

LABEL-CONFLICTING duplicates are called out separately at every level: the same
key carrying opposite labels on the two sides is worse than a plain duplicate,
because it makes that pair unlearnable as well as leaked.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import detect_field, read_jsonl

# Dimensions checked, in order of severity.
#
#   kind="code"     content-derived, computed here
#   kind="identity" upstream record identifiers, used only when present
#   kind="context"  dataset structure; informational only
DIMENSIONS = (
    ("exact_code", "code", "Byte-identical function body"),
    ("normalized_code", "code", "Function body ignoring whitespace differences"),
    ("hash", "identity", "Upstream PrimeVul `hash` field"),
    ("idx", "identity", "Upstream `idx` field"),
    ("big_vul_idx", "identity", "Upstream `big_vul_idx` field"),
    ("commit_id", "identity", "Source commit identifier"),
    ("project", "context", "Source repository / project"),
)

BLOCKING_KINDS = ("code", "identity")


def _exact_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()


def _normalized_hash(code: str) -> str:
    """
    Whitespace-collapsed hash.

    Deliberately conservative: it does not strip comments or rename
    identifiers, so it finds copies rather than semantic clones. Matching
    validate_dataset.py exactly keeps the two tools' numbers comparable.
    """

    return hashlib.sha256(
        " ".join(code.split()).encode("utf-8", errors="replace")
    ).hexdigest()


def collect_keys(path: Path) -> dict:
    """
    Build ``{dimension: {key: Counter({label: n})}}`` for one split.

    Labels are kept per key so that cross-split label conflicts can be detected
    rather than merely counted. Absent dimensions stay empty dicts, which is how
    a missing optional field is distinguished from a field with no overlap.
    """

    keys = {name: defaultdict(Counter) for name, _, _ in DIMENSIONS}

    summary = {
        "path": str(path),
        "exists": path.exists(),
        "records": 0,
        "usable": 0,
        "skipped_no_code": 0,
        "skipped_bad_label": 0,
        "labels": Counter(),
        "fields_present": Counter(),
        "code_field": None,
        "label_field": None,
    }

    if not path.exists():
        return {"keys": keys, "summary": summary}

    code_field = None
    label_field = None

    for _, record in read_jsonl(path, skip_invalid=True):
        summary["records"] += 1

        # Re-detect until BOTH are found. The original implementation gated
        # label detection behind `code_field is None`, so a first record that
        # had code but no label pinned label_field to None permanently and
        # every later record was counted as an invalid label.
        if code_field is None:
            code_field = detect_field(record, config.CODE_FIELD_CANDIDATES)
        if label_field is None:
            label_field = detect_field(record, config.LABEL_FIELD_CANDIDATES)

        code = record.get(code_field) if code_field else None

        if not isinstance(code, str) or not code.strip():
            summary["skipped_no_code"] += 1
            continue

        try:
            label = int(record.get(label_field))
        except (TypeError, ValueError):
            summary["skipped_bad_label"] += 1
            continue

        if label not in (0, 1):
            summary["skipped_bad_label"] += 1
            continue

        summary["usable"] += 1
        summary["labels"][label] += 1

        keys["exact_code"][_exact_hash(code)][label] += 1
        keys["normalized_code"][_normalized_hash(code)][label] += 1

        for name, kind, _ in DIMENSIONS:
            if kind == "code":
                continue

            value = record.get(name)

            # `0` and `""` are legitimate idx values, so test for None only.
            if value is None or value == "":
                continue

            summary["fields_present"][name] += 1
            keys[name][str(value)][label] += 1

    summary["code_field"] = code_field
    summary["label_field"] = label_field

    return {"keys": keys, "summary": summary}


def compare(left_keys: dict, right_keys: dict) -> dict:
    """Overlap between two splits along one dimension."""

    shared = set(left_keys) & set(right_keys)

    conflicting = []
    left_records = 0
    right_records = 0

    for key in shared:
        left_labels = left_keys[key]
        right_labels = right_keys[key]

        left_records += sum(left_labels.values())
        right_records += sum(right_labels.values())

        # Conflict = the label sets genuinely differ, not merely that counts do.
        if set(left_labels) != set(right_labels):
            conflicting.append(key)

    return {
        "shared_keys": len(shared),
        "left_records_affected": left_records,
        "right_records_affected": right_records,
        "label_conflicting_keys": len(conflicting),
        "examples": sorted(shared)[:5],
        "conflicting_examples": sorted(conflicting)[:5],
    }


def internal_duplicates(keys: dict) -> dict:
    """Duplicate keys *within* one split, along one dimension."""

    duplicate_keys = 0
    redundant_records = 0
    conflicting = 0

    for labels in keys.values():
        count = sum(labels.values())

        if count > 1:
            duplicate_keys += 1
            redundant_records += count - 1

            if len(labels) > 1:
                conflicting += 1

    return {
        "duplicate_keys": duplicate_keys,
        "redundant_records": redundant_records,
        "label_conflicting_keys": conflicting,
    }


def audit(directory: Path, splits=config.SPLITS) -> dict:
    """Run the full audit. Returns a JSON-serialisable report."""

    collected = {}

    for split in splits:
        path = directory / f"primevul_{split}.jsonl"
        print(f"  reading {split:<6} {path}")
        collected[split] = collect_keys(path)

    report = {
        "directory": str(directory),
        "splits": {},
        "within_split_duplicates": {},
        "cross_split": {},
        "findings": [],
        "concerns": [],
    }

    for split in splits:
        summary = dict(collected[split]["summary"])
        summary["labels"] = dict(summary["labels"])
        summary["fields_present"] = dict(summary["fields_present"])
        report["splits"][split] = summary

        report["within_split_duplicates"][split] = {
            name: internal_duplicates(collected[split]["keys"][name])
            for name, _, _ in DIMENSIONS
        }

    pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))

    for left, right in pairs:
        if left not in collected or right not in collected:
            continue

        if not collected[left]["summary"]["exists"]:
            continue
        if not collected[right]["summary"]["exists"]:
            continue

        pair_name = f"{left}__{right}"
        report["cross_split"][pair_name] = {}

        for name, kind, description in DIMENSIONS:
            left_keys = collected[left]["keys"][name]
            right_keys = collected[right]["keys"][name]

            if not left_keys or not right_keys:
                report["cross_split"][pair_name][name] = {
                    "kind": kind,
                    "available": False,
                    "reason": "dimension absent from at least one split",
                }
                continue

            result = compare(left_keys, right_keys)
            result["kind"] = kind
            result["available"] = True
            result["description"] = description

            report["cross_split"][pair_name][name] = result

            if not result["shared_keys"]:
                continue

            message = (
                f"{left} vs {right}: {result['shared_keys']} shared "
                f"{name} keys "
                f"({result['right_records_affected']} {right} records affected"
                f", {result['label_conflicting_keys']} label-conflicting)"
            )

            if kind in BLOCKING_KINDS:
                report["findings"].append(message)
            else:
                report["concerns"].append(message)

    return report


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_report(report: dict) -> None:
    print("\n" + "=" * 78)
    print("SPLIT SUMMARY")
    print("=" * 78)

    for split, summary in report["splits"].items():
        if not summary["exists"]:
            print(f"  {split:<6} MISSING  {summary['path']}")
            continue

        labels = summary["labels"]
        positives = labels.get(1, 0)
        negatives = labels.get(0, 0)
        ratio = f"{negatives / positives:.1f}:1" if positives else "n/a"

        print(
            f"  {split:<6} records={summary['records']:>8}"
            f"  usable={summary['usable']:>8}"
            f"  vuln={positives:>7}  benign={negatives:>7}  ratio={ratio}"
        )
        print(
            f"         code_field={summary['code_field']}"
            f"  label_field={summary['label_field']}"
            f"  dropped(no_code={summary['skipped_no_code']},"
            f" bad_label={summary['skipped_bad_label']})"
        )

        present = summary["fields_present"]
        available = ", ".join(f"{k}={v}" for k, v in sorted(present.items())) or "none"
        print(f"         identity fields present: {available}")

    print("\n" + "=" * 78)
    print("DUPLICATES WITHIN EACH SPLIT")
    print("=" * 78)
    print(f"  {'split':<7}{'dimension':<18}{'dup keys':>10}{'redundant':>11}{'conflict':>10}")

    for split, dimensions in report["within_split_duplicates"].items():
        for name, result in dimensions.items():
            if not result["duplicate_keys"]:
                continue
            print(
                f"  {split:<7}{name:<18}"
                f"{result['duplicate_keys']:>10}"
                f"{result['redundant_records']:>11}"
                f"{result['label_conflicting_keys']:>10}"
            )

    print("\n" + "=" * 78)
    print("CROSS-SPLIT OVERLAP  (by dimension, severity-ordered)")
    print("=" * 78)

    for pair_name, dimensions in report["cross_split"].items():
        left, right = pair_name.split("__")
        print(f"\n  {left} vs {right}")
        print(
            f"    {'dimension':<18}{'kind':<10}{'shared':>9}"
            f"{'affected':>10}{'conflict':>10}  verdict"
        )

        for name, kind, _ in DIMENSIONS:
            result = dimensions.get(name, {})

            if not result.get("available"):
                print(f"    {name:<18}{kind:<10}{'-':>9}{'-':>10}{'-':>10}  not available")
                continue

            shared = result["shared_keys"]

            if not shared:
                verdict = "CLEAN"
            elif kind == "context":
                verdict = "CONCERN (not leakage)"
            else:
                verdict = "LEAKAGE"

            print(
                f"    {name:<18}{kind:<10}{shared:>9}"
                f"{result['right_records_affected']:>10}"
                f"{result['label_conflicting_keys']:>10}  {verdict}"
            )

    print("\n" + "=" * 78)
    print("AUDIT RESULT")
    print("=" * 78)

    if report["findings"]:
        print("\n  LEAKAGE FINDINGS (blocking under --strict):")
        for item in report["findings"]:
            print(f"    - {item}")
    else:
        print("\n  No code-level or identity-level leakage detected.")

    if report["concerns"]:
        print("\n  GENERALISATION CONCERNS (never blocking):")
        for item in report["concerns"]:
            print(f"    - {item}")
        print(
            "\n    Project overlap is a property of the dataset's split design,\n"
            "    not proof of leakage. It does mean held-out metrics partly\n"
            "    measure within-project generalisation, so report it alongside\n"
            "    the headline numbers rather than treating them as cross-project."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.RAW_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when code/identity leakage is found",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full report as JSON",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("CODESENTINEL - SPLIT LEAKAGE AUDIT")
    print("=" * 78)
    print(f"Directory: {args.dir}\n")

    if not args.dir.exists():
        print(f"ERROR: directory not found: {args.dir}")
        raise SystemExit(2)

    report = audit(args.dir)
    print_report(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  JSON report written to {args.json}")

    if report["findings"] and args.strict:
        print("\nFAILED: --strict was set and leakage was detected.")
        raise SystemExit(1)

    print("\nAudit complete.")


if __name__ == "__main__":
    main()
