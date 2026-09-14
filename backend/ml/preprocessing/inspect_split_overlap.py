"""
Inspect exact duplicate functions across PrimeVul splits.

For each overlapping function, reports:
- split
- label
- hash
- project
- CWE

Also detects whether identical code has conflicting labels.
"""

import json
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DATASETS = {
    "train": (
        PROJECT_ROOT
        / "data"
        / "chunked"
        / "primevul_train_chunked.jsonl"
    ),
    "valid": (
        PROJECT_ROOT
        / "data"
        / "chunked"
        / "primevul_valid_chunked.jsonl"
    ),
    "test": (
        PROJECT_ROOT
        / "data"
        / "chunked"
        / "primevul_test_chunked.jsonl"
    ),
}


def make_signature(record):

    return json.dumps(
        record.get("chunks", []),
        separators=(",", ":"),
    )


def load_dataset(name, path):

    records = defaultdict(list)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            signature = make_signature(
                record
            )

            records[signature].append(
                {
                    "split": name,
                    "target": record.get(
                        "target"
                    ),
                    "hash": record.get(
                        "hash"
                    ),
                    "project": record.get(
                        "project"
                    ),
                    "cwe": record.get(
                        "cwe"
                    ),
                }
            )

    return records


def main():

    print("=" * 70)
    print("DETAILED PRIMEVUL SPLIT OVERLAP INSPECTION")
    print("=" * 70)

    all_records = defaultdict(list)

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    for name, path in DATASETS.items():

        print(
            f"\nLoading {name}: {path}"
        )

        split_records = load_dataset(
            name,
            path,
        )

        for signature, records in (
            split_records.items()
        ):

            all_records[
                signature
            ].extend(records)

    # --------------------------------------------------------
    # FIND CROSS-SPLIT DUPLICATES
    # --------------------------------------------------------

    overlaps = []

    same_label_count = 0
    conflicting_label_count = 0

    pair_counts = defaultdict(int)

    for signature, records in (
        all_records.items()
    ):

        splits = {
            record["split"]
            for record in records
        }

        if len(splits) > 1:

            labels = {
                record["target"]
                for record in records
            }

            overlaps.append(
                (
                    signature,
                    records,
                )
            )

            split_names = sorted(
                splits
            )

            for i in range(
                len(split_names)
            ):

                for j in range(
                    i + 1,
                    len(split_names),
                ):

                    pair = (
                        split_names[i],
                        split_names[j],
                    )

                    pair_counts[
                        pair
                    ] += 1

            if len(labels) == 1:

                same_label_count += 1

            else:

                conflicting_label_count += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Total unique cross-split "
        f"duplicate functions: "
        f"{len(overlaps)}"
    )

    print(
        f"Same-label duplicates: "
        f"{same_label_count}"
    )

    print(
        f"Conflicting-label duplicates: "
        f"{conflicting_label_count}"
    )

    print("\nOverlap pairs:")

    for pair, count in sorted(
        pair_counts.items()
    ):

        print(
            f"{pair[0]} vs "
            f"{pair[1]}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # SHOW EXAMPLES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERLAP EXAMPLES")
    print("=" * 70)

    max_examples = 10

    for index, (
        signature,
        records,
    ) in enumerate(
        overlaps[:max_examples],
        start=1,
    ):

        print(
            f"\nDuplicate Function "
            f"{index}"
        )

        print(
            "-" * 60
        )

        for record in records:

            print(
                f"Split: "
                f"{record['split']}"
            )

            print(
                f"Target: "
                f"{record['target']}"
            )

            print(
                f"Hash: "
                f"{record['hash']}"
            )

            print(
                f"Project: "
                f"{record['project']}"
            )

            print(
                f"CWE: "
                f"{record['cwe']}"
            )

            print()

    # --------------------------------------------------------
    # CONCLUSION
    # --------------------------------------------------------

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if conflicting_label_count > 0:

        print(
            "\nCRITICAL: Identical functions "
            "with conflicting labels were found."
        )

    elif len(overlaps) > 0:

        print(
            "\nWARNING: Exact duplicate functions "
            "exist across dataset splits."
        )

        print(
            "These should be removed from "
            "validation/test evaluation to prevent "
            "data leakage."
        )

    else:

        print(
            "\nPASS: No cross-split exact "
            "duplicates found."
        )


if __name__ == "__main__":
    main()
