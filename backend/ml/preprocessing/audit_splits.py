"""
Audit PrimeVul train/validation/test splits for overlap.

Checks:
1. Dataset sizes
2. Label distributions
3. Duplicate hashes within each split
4. Hash overlap between splits
5. Exact duplicate chunk sequences between splits
"""

import json
from pathlib import Path
from collections import Counter


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


def load_split(path):

    hashes = []
    labels = []
    chunk_signatures = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            target = record.get("target")

            record_hash = record.get(
                "hash",
                "",
            )

            chunks = record.get(
                "chunks",
                [],
            )

            labels.append(target)

            hashes.append(record_hash)

            # Exact representation of the tokenized function.
            # Used to detect exact duplicate content.
            signature = json.dumps(
                chunks,
                separators=(",", ":"),
            )

            chunk_signatures.append(
                signature
            )

    return {
        "hashes": hashes,
        "labels": labels,
        "chunk_signatures": chunk_signatures,
    }


def report_split(name, data):

    hashes = data["hashes"]

    labels = data["labels"]

    signatures = data[
        "chunk_signatures"
    ]

    hash_counter = Counter(
        hashes
    )

    signature_counter = Counter(
        signatures
    )

    duplicate_hashes = {
        key: value
        for key, value in hash_counter.items()
        if value > 1 and key
    }

    duplicate_signatures = {
        key: value
        for key, value in signature_counter.items()
        if value > 1
    }

    print("\n" + "=" * 70)

    print(
        f"{name.upper()} SPLIT"
    )

    print("=" * 70)

    print(
        f"Total samples: "
        f"{len(labels)}"
    )

    print(
        f"Label counts: "
        f"{Counter(labels)}"
    )

    print(
        f"Unique hashes: "
        f"{len(set(hashes))}"
    )

    print(
        f"Duplicate hashes: "
        f"{len(duplicate_hashes)}"
    )

    print(
        f"Unique exact contents: "
        f"{len(set(signatures))}"
    )

    print(
        f"Duplicate exact contents: "
        f"{len(duplicate_signatures)}"
    )


def compare_splits(
    name_a,
    data_a,
    name_b,
    data_b,
):

    hashes_a = {
        value
        for value in data_a["hashes"]
        if value
    }

    hashes_b = {
        value
        for value in data_b["hashes"]
        if value
    }

    signatures_a = set(
        data_a["chunk_signatures"]
    )

    signatures_b = set(
        data_b["chunk_signatures"]
    )

    hash_overlap = (
        hashes_a
        & hashes_b
    )

    content_overlap = (
        signatures_a
        & signatures_b
    )

    print("\n" + "-" * 70)

    print(
        f"{name_a.upper()} "
        f"vs "
        f"{name_b.upper()}"
    )

    print("-" * 70)

    print(
        f"Hash overlap: "
        f"{len(hash_overlap)}"
    )

    print(
        f"Exact content overlap: "
        f"{len(content_overlap)}"
    )

    if hash_overlap:

        print(
            "\nWARNING: Matching hashes "
            "found between splits."
        )

        examples = list(
            hash_overlap
        )[:5]

        print(
            "Example hashes:"
        )

        for value in examples:

            print(
                f"  {value}"
            )

    if content_overlap:

        print(
            "\nWARNING: Exact duplicate "
            "functions found between splits."
        )

    if (
        not hash_overlap
        and
        not content_overlap
    ):

        print(
            "\nPASS: No exact overlap "
            "detected."
        )


def main():

    print("=" * 70)

    print(
        "PRIMEVUL DATASET SPLIT AUDIT"
    )

    print("=" * 70)

    loaded_data = {}

    # --------------------------------------------------------
    # LOAD SPLITS
    # --------------------------------------------------------

    for name, path in DATASETS.items():

        print(
            f"\nLoading {name}:"
        )

        print(path)

        if not path.exists():

            print(
                "\nERROR: Dataset file "
                "not found."
            )

            return

        loaded_data[name] = (
            load_split(path)
        )

    # --------------------------------------------------------
    # INDIVIDUAL REPORTS
    # --------------------------------------------------------

    for name, data in (
        loaded_data.items()
    ):

        report_split(
            name,
            data,
        )

    # --------------------------------------------------------
    # CROSS-SPLIT AUDITS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "CROSS-SPLIT OVERLAP CHECK"
    )

    print("=" * 70)

    compare_splits(
        "train",
        loaded_data["train"],
        "valid",
        loaded_data["valid"],
    )

    compare_splits(
        "train",
        loaded_data["train"],
        "test",
        loaded_data["test"],
    )

    compare_splits(
        "valid",
        loaded_data["valid"],
        "test",
        loaded_data["test"],
    )

    print("\n" + "=" * 70)

    print(
        "AUDIT COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
