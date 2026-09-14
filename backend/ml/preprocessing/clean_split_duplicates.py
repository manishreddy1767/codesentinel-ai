import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = INPUT_DIR / "primevul_train_balanced.jsonl"
VALID_FILE = INPUT_DIR / "primevul_valid.jsonl"
TEST_FILE = INPUT_DIR / "primevul_test.jsonl"

OUTPUT_TRAIN = OUTPUT_DIR / "primevul_train_clean.jsonl"
OUTPUT_VALID = OUTPUT_DIR / "primevul_valid_clean.jsonl"
OUTPUT_TEST = OUTPUT_DIR / "primevul_test_clean.jsonl"


def normalize_code(code):
    """
    Normalize code for exact duplicate comparison.

    Removes leading/trailing whitespace and normalizes
    line endings while preserving the actual code content.
    """
    if not isinstance(code, str):
        return ""

    return code.replace("\r\n", "\n").strip()


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_code_set(records):
    return {
        normalize_code(record.get("func", ""))
        for record in records
    }


def remove_duplicates(records, blocked_codes):
    cleaned = []
    removed = 0

    for record in records:

        code = normalize_code(
            record.get("func", "")
        )

        if code in blocked_codes:
            removed += 1
            continue

        cleaned.append(record)

    return cleaned, removed


def print_stats(name, records):
    counts = Counter(
        record.get("target", 0)
        for record in records
    )

    total = len(records)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Total samples: {total}")

    for label in sorted(counts):

        count = counts[label]

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"Label {label}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


def main():

    print("=" * 70)
    print("PRIMEVUL CROSS-SPLIT DUPLICATE CLEANING")
    print("=" * 70)

    print("\nLoading datasets...")

    train_records = load_jsonl(
        TRAIN_FILE
    )

    valid_records = load_jsonl(
        VALID_FILE
    )

    test_records = load_jsonl(
        TEST_FILE
    )

    print_stats(
        "ORIGINAL TRAIN",
        train_records,
    )

    print_stats(
        "ORIGINAL VALIDATION",
        valid_records,
    )

    print_stats(
        "ORIGINAL TEST",
        test_records,
    )

    # --------------------------------------------------
    # KEEP TRAINING DATA UNCHANGED
    # --------------------------------------------------

    print(
        "\nKeeping training split unchanged..."
    )

    cleaned_train = train_records

    train_codes = get_code_set(
        cleaned_train
    )

    # --------------------------------------------------
    # CLEAN VALIDATION
    # --------------------------------------------------

    print(
        "\nRemoving validation functions "
        "that appear in training..."
    )

    cleaned_valid, removed_valid = (
        remove_duplicates(
            valid_records,
            train_codes,
        )
    )

    print(
        f"Removed from validation: "
        f"{removed_valid}"
    )

    valid_codes = get_code_set(
        cleaned_valid
    )

    # --------------------------------------------------
    # CLEAN TEST
    # --------------------------------------------------

    print(
        "\nRemoving test functions "
        "that appear in training "
        "or validation..."
    )

    blocked_test_codes = (
        train_codes
        | valid_codes
    )

    cleaned_test, removed_test = (
        remove_duplicates(
            test_records,
            blocked_test_codes,
        )
    )

    print(
        f"Removed from test: "
        f"{removed_test}"
    )

    # --------------------------------------------------
    # SAVE CLEAN DATASETS
    # --------------------------------------------------

    print(
        "\nSaving cleaned datasets..."
    )

    save_jsonl(
        cleaned_train,
        OUTPUT_TRAIN,
    )

    save_jsonl(
        cleaned_valid,
        OUTPUT_VALID,
    )

    save_jsonl(
        cleaned_test,
        OUTPUT_TEST,
    )

    # --------------------------------------------------
    # FINAL STATS
    # --------------------------------------------------

    print_stats(
        "CLEAN TRAIN",
        cleaned_train,
    )

    print_stats(
        "CLEAN VALIDATION",
        cleaned_valid,
    )

    print_stats(
        "CLEAN TEST",
        cleaned_test,
    )

    # --------------------------------------------------
    # FINAL OVERLAP CHECK
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CROSS-SPLIT OVERLAP CHECK")
    print("=" * 70)

    clean_train_codes = get_code_set(
        cleaned_train
    )

    clean_valid_codes = get_code_set(
        cleaned_valid
    )

    clean_test_codes = get_code_set(
        cleaned_test
    )

    train_valid_overlap = (
        clean_train_codes
        & clean_valid_codes
    )

    train_test_overlap = (
        clean_train_codes
        & clean_test_codes
    )

    valid_test_overlap = (
        clean_valid_codes
        & clean_test_codes
    )

    print(
        "\nTrain vs Validation overlap:",
        len(train_valid_overlap),
    )

    print(
        "Train vs Test overlap:",
        len(train_test_overlap),
    )

    print(
        "Validation vs Test overlap:",
        len(valid_test_overlap),
    )

    if (
        len(train_valid_overlap) == 0
        and len(train_test_overlap) == 0
        and len(valid_test_overlap) == 0
    ):

        print(
            "\nPASS: No exact duplicate "
            "functions remain across splits."
        )

    else:

        print(
            "\nWARNING: Cross-split duplicates "
            "still remain."
        )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)

    print("\nOutput files:")

    print(OUTPUT_TRAIN)
    print(OUTPUT_VALID)
    print(OUTPUT_TEST)


if __name__ == "__main__":
    main()
