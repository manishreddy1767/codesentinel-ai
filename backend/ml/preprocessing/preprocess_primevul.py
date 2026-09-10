import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "primevul_dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_code(code):
    """Basic normalization while preserving the source code structure."""
    if not isinstance(code, str):
        return ""

    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in code.split("\n")]

    # Remove excessive blank lines
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank and previous_blank:
            continue

        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def normalize_cwe(cwe):
    """Convert CWE values into a consistent list format."""
    if cwe is None:
        return []

    if isinstance(cwe, str):
        cwe = cwe.strip()
        return [cwe] if cwe else []

    if isinstance(cwe, list):
        return [
            str(item).strip()
            for item in cwe
            if str(item).strip()
        ]

    return [str(cwe).strip()]


def process_file(filename):
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / filename

    total = 0
    kept = 0
    skipped = 0

    print(f"\nProcessing: {filename}")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            code = normalize_code(record.get("func", ""))

            if not code:
                skipped += 1
                continue

            target = record.get("target")

            if target not in [0, 1]:
                skipped += 1
                continue

            processed_record = {
                "project": record.get("project", ""),
                "commit_id": record.get("commit_id", ""),
                "target": int(target),
                "func": code,
                "cwe": normalize_cwe(record.get("cwe")),
                "big_vul_idx": record.get("big_vul_idx"),
                "idx": record.get("idx"),
                "hash": record.get("hash")
            }

            outfile.write(
                json.dumps(processed_record, ensure_ascii=False) + "\n"
            )

            kept += 1

    print(f"Total records: {total}")
    print(f"Kept records: {kept}")
    print(f"Skipped records: {skipped}")
    print(f"Saved to: {output_path}")


def main():
    files = [
        "primevul_train.jsonl",
        "primevul_valid.jsonl",
        "primevul_test.jsonl"
    ]

    print("=" * 60)
    print("PRIMEVUL PREPROCESSING PIPELINE")
    print("=" * 60)

    for filename in files:
        process_file(filename)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
