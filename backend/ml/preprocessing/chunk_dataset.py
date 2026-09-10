import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunked"

MODEL_NAME = "microsoft/codebert-base"
MAX_TOKENS = 512


DATASETS = {
    "train": INPUT_DIR / "primevul_train_balanced.jsonl",
    "valid": INPUT_DIR / "primevul_valid.jsonl",
    "test": INPUT_DIR / "primevul_test.jsonl",
}


def chunk_function(tokenizer, code):
    tokens = tokenizer.encode(
        code,
        add_special_tokens=True,
        truncation=False
    )

    chunks = []

    for start in range(0, len(tokens), MAX_TOKENS):
        chunk = tokens[start:start + MAX_TOKENS]

        chunks.append(chunk)

    return chunks


def process_dataset(tokenizer, input_path, output_path):
    total_functions = 0
    total_chunks = 0
    max_chunks = 0

    with open(input_path, "r", encoding="utf-8") as input_file, \
         open(output_path, "w", encoding="utf-8") as output_file:

        for line in input_file:
            record = json.loads(line)

            code = record.get("func", "")
            target = record.get("target", 0)

            chunks = chunk_function(tokenizer, code)

            total_functions += 1
            total_chunks += len(chunks)
            max_chunks = max(max_chunks, len(chunks))

            output_record = {
                "target": target,
                "chunks": chunks,
                "num_chunks": len(chunks),
                "project": record.get("project", ""),
                "cwe": record.get("cwe", ""),
                "hash": record.get("hash", "")
            }

            output_file.write(
                json.dumps(output_record) + "\n"
            )

    return total_functions, total_chunks, max_chunks


def main():
    print("=" * 60)
    print("PRIMEVUL DATASET CHUNKING PIPELINE")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split_name, input_path in DATASETS.items():
        print("\n" + "=" * 60)
        print(f"PROCESSING: {split_name.upper()}")
        print("=" * 60)

        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            continue

        output_path = OUTPUT_DIR / f"primevul_{split_name}_chunked.jsonl"

        (
            total_functions,
            total_chunks,
            max_chunks
        ) = process_dataset(
            tokenizer,
            input_path,
            output_path
        )

        print(f"\nFunctions processed: {total_functions}")
        print(f"Total chunks: {total_chunks}")
        print(
            f"Average chunks per function: "
            f"{total_chunks / total_functions:.2f}"
        )
        print(f"Maximum chunks for one function: {max_chunks}")
        print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
