import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "primevul_train_balanced.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"

MAX_LENGTH = 512
STRIDE = 128


def chunk_code(code, tokenizer):
    """
    Split code into overlapping chunks.

    Each chunk has at most MAX_LENGTH tokens.
    STRIDE controls the overlap between chunks.
    """

    encoding = tokenizer(
        code,
        add_special_tokens=False,
        truncation=False
    )

    token_ids = encoding["input_ids"]

    chunk_size = MAX_LENGTH - 2
    step_size = chunk_size - STRIDE

    chunks = []

    for start in range(0, len(token_ids), step_size):
        chunk_ids = token_ids[start:start + chunk_size]

        if not chunk_ids:
            break

        input_ids = (
            [tokenizer.bos_token_id]
            + chunk_ids
            + [tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(input_ids)

        chunks.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def main():
    print("=" * 60)
    print("CODEBERT CHUNKING TEST")
    print("=" * 60)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records.append(record)

            if len(records) == 3:
                break

    for index, record in enumerate(records, start=1):
        code = record["func"]

        raw_tokens = tokenizer(
            code,
            add_special_tokens=False,
            truncation=False
        )["input_ids"]

        chunks = chunk_code(code, tokenizer)

        print("\n" + "-" * 60)
        print(f"SAMPLE {index}")
        print("-" * 60)
        print(f"Target: {record['target']}")
        print(f"Original token count: {len(raw_tokens)}")
        print(f"Number of chunks: {len(chunks)}")

        for chunk_index, chunk in enumerate(chunks, start=1):
            print(
                f"Chunk {chunk_index}: "
                f"{len(chunk['input_ids'])} tokens"
            )

    print("\nChunking test completed successfully.")


if __name__ == "__main__":
    main()
