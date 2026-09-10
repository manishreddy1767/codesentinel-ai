import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "primevul_train_balanced.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"


def main():
    print("=" * 60)
    print("CODEBERT TOKENIZER TEST")
    print("=" * 60)

    print(f"\nLoading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizer loaded successfully.")

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        record = json.loads(next(file))

    code = record["func"]

    print("\nSample information:")
    print(f"Target: {record['target']}")
    print(f"Character length: {len(code)}")

    tokens = tokenizer(
        code,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    print("\nTokenization successful.")
    print(f"Input token shape: {tokens['input_ids'].shape}")
    print(f"Number of tokens: {tokens['input_ids'].shape[1]}")

    print("\nFirst 20 token IDs:")
    print(tokens["input_ids"][0][:20].tolist())


if __name__ == "__main__":
    main()
