import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_NAME = "microsoft/codebert-base"

FILES = {
    "train": PROJECT_ROOT / "data" / "processed" / "primevul_train_balanced.jsonl",
    "valid": PROJECT_ROOT / "data" / "processed" / "primevul_valid.jsonl",
    "test": PROJECT_ROOT / "data" / "processed" / "primevul_test.jsonl",
}


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    index = int((len(values) - 1) * p)
    return values[index]


def analyze_file(name, path, tokenizer):
    lengths = []
    truncated_512 = 0

    print("\n" + "=" * 60)
    print(f"{name.upper()} DATASET")
    print("=" * 60)

    with open(path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            record = json.loads(line)
            code = record.get("func", "")

            tokens = tokenizer(
                code,
                add_special_tokens=True,
                truncation=False
            )

            token_length = len(tokens["input_ids"])
            lengths.append(token_length)

            if token_length > 512:
                truncated_512 += 1

            if i % 5000 == 0:
                print(f"Processed {i} samples...")

    total = len(lengths)

    print(f"\nSamples: {total}")
    print(f"Minimum tokens: {min(lengths)}")
    print(f"Maximum tokens: {max(lengths)}")
    print(f"Average tokens: {sum(lengths) / total:.2f}")

    print("\nToken length percentiles:")
    print(f"50th percentile: {percentile(lengths, 0.50)}")
    print(f"75th percentile: {percentile(lengths, 0.75)}")
    print(f"90th percentile: {percentile(lengths, 0.90)}")
    print(f"95th percentile: {percentile(lengths, 0.95)}")
    print(f"99th percentile: {percentile(lengths, 0.99)}")

    print("\n512-token limit analysis:")
    print(f"Samples exceeding 512 tokens: {truncated_512}")
    print(f"Percentage truncated: {(truncated_512 / total) * 100:.2f}%")


def main():
    print("=" * 60)
    print("CODEBERT TOKEN LENGTH ANALYSIS")
    print("=" * 60)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for name, path in FILES.items():
        analyze_file(name, path, tokenizer)


if __name__ == "__main__":
    main()
