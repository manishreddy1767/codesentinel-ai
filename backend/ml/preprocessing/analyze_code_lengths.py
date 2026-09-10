import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def analyze_file(name, path):
    lengths = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            code = record.get("func", "")
            lengths.append(len(code))

    print("\n" + "=" * 60)
    print(f"{name.upper()} DATASET")
    print("=" * 60)

    print(f"Samples: {len(lengths)}")
    print(f"Minimum characters: {min(lengths)}")
    print(f"Maximum characters: {max(lengths)}")
    print(f"Average characters: {sum(lengths) / len(lengths):.2f}")

    print("\nCharacter length percentiles:")
    print(f"50th percentile: {percentile(lengths, 0.50)}")
    print(f"75th percentile: {percentile(lengths, 0.75)}")
    print(f"90th percentile: {percentile(lengths, 0.90)}")
    print(f"95th percentile: {percentile(lengths, 0.95)}")
    print(f"99th percentile: {percentile(lengths, 0.99)}")


def main():
    print("=" * 60)
    print("CODE FUNCTION LENGTH ANALYSIS")
    print("=" * 60)

    for name, path in FILES.items():
        analyze_file(name, path)


if __name__ == "__main__":
    main()
