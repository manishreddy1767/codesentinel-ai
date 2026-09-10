import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "primevul_train.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "primevul_train_balanced.jsonl"
)

RANDOM_SEED = 42
BENIGN_TO_VULNERABLE_RATIO = 5


def load_records():
    vulnerable = []
    benign = []

    print("Loading training dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["target"] == 1:
                vulnerable.append(record)
            else:
                benign.append(record)

    return vulnerable, benign


def main():
    random.seed(RANDOM_SEED)

    vulnerable, benign = load_records()

    print(f"\nVulnerable samples: {len(vulnerable)}")
    print(f"Benign samples: {len(benign)}")

    required_benign = len(vulnerable) * BENIGN_TO_VULNERABLE_RATIO

    if required_benign > len(benign):
        required_benign = len(benign)

    selected_benign = random.sample(benign, required_benign)

    balanced_records = vulnerable + selected_benign

    random.shuffle(balanced_records)

    print("\nBalanced dataset:")
    print(f"Vulnerable: {len(vulnerable)}")
    print(f"Benign: {len(selected_benign)}")
    print(f"Total: {len(balanced_records)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in balanced_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"\nSaved balanced dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
