import json
from pathlib import Path

DATA_DIR = Path("data/primevul_dataset")

FILES = [
    "primevul_train.jsonl",
    "primevul_valid.jsonl",
    "primevul_test.jsonl",
]

for filename in FILES:
    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("=" * 60)

    count = 0
    first_record = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                count += 1

                if first_record is None:
                    first_record = record

    print("Total records:", count)

    if first_record:
        print("\nColumns:")
        print(list(first_record.keys()))

        print("\nFirst record:")
        for key, value in first_record.items():
            print(f"{key}: {str(value)[:500]}")
