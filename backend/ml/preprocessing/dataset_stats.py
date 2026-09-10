import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/primevul_dataset")

FILES = [
    "primevul_train.jsonl",
    "primevul_valid.jsonl",
    "primevul_test.jsonl",
]

for filename in FILES:
    path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    total = 0
    targets = Counter()
    cwes = Counter()
    projects = Counter()
    empty_functions = 0
    invalid_json = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)

                total += 1

                target = record.get("target")
                targets[target] += 1

                cwe = record.get("cwe", [])

                if isinstance(cwe, str):
                    cwe_values = [cwe.strip()] if cwe.strip() else []

                elif isinstance(cwe, list):
                    cwe_values = [
                        str(item).strip()
                        for item in cwe
                        if str(item).strip()
                    ]

                else:
                    cwe_values = []

                for cwe_value in cwe_values:
                    cwes[cwe_value] += 1

                project = record.get("project", "")
                if project:
                    projects[str(project)] += 1

                func = record.get("func", "")
                if not isinstance(func, str) or not func.strip():
                    empty_functions += 1

            except json.JSONDecodeError:
                invalid_json += 1

    print(f"\nTotal records: {total}")
    print(f"Vulnerable (target=1): {targets[1]}")
    print(f"Benign (target=0): {targets[0]}")

    if total > 0:
        vulnerable_percent = targets[1] / total * 100
        benign_percent = targets[0] / total * 100

        print(f"Vulnerable percentage: {vulnerable_percent:.2f}%")
        print(f"Benign percentage: {benign_percent:.2f}%")

    print(f"\nEmpty functions: {empty_functions}")
    print(f"Invalid JSON lines: {invalid_json}")
    print(f"Unique projects: {len(projects)}")
    print(f"Unique CWEs: {len(cwes)}")

    print("\nTop 10 CWEs:")
    for cwe, count in cwes.most_common(10):
        print(f"{cwe}: {count}")

    print("\nTop 10 projects:")
    for project, count in projects.most_common(10):
        print(f"{project}: {count}")
