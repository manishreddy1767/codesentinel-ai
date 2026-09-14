import json
from pathlib import Path

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/autotrain")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FILES = {
    "primevul_train_clean.jsonl": "train.jsonl",
    "primevul_valid_clean.jsonl": "valid.jsonl",
    "primevul_test_clean.jsonl": "test.jsonl",
}

for input_name, output_name in FILES.items():

    input_path = INPUT_DIR / input_name
    output_path = OUTPUT_DIR / output_name

    count = 0

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as infile, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as outfile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            sample = json.loads(line)

            text = sample.get(
                "func",
                "",
            )

            target = sample.get(
                "target",
            )

            if (
                not text
                or target not in [0, 1]
            ):
                continue

            output_sample = {
                "text": text,
                "target": str(target),
            }

            outfile.write(
                json.dumps(
                    output_sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print(
        f"{output_name}: "
        f"{count} samples"
    )

print("\nAutoTrain dataset preparation complete.")
