import json
from pathlib import Path


INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/autotrain")


FILES = {
    "train": INPUT_DIR / "primevul_train_clean.jsonl",
    "validation": INPUT_DIR / "primevul_valid_clean.jsonl",
    "test": INPUT_DIR / "primevul_test_clean.jsonl",
}


def convert_file(input_path, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file, open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        for line in input_file:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            text = item.get("func")
            label = item.get("target")

            if text is None:
                continue

            if label not in [0, 1]:
                continue

            output_item = {
                "text": text,
                "label": label,
            }

            output_file.write(
                json.dumps(output_item)
                + "\n"
            )

            count += 1

    print(
        f"{input_path.name}: "
        f"{count} samples"
    )


def main():

    print("=" * 60)
    print("PREPARING PRIMEVUL FOR HUGGING FACE AUTOTRAIN")
    print("=" * 60)

    for split, input_path in FILES.items():

        output_path = (
            OUTPUT_DIR
            / f"{split}.jsonl"
        )

        print(
            f"\nProcessing {split}..."
        )

        convert_file(
            input_path,
            output_path,
        )

        print(
            f"Saved: {output_path}"
        )

    print("\nCompleted successfully.")


if __name__ == "__main__":
    main()