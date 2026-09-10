from pathlib import Path
import json
from typing import Iterator, Dict, Any


def load_jsonl(file_path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    Load records from a JSONL file one at a time.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                yield record

            except json.JSONDecodeError as error:
                print(
                    f"Skipping invalid JSON "
                    f"at line {line_number}: {error}"
                )


def count_records(file_path: str | Path) -> int:
    """
    Count valid records in a JSONL dataset.
    """

    return sum(1 for _ in load_jsonl(file_path))
