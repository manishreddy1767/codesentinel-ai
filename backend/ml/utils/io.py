"""JSONL helpers shared by every stage of the ML pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple


def read_jsonl(
    path: str | Path,
    skip_invalid: bool = True,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    Yield ``(line_number, record)`` pairs from a JSONL file.

    Blank lines are ignored. Malformed lines are skipped when
    ``skip_invalid`` is set, otherwise they raise.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records as JSONL, creating parent directories. Returns the count."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written


def count_lines(path: str | Path) -> int:
    """Count non-blank lines without parsing JSON."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def detect_field(record: Dict[str, Any], candidates: Iterable[str]) -> str | None:
    """Return the first candidate key present in ``record``."""

    for candidate in candidates:
        if candidate in record:
            return candidate

    return None
