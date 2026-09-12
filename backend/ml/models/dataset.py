"""
Torch Dataset over the chunked PrimeVul files.

One dataset item is one FUNCTION, carrying a variable number of chunks.
The collate function flattens the batch; the model re-groups chunks by
function. Labels are attached per function, never per chunk, so the loss is
computed once per function exactly as the task requires.

Memory
------
The full PrimeVul training split is ~184k functions averaging several 512-token
chunks each. Holding those token ids as Python lists costs 28 bytes per int and
runs to several GB before training even starts, so by default the dataset only
indexes byte offsets and parses each record on access. ``in_memory=True``
restores eager loading, which is worth it for small splits and for tests.
"""

import json
from pathlib import Path

from torch.utils.data import Dataset

from backend.ml import config


class ChunkedFunctionDataset(Dataset):
    """Reads a ``*_chunked.jsonl`` file produced by chunk_dataset.py."""

    def __init__(
        self,
        path: str | Path,
        max_chunks: int | None = None,
        in_memory: bool = False,
    ):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Chunked dataset not found: {self.path}\n"
                f"Run: python -m backend.ml.preprocessing.chunk_dataset"
            )

        self.max_chunks = max_chunks or config.MAX_CHUNKS_PER_FUNCTION
        self.in_memory = in_memory

        # Always kept in memory: cheap, and needed for class statistics.
        self.targets = []
        self.chunk_counts = []

        self._offsets = []
        self._records = [] if in_memory else None

        # Opened in binary mode so tell() returns real byte offsets that are
        # not perturbed by newline translation on Windows.
        with self.path.open("rb") as handle:
            offset = handle.tell()
            line = handle.readline()

            line_number = 0

            while line:
                line_number += 1
                stripped = line.strip()

                if stripped:
                    record = json.loads(stripped.decode("utf-8"))
                    chunks = record.get("chunks") or []

                    if not chunks:
                        raise ValueError(
                            f"{self.path}:{line_number} has zero chunks. "
                            f"Re-run chunk_dataset.py."
                        )

                    self._offsets.append(offset)
                    self.targets.append(int(record["target"]))
                    self.chunk_counts.append(min(len(chunks), self.max_chunks))

                    if in_memory:
                        self._records.append(self._clip(chunks))

                offset = handle.tell()
                line = handle.readline()

        if not self.targets:
            raise ValueError(f"{self.path} contained no records")

    def _clip(self, chunks):
        """
        Defensive second cap: the file may have been produced with a larger
        MAX_CHUNKS_PER_FUNCTION than the config currently in force.
        """

        return chunks[: self.max_chunks] if len(chunks) > self.max_chunks else chunks

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        if self.in_memory:
            chunks = self._records[index]
        else:
            with self.path.open("rb") as handle:
                handle.seek(self._offsets[index])
                record = json.loads(handle.readline().decode("utf-8"))

            chunks = self._clip(record["chunks"])

        return {
            "chunks": chunks,
            "target": self.targets[index],
            "index": index,
        }

    # ------------------------------------------------------------------
    # Introspection used by the trainer
    # ------------------------------------------------------------------

    def class_counts(self) -> tuple:
        """Return ``(negatives, positives)``."""

        positives = sum(self.targets)
        return len(self.targets) - positives, positives

    def pos_weight(self, cap: float | None = None) -> float:
        """
        n_negative / n_positive, the pos_weight for BCEWithLogitsLoss.

        Computed from this dataset, so it stays correct whether the file is
        the full training split or an undersampled copy.
        """

        negatives, positives = self.class_counts()

        if positives == 0:
            return 1.0

        weight = negatives / positives
        cap = config.MAX_POS_WEIGHT if cap is None else cap

        return min(weight, cap)

    def total_chunks(self) -> int:
        return sum(self.chunk_counts)
