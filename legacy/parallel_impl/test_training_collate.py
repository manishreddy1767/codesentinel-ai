from pathlib import Path
from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from backend.ml.preprocessing.loader import (
    PrimeVulChunkedDataset,
)

from backend.ml.models.training_collate import (
    collate_training_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunked"
    / "primevul_train_chunked.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"


def test_training_collate():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = PrimeVulChunkedDataset(
        DATASET_PATH
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=partial(
            collate_training_batch,
            tokenizer=tokenizer,
        ),
    )

    batch = next(iter(loader))

    assert batch["input_ids"].dim() == 2

    assert (
        batch["attention_mask"].shape
        == batch["input_ids"].shape
    )

    assert len(
        batch["chunk_counts"]
    ) == 4

    assert len(
        batch["targets"]
    ) == 4

    total_chunks = (
        batch["chunk_counts"].sum().item()
    )

    assert (
        total_chunks
        == batch["input_ids"].shape[0]
    )
