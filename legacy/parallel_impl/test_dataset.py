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


def test_dataset_loading():

    dataset = PrimeVulChunkedDataset(
        DATASET_PATH
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert "target" in sample
    assert "chunks" in sample
    assert "num_chunks" in sample

    assert len(sample["chunks"]) > 0

    assert (
        sample["num_chunks"]
        == len(sample["chunks"])
    )


def test_dataloader_collation():

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

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "chunk_counts" in batch
    assert "targets" in batch

    assert len(batch["targets"]) == 4

    assert (
        batch["chunk_counts"].sum().item()
        == batch["input_ids"].shape[0]
    )
