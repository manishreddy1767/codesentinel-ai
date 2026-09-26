from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from backend.ml.preprocessing.loader import (
    PrimeVulChunkedDataset,
)

from backend.ml.models.training_collate import (
    collate_training_batch,
)

from backend.ml.models.codebert_classifier import (
    CodeBERTClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunked"
    / "primevul_train_chunked.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"


def test_model_forward_pass():

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

    model = CodeBERTClassifier(
        model_name=MODEL_NAME
    )

    model.eval()

    batch = next(iter(loader))

    with torch.no_grad():

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch[
                "attention_mask"
            ],
            chunk_counts=batch[
                "chunk_counts"
            ],
        )

    assert logits.shape == (4,)

    assert torch.isfinite(
        logits
    ).all()

    probabilities = torch.sigmoid(
        logits
    )

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )
