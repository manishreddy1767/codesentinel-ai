"""
Integration tests for the CodeBERT training pipeline.
"""

from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
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

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunked"
    / "primevul_train_chunked.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"


def test_training_step():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = PrimeVulChunkedDataset(
        TRAIN_FILE
    )

    subset = Subset(
        dataset,
        list(range(8)),
    )

    dataloader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        collate_fn=partial(
            collate_training_batch,
            tokenizer=tokenizer,
        ),
    )

    model = CodeBERTClassifier(
        model_name=MODEL_NAME
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    batch = next(iter(dataloader))

    input_ids = batch[
        "input_ids"
    ].to(device)

    attention_mask = batch[
        "attention_mask"
    ].to(device)

    chunk_counts = batch[
        "chunk_counts"
    ].to(device)

    targets = batch[
        "targets"
    ].to(device)

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_counts=chunk_counts,
    )

    # One output logit per function.
    assert logits.shape == targets.shape

    loss = criterion(
        logits,
        targets,
    )

    # Loss must be finite.
    assert torch.isfinite(loss)

    loss.backward()

    # --------------------------------------------------
    # VERIFY GRADIENTS
    # --------------------------------------------------

    has_gradient = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            has_gradient = True

            assert torch.isfinite(
                parameter.grad
            ).all()

    assert has_gradient

    # --------------------------------------------------
    # VERIFY OPTIMIZER UPDATES WEIGHTS
    # --------------------------------------------------

    parameter_before = next(
        model.parameters()
    ).detach().clone()

    optimizer.step()

    parameter_after = next(
        model.parameters()
    ).detach()

    assert not torch.equal(
        parameter_before,
        parameter_after,
    )
