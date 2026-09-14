"""
Integration tests for the CodeSentinel evaluation pipeline.
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

from backend.ml.evaluation.evaluator import (
    evaluate_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

VALID_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunked"
    / "primevul_valid_chunked.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"


def test_evaluation_pipeline():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = PrimeVulChunkedDataset(
        VALID_FILE
    )

    subset = Subset(
        dataset,
        list(range(16)),
    )

    loader = DataLoader(
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

    metrics = evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }

    assert set(
        metrics.keys()
    ) == expected_metrics

    # --------------------------------------------------
    # VERIFY METRIC RANGES
    # --------------------------------------------------

    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):

        assert 0.0 <= metrics[
            metric_name
        ] <= 1.0

    # --------------------------------------------------
    # CONFUSION MATRIX MUST MATCH SAMPLE COUNT
    # --------------------------------------------------

    total_predictions = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )

    assert total_predictions == len(subset)
