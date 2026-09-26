"""
Model evaluation utilities for CodeSentinel.
"""

import torch

from backend.ml.training.metrics import calculate_metrics


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    """
    Evaluate the model on a dataset.

    Args:
        model: CodeBERT classifier model.
        dataloader: PyTorch DataLoader.
        device: CPU or CUDA device.
        threshold: Classification threshold.

    Returns:
        Dictionary containing evaluation metrics.
    """

    model.eval()

    all_targets = []
    all_probabilities = []

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        chunk_counts = batch[
            "chunk_counts"
        ].to(device)

        targets = batch["targets"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_counts=chunk_counts,
        )

        probabilities = torch.sigmoid(logits)

        all_targets.extend(
            targets.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

<<<<<<< ours
    return calculate_metrics(
        targets=all_targets,
        probabilities=all_probabilities,
        threshold=threshold,
    )
=======
    model = HierarchicalCodeBERTClassifier(
        model_name=saved.get("model_name", config.MODEL_NAME),
        chunk_pooling=saved.get("chunk_pooling", config.CHUNK_POOLING),
        function_pooling=saved.get("function_pooling", config.FUNCTION_POOLING),
        dropout=saved.get("dropout", config.CLASSIFIER_DROPOUT),
        gradient_checkpointing=False,
        chunk_micro_batch=chunk_micro_batch or config.CHUNK_MICRO_BATCH,
    )

    # strict=True: a checkpoint whose recorded architecture disagrees with its
    # weights must fail loudly here rather than evaluate a half-loaded model.
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)

    return model, payload
>>>>>>> theirs
