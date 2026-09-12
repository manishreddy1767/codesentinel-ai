"""
Reusable evaluation routine shared by validation and final testing.

Kept separate from Trainer so that evaluating a saved checkpoint never has to
construct an optimizer, a scheduler or a training dataset.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.ml import config
from backend.ml.training.metrics import compute_metrics


@torch.no_grad()
def predict(
    model,
    dataset,
    collate_fn,
    device,
    batch_size: int = config.EVAL_BATCH_SIZE,
    use_amp: bool | None = None,
    num_workers: int = 0,
    progress_every: int = 50,
):
    """
    Run inference and return ``(probabilities, targets)`` as numpy arrays.

    Evaluation invariants: ``model.eval()``, ``torch.no_grad()``, no shuffling,
    no resampling, no augmentation. The dataset is consumed exactly as given.
    """

    if use_amp is None:
        use_amp = config.USE_AMP and device.type == "cuda"
    else:
        use_amp = use_amp and device.type == "cuda"

    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    all_probabilities = []
    all_targets = []

    for index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        function_count = batch["targets"].size(0)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                function_index=batch["function_index"],
                batch_size=function_count,
            )

        # Upcast before sigmoid so fp16 does not saturate probabilities.
        probabilities = torch.sigmoid(logits.float())

        all_probabilities.append(probabilities.cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())

        if progress_every and index and index % progress_every == 0:
            print(f"    ...{index * batch_size} functions")

    return (
        np.concatenate(all_probabilities),
        np.concatenate(all_targets).astype(np.int64),
    )


def evaluate_at_threshold(probabilities, targets, threshold: float, title: str):
    metrics = compute_metrics(probabilities, targets, threshold)
    print("\n" + metrics.format(title))
    return metrics


def load_model_from_checkpoint(path, device, chunk_micro_batch=None):
    """
    Rebuild the classifier and load weights from a checkpoint.

    The architecture settings recorded in the checkpoint take precedence over
    the current config, so evaluating an old checkpoint cannot silently use a
    different pooling strategy than the one it was trained with.
    """

    from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})

    model = HierarchicalCodeBERTClassifier(
        model_name=saved.get("model_name", config.MODEL_NAME),
        chunk_pooling=saved.get("chunk_pooling", config.CHUNK_POOLING),
        function_pooling=saved.get("function_pooling", config.FUNCTION_POOLING),
        gradient_checkpointing=False,
        chunk_micro_batch=chunk_micro_batch or config.CHUNK_MICRO_BATCH,
    )

    model.load_state_dict(payload["model_state"])
    model.to(device)

    return model, payload
