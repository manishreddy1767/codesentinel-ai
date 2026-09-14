"""
Training and validation utilities for the CodeBERT vulnerability classifier.
"""

import torch

from backend.ml.training.metrics import calculate_metrics


def aggregate_chunk_logits(chunk_logits, chunk_counts):
    """
    Aggregate chunk-level logits into function-level logits.

    Each function may contain multiple chunks. The mean logit of all
    chunks belonging to a function is used as the function-level logit.

    Args:
        chunk_logits: Tensor containing logits for all chunks.
        chunk_counts: List containing number of chunks per function.

    Returns:
        Tensor containing one logit per function.
    """

    function_logits = []
    start = 0

    for count in chunk_counts:
        end = start + count

        function_logit = chunk_logits[start:end].mean()

        function_logits.append(function_logit)

        start = end

    return torch.stack(function_logits)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    """
    Train the model for one epoch.

    Returns:
        Average training loss.
    """

    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        chunk_counts = batch["chunk_counts"]
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        chunk_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        function_logits = aggregate_chunk_logits(
            chunk_logits,
            chunk_counts,
        )

        loss = criterion(
            function_logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    average_loss = total_loss / max(total_batches, 1)

    return average_loss


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Validate the model.

    Returns:
        Tuple containing:
        - average validation loss
        - evaluation metrics dictionary
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_targets = []
    all_probabilities = []

    with torch.no_grad():

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chunk_counts = batch["chunk_counts"]
            targets = batch["targets"].to(device)

            chunk_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            function_logits = aggregate_chunk_logits(
                chunk_logits,
                chunk_counts,
            )

            loss = criterion(
                function_logits,
                targets,
            )

            probabilities = torch.sigmoid(function_logits)

            total_loss += loss.item()
            total_batches += 1

            all_targets.extend(
                targets.cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    average_loss = total_loss / max(total_batches, 1)

    metrics = calculate_metrics(
        all_targets,
        all_probabilities,
    )

    return average_loss, metrics
