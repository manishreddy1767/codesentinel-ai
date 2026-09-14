import torch


# Maximum number of chunks allowed per function.
#
# This protects GPU memory when a very large function produces
# an unusually high number of 512-token chunks.
MAX_CHUNKS_PER_FUNCTION = 8


def collate_training_batch(batch, tokenizer):
    """
    Prepare a batch of functions for hierarchical CodeBERT training.

    Large functions are limited to MAX_CHUNKS_PER_FUNCTION
    chunks to prevent GPU memory exhaustion.
    """

    all_chunks = []
    chunk_counts = []
    targets = []

    original_chunk_counts = []

    for item in batch:

        chunks = item["chunks"]

        original_chunk_counts.append(
            len(chunks)
        )

        # Limit extremely large functions.
        if len(chunks) > MAX_CHUNKS_PER_FUNCTION:

            chunks = chunks[
                :MAX_CHUNKS_PER_FUNCTION
            ]

        all_chunks.extend(
            chunks
        )

        chunk_counts.append(
            len(chunks)
        )

        targets.append(
            item["target"]
        )

    # Safety check.
    if not all_chunks:

        raise ValueError(
            "Batch contains no code chunks."
        )

    max_length = max(
        len(chunk)
        for chunk in all_chunks
    )

    padded_chunks = []

    for chunk in all_chunks:

        padding_length = (
            max_length
            - len(chunk)
        )

        padded_chunk = (
            chunk
            + [tokenizer.pad_token_id]
            * padding_length
        )

        padded_chunks.append(
            padded_chunk
        )

    input_ids = torch.tensor(
        padded_chunks,
        dtype=torch.long,
    )

    attention_mask = (
        input_ids
        != tokenizer.pad_token_id
    ).long()

    targets = torch.tensor(
        targets,
        dtype=torch.float,
    )

    chunk_counts = torch.tensor(
        chunk_counts,
        dtype=torch.long,
    )

    original_chunk_counts = torch.tensor(
        original_chunk_counts,
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_counts": chunk_counts,
        "original_chunk_counts":
            original_chunk_counts,
        "targets": targets,
    }
