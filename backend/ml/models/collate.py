"""
Collate function for variable-chunk-count functions.

A batch of B functions expands to N chunks, where N = sum of each function's
chunk count. Rather than padding to a rectangular [B, max_chunks, L] tensor -
which wastes encoder compute on padding chunks - the chunks are flattened to
[N, L] and a ``function_index`` vector of length N records which function each
chunk belongs to. The model uses that vector to scatter chunk embeddings back
into per-function embeddings.

Shapes produced:
    input_ids      [N, L]  long
    attention_mask [N, L]  long
    function_index [N]     long, values in [0, B)
    targets        [B]     float  <- matches the model's [B] logits exactly
    num_chunks     [B]     long
"""

import torch

from backend.ml import config


def make_collate_fn(pad_token_id: int, max_length: int = config.MAX_LENGTH):
    """Build a collate_fn bound to a tokenizer's pad id."""

    def collate(batch):
        all_chunks = []
        function_index = []
        targets = []
        num_chunks = []

        for function_position, item in enumerate(batch):
            chunks = item["chunks"]

            if not chunks:
                raise ValueError(
                    "Encountered a function with zero chunks; "
                    "chunk_dataset.py guarantees at least one."
                )

            for chunk in chunks:
                if not chunk:
                    raise ValueError("Encountered an empty chunk")

                # Truncate defensively so a stale chunked file can never
                # index past CodeBERT's positional embedding table.
                all_chunks.append(chunk[:max_length])
                function_index.append(function_position)

            targets.append(float(item["target"]))
            num_chunks.append(len(chunks))

        batch_max = max(len(chunk) for chunk in all_chunks)

        total = len(all_chunks)
        input_ids = torch.full(
            (total, batch_max), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((total, batch_max), dtype=torch.long)

        for row, chunk in enumerate(all_chunks):
            length = len(chunk)
            input_ids[row, :length] = torch.tensor(chunk, dtype=torch.long)
            attention_mask[row, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "function_index": torch.tensor(function_index, dtype=torch.long),
            # float32 so it can be fed straight to BCEWithLogitsLoss.
            "targets": torch.tensor(targets, dtype=torch.float32),
            "num_chunks": torch.tensor(num_chunks, dtype=torch.long),
        }

    return collate
