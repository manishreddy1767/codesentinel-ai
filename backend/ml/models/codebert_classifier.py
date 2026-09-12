"""
Hierarchical CodeBERT vulnerability classifier.

    source function
        -> code tokenization          (chunk_dataset.py)
        -> code chunks                [N, L]
        -> CodeBERT                   [N, L, H]
        -> chunk embeddings           [N, H]
        -> chunk aggregation          scatter-mean by function_index
        -> function embedding         [B, H]
        -> classification head        Linear(H, 1)
        -> vulnerability logit        [B]

The head returns RAW LOGITS of shape [B] - one per function - so that
``logits.shape == targets.shape`` and BCEWithLogitsLoss can be applied
directly. Sigmoid is never applied inside the model.
"""

import torch
import torch.nn as nn

from backend.ml import config


class HierarchicalCodeBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        chunk_pooling: str = config.CHUNK_POOLING,
        function_pooling: str = config.FUNCTION_POOLING,
        dropout: float = config.CLASSIFIER_DROPOUT,
        gradient_checkpointing: bool = False,
        chunk_micro_batch: int = config.CHUNK_MICRO_BATCH,
        encoder=None,
    ):
        super().__init__()

        from transformers import AutoConfig, AutoModel

        if chunk_pooling not in ("cls", "mean"):
            raise ValueError(f"Unknown chunk_pooling: {chunk_pooling}")

        if function_pooling not in ("mean", "max"):
            raise ValueError(f"Unknown function_pooling: {function_pooling}")

        self.model_name = model_name
        self.chunk_pooling = chunk_pooling
        self.function_pooling = function_pooling
        self.chunk_micro_batch = max(1, chunk_micro_batch)

        if encoder is not None:
            # Injection seam: lets tests exercise the full forward/loss/backward
            # path with a tiny randomly-initialised encoder and no model download.
            self.encoder = encoder
            self.encoder_config = encoder.config
        else:
            self.encoder_config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(
                model_name, config=self.encoder_config
            )

        hidden_size = self.encoder_config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Memory controls
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        # use_reentrant=False is required here: with the reentrant autograd
        # path, checkpointing silently produces no gradients when the inputs
        # to the checkpointed block do not require grad (which is the case
        # for token ids).
        self.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def disable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing_disable()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CodeBERT over [N, L] chunks and return [N, H] chunk embeddings.

        Chunks are pushed through in micro-batches so that peak activation
        memory is bounded by ``chunk_micro_batch`` rather than by however many
        chunks the batch happened to expand into. Results are concatenated,
        so the autograd graph is preserved and gradients still flow.
        """

        total = input_ids.size(0)
        step = self.chunk_micro_batch

        embeddings = []

        for start in range(0, total, step):
            stop = min(start + step, total)

            outputs = self.encoder(
                input_ids=input_ids[start:stop],
                attention_mask=attention_mask[start:stop],
            )

            hidden = outputs.last_hidden_state  # [n, L, H]

            if self.chunk_pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask[start:stop].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                # clamp_min guards the impossible-but-cheap all-padding case.
                counts = mask.sum(dim=1).clamp_min(1.0)
                pooled = summed / counts

            embeddings.append(pooled)

        return torch.cat(embeddings, dim=0)

    def _aggregate(
        self,
        chunk_embeddings: torch.Tensor,
        function_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Scatter-aggregate [N, H] chunk embeddings into [B, H]."""

        hidden_size = chunk_embeddings.size(-1)

        if self.function_pooling == "mean":
            summed = torch.zeros(
                batch_size,
                hidden_size,
                dtype=chunk_embeddings.dtype,
                device=chunk_embeddings.device,
            )
            summed.index_add_(0, function_index, chunk_embeddings)

            counts = torch.zeros(
                batch_size,
                dtype=chunk_embeddings.dtype,
                device=chunk_embeddings.device,
            )
            counts.index_add_(
                0,
                function_index,
                torch.ones_like(function_index, dtype=chunk_embeddings.dtype),
            )

            # Every function has >= 1 chunk, so counts is never 0; clamp_min
            # only protects against a malformed function_index.
            return summed / counts.clamp_min(1.0).unsqueeze(-1)

        pooled = torch.full(
            (batch_size, hidden_size),
            float("-inf"),
            dtype=chunk_embeddings.dtype,
            device=chunk_embeddings.device,
        )
        expanded = function_index.unsqueeze(-1).expand(-1, hidden_size)
        pooled = pooled.scatter_reduce(
            0, expanded, chunk_embeddings, reduce="amax", include_self=True
        )

        return torch.nan_to_num(pooled, neginf=0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        function_index: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Returns raw logits of shape [B] - one scalar per function.

        Args:
            input_ids:      [N, L]
            attention_mask: [N, L]
            function_index: [N], values in [0, B)
            batch_size:     B. Inferred from function_index when omitted, but
                            pass it explicitly to stay correct in the edge
                            case where the final functions of a batch
                            contribute no chunks.
        """

        if batch_size is None:
            batch_size = int(function_index.max().item()) + 1

        chunk_embeddings = self._encode_chunks(input_ids, attention_mask)

        function_embeddings = self._aggregate(
            chunk_embeddings, function_index, batch_size
        )

        pooled = self.dropout(function_embeddings)

        # [B, 1] -> [B] so the shape matches the [B] target vector.
        return self.classifier(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Sigmoid of the logits. Only for inference - never feed this to the loss."""

        return torch.sigmoid(self.forward(*args, **kwargs))
