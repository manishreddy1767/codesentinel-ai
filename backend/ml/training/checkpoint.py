"""
Checkpoint save/load and a sampler that makes mid-epoch resume exact.

Two checkpoint kinds are written:

  data/checkpoints/latest_checkpoint.pt
      Full training state, written every N steps and at every epoch end.
      Used to resume an interrupted run.

  data/checkpoints/best_codebert.pt
      Model weights of the best validation F1 so far, plus the validation-tuned
      decision threshold and the metrics that justified it. Used for final
      test evaluation and deployment.

Mid-epoch reproducibility
-------------------------
A plain ``DataLoader(shuffle=True)`` cannot be resumed exactly: the shuffle is
driven by a global RNG whose state at the start of the interrupted epoch is
not recoverable, so "resume" silently reshuffles and replays samples the model
already saw. ``ResumableSampler`` fixes this by deriving each epoch's
permutation from a fixed ``(seed, epoch)`` pair, which makes the order a pure
function of two saved integers. On resume the same permutation is regenerated
and the first ``skip_batches * batch_size`` indices are skipped, so training
continues on exactly the samples it had not reached yet.
"""

from pathlib import Path

import torch
from torch.utils.data import Sampler

from backend.ml import config


class ResumableSampler(Sampler):
    """
    Deterministic shuffling sampler supporting exact mid-epoch resume.

    ``set_epoch`` selects the permutation; ``set_skip`` drops a prefix of it.
    """

    def __init__(self, data_source, seed: int = config.SEED, shuffle: bool = True):
        self.data_source = data_source
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip(self, skip: int) -> None:
        """Skip the first ``skip`` *samples* of this epoch's order."""

        self.skip = max(0, skip)

    def __iter__(self):
        size = len(self.data_source)

        if self.shuffle:
            generator = torch.Generator()
            # Permutation depends only on (seed, epoch) -> recoverable.
            generator.manual_seed(self.seed * 1_000_003 + self.epoch)
            order = torch.randperm(size, generator=generator).tolist()
        else:
            order = list(range(size))

        if self.skip:
            order = order[self.skip:]

        return iter(order)

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.skip)


def save_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: int = 0,
    global_step: int = 0,
    batch_in_epoch: int = 0,
    best_f1: float = -1.0,
    best_threshold: float = config.DEFAULT_THRESHOLD,
    epochs_without_improvement: int = 0,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Atomically write a checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "batch_in_epoch": batch_in_epoch,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "epochs_without_improvement": epochs_without_improvement,
        "metrics": metrics or {},
        "config": {
            "model_name": config.MODEL_NAME,
            "max_length": config.MAX_LENGTH,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "max_chunks": config.MAX_CHUNKS_PER_FUNCTION,
            "chunk_pooling": config.CHUNK_POOLING,
            "function_pooling": config.FUNCTION_POOLING,
            "seed": config.SEED,
        },
        "extra": extra or {},
    }

    # Write to a temp file first so an interrupted save cannot corrupt the
    # only good checkpoint on disk.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)

    return path


def load_checkpoint(
    path: Path,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
) -> dict:
    """
    Restore a checkpoint into the supplied objects and return its payload.

    Raises FileNotFoundError if the checkpoint is absent, so a caller can
    never mistakenly believe it resumed when it actually started fresh.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])

    return payload


def describe_checkpoint(payload: dict) -> str:
    return (
        f"epoch={payload.get('epoch')}  "
        f"batch_in_epoch={payload.get('batch_in_epoch')}  "
        f"global_step={payload.get('global_step')}  "
        f"best_f1={payload.get('best_f1'):.4f}  "
        f"best_threshold={payload.get('best_threshold'):.4f}  "
        f"epochs_without_improvement={payload.get('epochs_without_improvement')}"
    )
