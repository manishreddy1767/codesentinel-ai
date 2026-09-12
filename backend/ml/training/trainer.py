"""
Training loop for the hierarchical CodeBERT vulnerability classifier.

Responsibilities:
  * AdamW + linear-warmup schedule, gradient clipping, mixed precision
  * BCEWithLogitsLoss with a pos_weight derived from the training split
  * per-epoch validation, F1-based early stopping, best-model selection
  * validation-only threshold optimisation
  * periodic checkpointing and exact mid-epoch resume
  * bounded, observable GPU memory
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.ml import config
from backend.ml.training import checkpoint as ckpt
from backend.ml.training.metrics import compute_metrics, find_best_threshold
from backend.ml.utils.gpu import memory_report, reset_peak_memory


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        collate_fn,
        device,
        pos_weight: float | None = None,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        eval_batch_size: int = config.EVAL_BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        warmup_ratio: float = config.WARMUP_RATIO,
        max_grad_norm: float = config.MAX_GRAD_NORM,
        grad_accum_steps: int = config.GRADIENT_ACCUMULATION_STEPS,
        use_amp: bool = config.USE_AMP,
        num_workers: int = config.NUM_WORKERS,
        checkpoint_dir: Path = config.CHECKPOINT_DIR,
        checkpoint_every: int = config.CHECKPOINT_EVERY,
        log_every: int = config.LOG_EVERY,
        gpu_log_every: int = config.GPU_LOG_EVERY,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        seed: int = config.SEED,
        high_loss_threshold: float = config.HIGH_LOSS_THRESHOLD,
    ):
        self.model = model.to(device)
        self.device = device

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.num_workers = num_workers

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        self.best_path = self.checkpoint_dir / "best_codebert.pt"

        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.gpu_log_every = gpu_log_every
        self.patience = patience
        self.seed = seed
        self.high_loss_threshold = high_loss_threshold

        # AMP only makes sense on CUDA; on CPU it is disabled outright so the
        # same code path runs unchanged on a laptop.
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # --- imbalance handling -------------------------------------------
        if pos_weight is None:
            pos_weight = train_dataset.pos_weight()

        self.pos_weight_value = float(pos_weight)

        # reduction="none" so per-sample losses are available for the
        # high-loss diagnostics; the mean is taken explicitly below.
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_weight_value, device=device),
            reduction="none",
        )

        # --- sampler / loaders --------------------------------------------
        self.train_sampler = ckpt.ResumableSampler(
            train_dataset, seed=seed, shuffle=True
        )

        self.steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        self.optimizer_steps_per_epoch = math.ceil(
            self.steps_per_epoch / self.grad_accum_steps
        )

        self.optimizer = self._build_optimizer(learning_rate, weight_decay)
        self.scheduler = self._build_scheduler(warmup_ratio)

        # --- run state ------------------------------------------------------
        self.start_epoch = 0
        self.start_batch = 0
        self.global_step = 0
        self.best_f1 = -1.0
        self.best_threshold = config.DEFAULT_THRESHOLD
        self.epochs_without_improvement = 0
        self.history = []

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self, learning_rate: float, weight_decay: float):
        # Biases and LayerNorm weights are conventionally excluded from decay.
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

        decay_params = []
        no_decay_params = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            if any(token in name for token in no_decay):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )

    def _build_scheduler(self, warmup_ratio: float):
        from transformers import get_linear_schedule_with_warmup

        total_steps = max(1, self.optimizer_steps_per_epoch * self.epochs)
        warmup_steps = int(total_steps * warmup_ratio)

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _train_loader(self, epoch: int, skip_batches: int = 0):
        self.train_sampler.set_epoch(epoch)
        self.train_sampler.set_skip(skip_batches * self.batch_size)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _eval_loader(self, dataset):
        # No shuffling, no resampling: evaluation sees the split as-is.
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

    def _to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def maybe_resume(self, resume: bool = True) -> bool:
        """
        Restore full training state from the latest checkpoint.

        Returns True only if state was genuinely restored. When it returns
        False the run really is starting from scratch.
        """

        if not resume or not self.latest_path.exists():
            print("No checkpoint restored - starting from scratch.")
            return False

        payload = ckpt.load_checkpoint(
            self.latest_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )

        self.start_epoch = payload.get("epoch", 0)
        self.start_batch = payload.get("batch_in_epoch", 0)
        self.global_step = payload.get("global_step", 0)
        self.best_f1 = payload.get("best_f1", -1.0)
        self.best_threshold = payload.get("best_threshold", config.DEFAULT_THRESHOLD)
        self.epochs_without_improvement = payload.get(
            "epochs_without_improvement", 0
        )

        print("RESUMED from checkpoint:")
        print("  " + ckpt.describe_checkpoint(payload))
        print(
            f"  Skipping the first {self.start_batch} batches of epoch "
            f"{self.start_epoch + 1} using the (seed, epoch) permutation."
        )

        return True

    def _save_latest(self, epoch: int, batch_in_epoch: int) -> None:
        ckpt.save_checkpoint(
            self.latest_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            batch_in_epoch=batch_in_epoch,
            best_f1=self.best_f1,
            best_threshold=self.best_threshold,
            epochs_without_improvement=self.epochs_without_improvement,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _report_high_loss(
        self,
        epoch: int,
        batch_index: int,
        per_sample_loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_chunks: torch.Tensor,
    ) -> None:
        """
        Print details for samples whose loss exceeds the threshold.

        Purely informational. A confidently wrong prediction *should* have a
        high loss, so this never stops or alters training.
        """

        offenders = (per_sample_loss > self.high_loss_threshold).nonzero()

        if offenders.numel() == 0:
            return

        offenders = offenders.flatten()[: config.HIGH_LOSS_MAX_REPORTS]

        probabilities = torch.sigmoid(logits.detach().float())

        print(
            f"  [high-loss] epoch {epoch + 1} batch {batch_index} "
            f"({offenders.numel()} sample(s) over {self.high_loss_threshold})",
            flush=True,
        )

        for position in offenders.tolist():
            print(
                f"    label={int(targets[position].item())}"
                f"  p(vuln)={probabilities[position].item():.4f}"
                f"  logit={logits[position].item():+.4f}"
                f"  loss={per_sample_loss[position].item():.4f}"
                f"  chunks={int(num_chunks[position].item())}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Train / evaluate
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, skip_batches: int = 0) -> dict:
        self.model.train()
        reset_peak_memory()

        loader = self._train_loader(epoch, skip_batches)

        running_loss = 0.0
        seen_batches = 0
        started = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for offset, batch in enumerate(loader):
            batch_index = skip_batches + offset

            batch = self._to_device(batch)
            targets = batch["targets"]
            batch_size = targets.size(0)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    function_index=batch["function_index"],
                    batch_size=batch_size,
                )

                # The contract the whole pipeline depends on.
                assert logits.shape == targets.shape, (
                    f"logits {tuple(logits.shape)} != targets {tuple(targets.shape)}"
                )

                per_sample_loss = self.criterion(logits, targets)
                loss = per_sample_loss.mean()

            loss_value = loss.item()

            # Only genuinely unrecoverable numerics abort the run.
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite loss ({loss_value}) at epoch {epoch + 1} "
                    f"batch {batch_index}. Aborting."
                )

            if self.high_loss_threshold > 0:
                self._report_high_loss(
                    epoch,
                    batch_index,
                    per_sample_loss.detach(),
                    logits.detach(),
                    targets,
                    batch["num_chunks"],
                )

            self.scaler.scale(loss / self.grad_accum_steps).backward()

            is_step_boundary = (offset + 1) % self.grad_accum_steps == 0
            is_last_batch = offset + 1 == len(loader)

            if is_step_boundary or is_last_batch:
                # Unscale before clipping so max_grad_norm means what it says.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            # Accumulate the float, not the tensor: keeping the tensor would
            # retain its autograd graph and leak memory across the epoch.
            running_loss += loss_value
            seen_batches += 1

            if self.log_every and batch_index % self.log_every == 0:
                average = running_loss / max(1, seen_batches)
                learning_rate = self.scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch + 1} | batch {batch_index}/{self.steps_per_epoch}"
                    f" | loss {loss_value:.4f} | avg {average:.4f}"
                    f" | lr {learning_rate:.2e}"
                    f" | chunks {batch['input_ids'].size(0)}",
                    flush=True,
                )

            if self.gpu_log_every and batch_index % self.gpu_log_every == 0:
                print("  " + memory_report(), flush=True)

            if (
                self.checkpoint_every
                and batch_index > 0
                and batch_index % self.checkpoint_every == 0
            ):
                self._save_latest(epoch, batch_index + 1)
                print(
                    f"  checkpoint saved at epoch {epoch + 1} batch {batch_index}",
                    flush=True,
                )

        elapsed = time.time() - started

        return {
            "loss": running_loss / max(1, seen_batches),
            "batches": seen_batches,
            "seconds": elapsed,
        }

    @torch.no_grad()
    def evaluate(self, dataset, name: str = "valid") -> tuple:
        """
        Run inference over a dataset and return ``(probabilities, targets)``.

        Strictly read-only: eval mode, no grad, no resampling, no augmentation.
        """

        self.model.eval()

        loader = self._eval_loader(dataset)

        all_probabilities = []
        all_targets = []

        for batch in loader:
            batch = self._to_device(batch)
            batch_size = batch["targets"].size(0)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    function_index=batch["function_index"],
                    batch_size=batch_size,
                )

            # float() before sigmoid so fp16 autocast cannot saturate the
            # probabilities used for AUC and threshold search.
            probabilities = torch.sigmoid(logits.float())

            all_probabilities.append(probabilities.detach().cpu().numpy())
            all_targets.append(batch["targets"].detach().cpu().numpy())

        probabilities = np.concatenate(all_probabilities)
        targets = np.concatenate(all_targets).astype(np.int64)

        return probabilities, targets

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

        negatives, positives = self.train_dataset.class_counts()
        print(f"  train functions     : {len(self.train_dataset)}")
        print(f"  train chunks        : {self.train_dataset.total_chunks()}")
        print(f"  train vulnerable    : {positives}")
        print(f"  train benign        : {negatives}")
        print(f"  pos_weight          : {self.pos_weight_value:.4f}")
        print(f"  valid functions     : {len(self.valid_dataset)}")
        print(f"  device              : {self.device}")
        print(f"  mixed precision     : {self.use_amp}")
        print(f"  batches/epoch       : {self.steps_per_epoch}")
        print("  " + memory_report())

        for epoch in range(self.start_epoch, self.epochs):
            skip = self.start_batch if epoch == self.start_epoch else 0

            print("\n" + "-" * 70)
            print(f"EPOCH {epoch + 1}/{self.epochs}")
            print("-" * 70)

            stats = self.train_epoch(epoch, skip_batches=skip)

            print(
                f"\n  train loss: {stats['loss']:.4f}"
                f"  ({stats['batches']} batches in {stats['seconds']:.1f}s)"
            )
            print("  " + memory_report())

            # ---- validation ------------------------------------------------
            probabilities, targets = self.evaluate(self.valid_dataset, "valid")

            default_metrics = compute_metrics(
                probabilities, targets, config.DEFAULT_THRESHOLD
            )
            print("\n" + default_metrics.format("VALIDATION @ threshold 0.5"))

            tuned_threshold, tuned_metrics, _ = find_best_threshold(
                probabilities, targets
            )
            print(
                "\n"
                + tuned_metrics.format(
                    f"VALIDATION @ tuned threshold {tuned_threshold:.3f}"
                )
            )

            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": stats["loss"],
                    "valid_f1_at_0.5": default_metrics.f1,
                    "valid_f1_tuned": tuned_metrics.f1,
                    "tuned_threshold": tuned_threshold,
                }
            )

            # ---- best model selection / early stopping ---------------------
            improved = tuned_metrics.f1 > self.best_f1

            if improved:
                self.best_f1 = tuned_metrics.f1
                self.best_threshold = tuned_threshold
                self.epochs_without_improvement = 0

                ckpt.save_checkpoint(
                    self.best_path,
                    self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_f1=self.best_f1,
                    best_threshold=self.best_threshold,
                    metrics=tuned_metrics.to_dict(),
                    extra={"pos_weight": self.pos_weight_value},
                )
                print(
                    f"\n  NEW BEST validation F1 {self.best_f1:.4f} "
                    f"(threshold {self.best_threshold:.3f}) -> {self.best_path}"
                )
            else:
                self.epochs_without_improvement += 1
                print(
                    f"\n  No improvement ({tuned_metrics.f1:.4f} <= "
                    f"{self.best_f1:.4f}). "
                    f"Patience {self.epochs_without_improvement}/{self.patience}"
                )

            # batch_in_epoch=0 marks the epoch as finished, so a resume from
            # here starts cleanly at the next epoch.
            self._save_latest(epoch + 1, 0)
            self.start_batch = 0

            if self.epochs_without_improvement >= self.patience:
                print(
                    f"\nEARLY STOPPING: no validation F1 improvement for "
                    f"{self.patience} epoch(s)."
                )
                break

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  best validation F1 : {self.best_f1:.4f}")
        print(f"  selected threshold : {self.best_threshold:.4f}")
        print(f"  best checkpoint    : {self.best_path}")

        return {
            "best_f1": self.best_f1,
            "best_threshold": self.best_threshold,
            "history": self.history,
            "best_checkpoint": str(self.best_path),
        }
