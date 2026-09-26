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
from torch import nn
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
        monitor: str = config.EARLY_STOPPING_MONITOR,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        threshold_objective: str = config.THRESHOLD_OBJECTIVE,
        threshold_beta: float = config.THRESHOLD_BETA,
        threshold_min_precision: float = config.THRESHOLD_MIN_PRECISION,
        restore_best: bool = True,
        pin_memory: bool | None = None,
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

        if monitor not in ("pr_auc", "f1", "mcc", "roc_auc"):
            raise ValueError(f"Unknown early-stopping monitor: {monitor}")

        self.monitor = monitor
        self.min_delta = float(min_delta)
        self.threshold_objective = threshold_objective
        self.threshold_beta = threshold_beta
        self.threshold_min_precision = threshold_min_precision
        self.restore_best = restore_best

        # Pinned (page-locked) host memory speeds host->device copies, but it
        # is a scarce OS resource and PyTorch's host caching allocator does not
        # reliably release it. On this project's hardware it has now aborted
        # two separate long runs from inside its own free(), as an unhandled
        # C++ exception rather than a catchable Python error:
        #
        #   tuning sweep, trial 3  -> "Exception in pinned allocator free()"
        #   final training, batch 2500 -> CUBLAS_STATUS_EXECUTION_FAILED,
        #                                 then the same allocator abort
        #
        # A crash that destroys hours of training is far more expensive than
        # the small transfer speedup, so it defaults OFF here and is opt-in.
        # The bottleneck in this pipeline is encoder compute, not H2D copies:
        # a batch of 8 functions moves ~14 chunks of int64 ids, which is tiny.
        self.pin_memory = False if pin_memory is None else bool(pin_memory)

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
        self.best_score = -1.0
        self.best_epoch = -1
        self.best_threshold = config.DEFAULT_THRESHOLD
        self.epochs_without_improvement = 0
        self.history = []

        # OOM resilience counters, surfaced in the epoch summary so a run that
        # quietly dropped batches cannot be mistaken for a clean one.
        self.oom_retries = 0
        self.skipped_batches = 0

    @property
    def best_f1(self) -> float:
        """
        Backwards-compatible alias for ``best_score``.

        Early stopping is no longer necessarily driven by F1 (``monitor``
        defaults to PR-AUC), but the attribute name, the checkpoint key and
        the ``fit()`` result key are all long-lived contracts, so the alias
        stays. Read ``monitor`` to learn which metric the number actually is.
        """

        return self.best_score

    @best_f1.setter
    def best_f1(self, value: float) -> None:
        self.best_score = value

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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
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
        self.best_epoch = payload.get("extra", {}).get("best_epoch", -1)

        # ResumableSampler already makes the data order exact; this restores the
        # dropout/init RNG stream so the resumed run continues the same
        # trajectory rather than a merely statistically-similar one.
        rng_restored = ckpt.restore_rng_state(payload.get("rng_state"))

        print("RESUMED from checkpoint:")
        print(
            f"  RNG state: {'restored' if rng_restored else 'ABSENT (pre-RNG checkpoint)'}"
        )
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
            best_f1=self.best_score,
            best_threshold=self.best_threshold,
            epochs_without_improvement=self.epochs_without_improvement,
            # Carried on the latest checkpoint too, not just the best one, so a
            # resumed run can still report which epoch the incumbent came from.
            extra={
                "monitor": self.monitor,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
            },
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

    def _forward_with_oom_retry(self, batch, targets, batch_size):
        """
        Forward + loss, retrying at a smaller chunk micro-batch after an OOM.

        Returns ``(logits, per_sample_loss, loss)``, or None if every attempt
        ran out of memory and the batch has to be skipped.

        Only the micro-batch is reduced. Reducing the *batch* would silently
        change the effective batch size mid-epoch, and dropping chunks would
        silently change the input the label refers to.
        """

        original_micro_batch = getattr(self.model, "chunk_micro_batch", None)
        attempts = [original_micro_batch]

        if original_micro_batch:
            attempts += [
                m for m in (original_micro_batch // 2, original_micro_batch // 4, 1)
                if m >= 1 and m < original_micro_batch
            ]

        for attempt, micro_batch in enumerate(dict.fromkeys(attempts)):
            try:
                if micro_batch is not None:
                    self.model.chunk_micro_batch = micro_batch

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        function_index=batch["function_index"],
                        batch_size=batch_size,
                    )

                    # The contract the whole pipeline depends on.
                    assert logits.shape == targets.shape, (
                        f"logits {tuple(logits.shape)} != targets "
                        f"{tuple(targets.shape)}"
                    )

                    per_sample_loss = self.criterion(logits, targets)
                    loss = per_sample_loss.mean()

                if attempt:
                    self.oom_retries += 1
                    print(
                        f"  [oom] recovered at chunk_micro_batch={micro_batch} "
                        f"({batch['input_ids'].size(0)} chunks)",
                        flush=True,
                    )

                return logits, per_sample_loss, loss

            except torch.cuda.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            finally:
                if original_micro_batch is not None:
                    self.model.chunk_micro_batch = original_micro_batch

        print(
            f"  [oom] SKIPPING a batch of {batch['input_ids'].size(0)} chunks: "
            f"out of memory even at chunk_micro_batch=1",
            flush=True,
        )
        torch.cuda.empty_cache()

        return None

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

            # A batch whose functions happen to all sit near the chunk cap can
            # be several times the average size. On a small card that single
            # batch can OOM and destroy a multi-hour run, so it is retried with
            # a smaller chunk micro-batch before being given up on. Skipping is
            # the last resort and is counted, never silent.
            outcome = self._forward_with_oom_retry(batch, targets, batch_size)

            if outcome is None:
                self.skipped_batches += 1
                continue

            logits, per_sample_loss, loss = outcome
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

            # Keyed to the absolute position in the epoch, not the position in
            # this (possibly resumed) loader. With `offset`, resuming at batch
            # 37 with grad_accum=4 put the step boundaries on a different
            # phase than the original run, so a resumed epoch took a different
            # number of optimizer steps than an uninterrupted one.
            is_step_boundary = (batch_index + 1) % self.grad_accum_steps == 0
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
            "oom_retries": self.oom_retries,
            "skipped_batches": self.skipped_batches,
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
                probabilities,
                targets,
                objective=self.threshold_objective,
                beta=self.threshold_beta,
                min_precision=self.threshold_min_precision,
            )
            print(
                "\n"
                + tuned_metrics.format(
                    f"VALIDATION @ tuned threshold {tuned_threshold:.3f}"
                    f" (objective={self.threshold_objective})"
                )
            )

            # The monitored score is read off the threshold-tuned metrics, but
            # for pr_auc/roc_auc it is threshold-free, so the tuning cannot
            # influence it.
            score = getattr(tuned_metrics, self.monitor)

            if not math.isfinite(score):
                # A single-class validation split makes the AUCs NaN, and NaN
                # loses every comparison, so early stopping would silently fire
                # on epoch one. Fail loudly instead.
                raise RuntimeError(
                    f"Validation {self.monitor} is {score}. This usually means "
                    f"the validation split contains a single class. Early "
                    f"stopping cannot proceed."
                )

            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": stats["loss"],
                    "train_seconds": stats["seconds"],
                    "monitor": self.monitor,
                    "monitor_score": float(score),
                    "valid_f1_at_0.5": default_metrics.f1,
                    "valid_f1_tuned": tuned_metrics.f1,
                    "valid_pr_auc": tuned_metrics.pr_auc,
                    "valid_roc_auc": tuned_metrics.roc_auc,
                    "valid_precision": tuned_metrics.precision,
                    "valid_recall": tuned_metrics.recall,
                    "valid_brier": tuned_metrics.brier,
                    "tuned_threshold": tuned_threshold,
                }
            )

            # ---- best model selection / early stopping ---------------------
            # min_delta: an epoch has to beat the incumbent by a real margin,
            # otherwise float noise keeps resetting the patience counter.
            improved = score > self.best_score + self.min_delta

            if improved:
                self.best_score = float(score)
                self.best_threshold = tuned_threshold
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0

                ckpt.save_checkpoint(
                    self.best_path,
                    self.model,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_f1=self.best_score,
                    best_threshold=self.best_threshold,
                    metrics=tuned_metrics.to_dict(),
                    extra={
                        "pos_weight": self.pos_weight_value,
                        "monitor": self.monitor,
                        "best_score": self.best_score,
                        "best_epoch": self.best_epoch,
                        "threshold_objective": self.threshold_objective,
                    },
                )
                print(
                    f"\n  NEW BEST validation {self.monitor} {self.best_score:.4f} "
                    f"(threshold {self.best_threshold:.3f}) -> {self.best_path}"
                )
            else:
                self.epochs_without_improvement += 1
                print(
                    f"\n  No improvement in {self.monitor} "
                    f"({score:.4f} <= {self.best_score:.4f} + {self.min_delta:g}). "
                    f"Patience {self.epochs_without_improvement}/{self.patience}"
                )

            # batch_in_epoch=0 marks the epoch as finished, so a resume from
            # here starts cleanly at the next epoch.
            self._save_latest(epoch + 1, 0)
            self.start_batch = 0

            if self.epochs_without_improvement >= self.patience:
                print(
                    f"\nEARLY STOPPING: no validation {self.monitor} improvement "
                    f"above {self.min_delta:g} for {self.patience} epoch(s). "
                    f"Best was {self.best_score:.4f} at epoch {self.best_epoch}."
                )
                break

        # ---- restore the best weights --------------------------------------
        # Without this, self.model still holds the LAST epoch's weights. After
        # early stopping those are by definition worse than the best epoch's,
        # so any caller that evaluated the live model - rather than reloading
        # best_codebert.pt by hand - was measuring the wrong thing.
        restored = False

        if self.restore_best and self.best_path.exists():
            ckpt.load_checkpoint(
                self.best_path, model=self.model, map_location=self.device
            )
            self.model.to(self.device)
            restored = True

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"  monitored metric   : {self.monitor}")
        print(f"  best validation    : {self.best_score:.4f} (epoch {self.best_epoch})")
        print(f"  selected threshold : {self.best_threshold:.4f}")
        print(f"  best checkpoint    : {self.best_path}")
        print(
            f"  in-memory weights  : "
            f"{'restored to best epoch' if restored else 'LAST epoch (not restored)'}"
        )

        result = {
            # Historical key name; holds the monitored score, see best_f1 alias.
            "best_f1": self.best_score,
            "best_score": self.best_score,
            "monitor": self.monitor,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "threshold_objective": self.threshold_objective,
            "restored_best_weights": restored,
            "history": self.history,
            "best_checkpoint": str(self.best_path),
        }

        self._write_history(result)

        return result

    def _write_history(self, result: dict) -> None:
        """Persist per-epoch history next to the checkpoints, for the report."""

        import json

        path = self.checkpoint_dir / "training_history.json"

        payload = {
            "monitor": self.monitor,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "threshold_objective": self.threshold_objective,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "best_threshold": self.best_threshold,
            "pos_weight": self.pos_weight_value,
            "epochs_requested": self.epochs,
            "history": result["history"],
        }

        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as error:
            # Losing the history file must never destroy a finished training run.
            print(f"  WARNING: could not write {path}: {error}")
