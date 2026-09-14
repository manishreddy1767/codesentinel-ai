"""
CodeSentinel CodeBERT training pipeline.

Features:
- Mixed precision training
- Periodic checkpointing
- Resume training
- High-loss diagnostics
- Best model saving
- Early stopping
"""

from pathlib import Path
from functools import partial
import copy
import math

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "microsoft/codebert-base"

TRAIN_DATA_PATH = Path(
    "data/chunked/primevul_train_chunked.jsonl"
)

VALID_DATA_PATH = Path(
    "data/chunked/primevul_valid_chunked.jsonl"
)

CHECKPOINT_DIR = Path(
    "data/checkpoints"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "best_codebert.pt"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "latest_checkpoint.pt"
)

BATCH_SIZE = 1

NUM_EPOCHS = 5

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

PATIENCE = 2

CHECKPOINT_INTERVAL = 500

HIGH_LOSS_THRESHOLD = 6.0


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    batch_index,
    model,
    optimizer,
    scheduler,
    scaler,
    best_f1,
    best_model_state,
    epochs_without_improvement,
):

    checkpoint = {
        "epoch": epoch,
        "batch_index": batch_index,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": best_f1,
        "epochs_without_improvement":
            epochs_without_improvement,
    }

    if best_model_state is not None:

        checkpoint[
            "best_model_state"
        ] = best_model_state

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# HIGH LOSS DIAGNOSTICS
# ============================================================

def print_high_loss_diagnostics(
    epoch,
    batch_index,
    loss,
    logits,
    targets,
    chunk_counts,
):

    probabilities = torch.sigmoid(
        logits.detach()
    )

    print("\n" + "!" * 70)
    print("HIGH LOSS DETECTED")
    print("!" * 70)

    print(f"Epoch: {epoch}")
    print(f"Batch: {batch_index}")
    print(f"Loss: {loss.item():.4f}")

    for index in range(
        len(targets)
    ):

        probability = (
            probabilities[index]
            .detach()
            .float()
            .cpu()
            .item()
        )

        logit = (
            logits[index]
            .detach()
            .float()
            .cpu()
            .item()
        )

        target = (
            targets[index]
            .detach()
            .float()
            .cpu()
            .item()
        )

        print(
            f"\nSample {index + 1}"
        )

        print(
            f"True label: {target:.0f}"
        )

        print(
            f"Predicted vulnerability probability: "
            f"{probability:.6f}"
        )

        print(
            f"Logit: {logit:.6f}"
        )

    print(
        f"\nChunk counts: "
        f"{chunk_counts.detach().cpu().tolist()}"
    )

    print("!" * 70 + "\n")


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    criterion,
    scaler,
    epoch,
    start_batch,
    best_f1,
    best_model_state,
    epochs_without_improvement,
):

    model.train()

    total_loss = 0.0
    processed_batches = 0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        # Skip batches already completed before checkpoint.
        if batch_index <= start_batch:
            continue

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        chunk_counts = batch[
            "chunk_counts"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "targets"
        ].to(
            device,
            non_blocking=True,
        )

        # Original number of chunks before limiting.
        original_chunk_counts = batch.get(
            "original_chunk_counts",
            None,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        try:

            with autocast(
                device_type="cuda",
                enabled=(device.type == "cuda"),
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_counts=chunk_counts,
                )

                loss = criterion(
                    logits,
                    targets,
                )

            # ------------------------------------------------
            # HIGH LOSS DIAGNOSTICS
            # ------------------------------------------------

            if not torch.isfinite(loss):

                print_high_loss_diagnostics(
                    epoch=epoch,
                    batch_index=batch_index,
                    loss=loss,
                    logits=logits,
                    targets=targets,
                    chunk_counts=chunk_counts,
                )

                if original_chunk_counts is not None:

                    print(
                        "Original chunk counts: "
                        f"{original_chunk_counts.tolist()}"
                    )

            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Training stopped because loss "
                    "became NaN or Inf."
                )

            # --------------------------------------------
            # BACKPROPAGATION
            # --------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            loss_value = loss.item()

            total_loss += loss_value
            processed_batches += 1

            # --------------------------------------------
            # PROGRESS
            # --------------------------------------------

            if batch_index % 100 == 0:

                print(
                    f"Batch "
                    f"{batch_index}/{len(dataloader)} "
                    f"| Loss: {loss_value:.4f}"
                )

                if device.type == "cuda":

                    allocated = (
                        torch.cuda.memory_allocated()
                        / (1024 ** 2)
                    )

                    reserved = (
                        torch.cuda.memory_reserved()
                        / (1024 ** 2)
                    )

                    max_allocated = (
                        torch.cuda.max_memory_allocated()
                        / (1024 ** 2)
                    )

                    print(
                        f"GPU memory | "
                        f"Allocated: {allocated:.0f} MiB | "
                        f"Reserved: {reserved:.0f} MiB | "
                        f"Peak: {max_allocated:.0f} MiB"
                    )

            # --------------------------------------------
            # PERIODIC CHECKPOINT
            # --------------------------------------------

            if (
                batch_index
                % CHECKPOINT_INTERVAL
                == 0
            ):

                save_checkpoint(
                    path=LATEST_CHECKPOINT_PATH,
                    epoch=epoch,
                    batch_index=batch_index,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_f1=best_f1,
                    best_model_state=best_model_state,
                    epochs_without_improvement=
                        epochs_without_improvement,
                )

                print(
                    f"\nCheckpoint saved "
                    f"at epoch {epoch}, "
                    f"batch {batch_index}.\n"
                )

        finally:

            # Explicitly release references to tensors from
            # the current training iteration.
            del (
                input_ids,
                attention_mask,
                chunk_counts,
                targets,
            )

            if "logits" in locals():

                del logits

            if "loss" in locals():

                del loss

            # Clear unused cached memory periodically.
            if (
                device.type == "cuda"
                and batch_index % 100 == 0
            ):

                torch.cuda.empty_cache()

    if processed_batches == 0:

        return 0.0

    average_loss = (
        total_loss
        / processed_batches
    )

    return average_loss


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("CODESENTINEL CODEBERT TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # CHECKPOINT DIRECTORY
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset = (
        PrimeVulChunkedDataset(
            TRAIN_DATA_PATH
        )
    )

    print(
        f"Training functions: "
        f"{len(train_dataset)}"
    )

    print("\nLoading validation dataset...")

    valid_dataset = (
        PrimeVulChunkedDataset(
            VALID_DATA_PATH
        )
    )

    print(
        f"Validation functions: "
        f"{len(valid_dataset)}"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    collate_fn = partial(
        collate_training_batch,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading CodeBERT classifier...")

    model = CodeBERTClassifier()

    model.to(device)

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    # Give vulnerable samples more importance because
    # the training dataset has a 5:1 benign-to-vulnerable ratio.
    pos_weight = torch.tensor(
        [5.0],
        device=device,
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    total_training_steps = (
        len(train_loader)
        * NUM_EPOCHS
    )

    warmup_steps = int(
        total_training_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=
                total_training_steps,
        )
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    start_epoch = 1
    start_batch = 0

    best_f1 = -1.0

    best_model_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # RESUME CHECKPOINT
    # --------------------------------------------------------

    if LATEST_CHECKPOINT_PATH.exists():

        print(
            "\nFound latest checkpoint."
        )

        checkpoint = torch.load(
            LATEST_CHECKPOINT_PATH,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):

            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        best_f1 = checkpoint.get(
            "best_f1",
            -1.0,
        )

        best_model_state = checkpoint.get(
            "best_model_state",
            None,
        )

        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        checkpoint_epoch = checkpoint[
            "epoch"
        ]

        checkpoint_batch = checkpoint[
            "batch_index"
        ]

        # Continue current epoch if it was not finished.
        if checkpoint_batch < len(train_loader):

            start_epoch = checkpoint_epoch

            start_batch = checkpoint_batch

        else:

            start_epoch = (
                checkpoint_epoch + 1
            )

            start_batch = 0

        print(
            f"Resuming from "
            f"epoch {start_epoch}, "
            f"batch {start_batch}."
        )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1,
    ):

        print("\n" + "=" * 70)

        print(
            f"EPOCH "
            f"{epoch}/{NUM_EPOCHS}"
        )

        print("=" * 70)

        print("\nTraining...")

        epoch_start_batch = (
            start_batch
            if epoch == start_epoch
            else 0
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            start_batch=epoch_start_batch,
            best_f1=best_f1,
            best_model_state=best_model_state,
            epochs_without_improvement=
                epochs_without_improvement,
        )

        # Reset after first resumed epoch.
        start_batch = 0

        print(
            f"\nAverage training loss: "
            f"{train_loss:.4f}"
        )

        # ----------------------------------------------------
        # SAVE END-OF-EPOCH CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            path=LATEST_CHECKPOINT_PATH,
            epoch=epoch,
            batch_index=len(train_loader),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_f1=best_f1,
            best_model_state=best_model_state,
            epochs_without_improvement=
                epochs_without_improvement,
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print("\nValidating...")

        validation_metrics = (
            evaluate_model(
                model=model,
                dataloader=valid_loader,
                device=device,
            )
        )

        print(
            "\nValidation metrics:"
        )

        for name, value in (
            validation_metrics.items()
        ):

            if isinstance(
                value,
                float,
            ):

                print(
                    f"{name}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{name}: "
                    f"{value}"
                )

        current_f1 = (
            validation_metrics["f1"]
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        best_model_state,
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_f1":
                        best_f1,
                    "validation_metrics":
                        validation_metrics,
                },
                BEST_CHECKPOINT_PATH,
            )

            print(
                "\nNew best model saved!"
            )

            print(
                f"Best validation F1: "
                f"{best_f1:.4f}"
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            print(
                "\nNo improvement."
            )

            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}"
                f"/{PATIENCE}"
            )

        # Update latest checkpoint with
        # latest early-stopping state.
        save_checkpoint(
            path=LATEST_CHECKPOINT_PATH,
            epoch=epoch,
            batch_index=len(train_loader),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_f1=best_f1,
            best_model_state=best_model_state,
            epochs_without_improvement=
                epochs_without_improvement,
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # --------------------------------------------------------
    # TRAINING COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nBest validation F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best model saved at:\n"
        f"{BEST_CHECKPOINT_PATH}"
    )

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    print(
        "\nCodeSentinel training "
        "finished successfully."
    )


if __name__ == "__main__":
    main()
