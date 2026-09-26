"""
Tiny dataset overfitting test for CodeSentinel.

This test checks whether the complete training pipeline can
intentionally overfit a very small fixed dataset.

If the model cannot learn a tiny dataset, there is likely a problem
with labels, data flow, loss calculation, gradients, or optimization.
"""

from pathlib import Path
from functools import partial
import random

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from backend.ml.preprocessing.loader import (
    PrimeVulChunkedDataset,
)

from backend.ml.models.training_collate import (
    collate_training_batch,
)

from backend.ml.models.codebert_classifier import (
    CodeBERTClassifier,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunked"
    / "primevul_train_chunked.jsonl"
)

MODEL_NAME = "microsoft/codebert-base"

NUM_SAMPLES_PER_CLASS = 4

NUM_EPOCHS = 30

LEARNING_RATE = 2e-5

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print("CODEBERT TINY DATASET OVERFITTING TEST")

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

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(SEED)


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    dataset = PrimeVulChunkedDataset(
        TRAIN_DATA_PATH
    )


    # --------------------------------------------------------
    # SELECT BALANCED TINY DATASET
    # --------------------------------------------------------

    vulnerable_indices = []

    benign_indices = []


    for index, sample in enumerate(dataset):

        target = sample["target"]

        if target == 1:

            if (
                len(vulnerable_indices)
                < NUM_SAMPLES_PER_CLASS
            ):

                vulnerable_indices.append(
                    index
                )

        else:

            if (
                len(benign_indices)
                < NUM_SAMPLES_PER_CLASS
            ):

                benign_indices.append(
                    index
                )


        if (
            len(vulnerable_indices)
            == NUM_SAMPLES_PER_CLASS
            and
            len(benign_indices)
            == NUM_SAMPLES_PER_CLASS
        ):

            break


    selected_indices = (
        vulnerable_indices
        + benign_indices
    )


    random.shuffle(
        selected_indices
    )


    print(
        "\nTiny dataset composition:"
    )

    print(
        f"Vulnerable: "
        f"{len(vulnerable_indices)}"
    )

    print(
        f"Benign: "
        f"{len(benign_indices)}"
    )

    print(
        f"Total: "
        f"{len(selected_indices)}"
    )


    tiny_dataset = Subset(
        dataset,
        selected_indices,
    )


    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    collate_fn = partial(
        collate_training_batch,
        tokenizer=tokenizer,
    )


    loader = DataLoader(
        tiny_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nLoading CodeBERT classifier..."
    )


    model = CodeBERTClassifier()

    model.to(device)

    model.train()


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print(
        "\nStarting overfitting test..."
    )


    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        total_loss = 0.0

        correct = 0

        total = 0


        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            chunk_counts = batch[
                "chunk_counts"
            ].to(device)

            targets = batch[
                "targets"
            ].to(device)


            optimizer.zero_grad()


            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_counts=chunk_counts,
            )


            loss = criterion(
                logits,
                targets,
            )


            loss.backward()


            optimizer.step()


            total_loss += (
                loss.item()
                * targets.size(0)
            )


            probabilities = torch.sigmoid(
                logits
            )


            predictions = (
                probabilities >= 0.5
            ).float()


            correct += (
                predictions == targets
            ).sum().item()


            total += (
                targets.size(0)
            )


        average_loss = (
            total_loss / total
        )

        accuracy = (
            correct / total
        )


        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
        )


    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL OVERFITTING CHECK"
    )

    print(
        "=" * 70
    )


    model.eval()


    correct = 0

    total = 0


    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            chunk_counts = batch[
                "chunk_counts"
            ].to(device)

            targets = batch[
                "targets"
            ].to(device)


            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_counts=chunk_counts,
            )


            probabilities = torch.sigmoid(
                logits
            )


            predictions = (
                probabilities >= 0.5
            ).float()


            correct += (
                predictions == targets
            ).sum().item()


            total += (
                targets.size(0)
            )


            print(
                "\nPredictions:"
            )


            for (
                target,
                probability,
                prediction,
            ) in zip(
                targets.cpu().tolist(),
                probabilities.cpu().tolist(),
                predictions.cpu().tolist(),
            ):

                print(
                    f"True: {int(target)} | "
                    f"Probability: "
                    f"{probability:.4f} | "
                    f"Prediction: "
                    f"{int(prediction)}"
                )


    final_accuracy = (
        correct / total
    )


    print(
        f"\nFinal tiny-dataset accuracy: "
        f"{final_accuracy * 100:.2f}%"
    )


    print(
        "\nInterpretation:"
    )


    if final_accuracy >= 0.95:

        print(
            "PASS: The model successfully "
            "learned the tiny dataset."
        )

    else:

        print(
            "FAIL: The model could not sufficiently "
            "overfit the tiny dataset."
        )


    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":

    main()
