"""
Evaluation metrics for CodeSentinel vulnerability detection.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def calculate_metrics(
    targets,
    probabilities,
    threshold=0.5,
):
    """
    Calculate binary classification metrics.

    Args:
        targets: True binary labels.
        probabilities: Predicted vulnerability probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary containing evaluation metrics.
    """

    predictions = [
        1
        if probability >= threshold
        else 0
        for probability in probabilities
    ]

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    # ROC-AUC is undefined when only one class
    # exists in the evaluation targets.
    if len(set(targets)) < 2:

        roc_auc = None

    else:

        roc_auc = roc_auc_score(
            targets,
            probabilities,
        )

    metrics = {
        "accuracy": accuracy_score(
            targets,
            predictions,
        ),

        "precision": precision_score(
            targets,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            targets,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            targets,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc,

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics
