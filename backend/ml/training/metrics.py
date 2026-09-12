"""
Binary classification metrics and validation-only threshold optimisation.

Accuracy is reported but deliberately de-emphasised: on a ~97% benign split a
model that always predicts "benign" scores 97% accuracy and is worthless.
F1, precision, recall and ROC-AUC are the metrics that matter here.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from backend.ml import config


@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    support_positive: int
    support_negative: int
    predicted_positive: int
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self, title: str = "METRICS") -> str:
        lines = [
            "=" * 60,
            title,
            "=" * 60,
            f"  threshold : {self.threshold:.4f}",
            f"  accuracy  : {self.accuracy:.4f}   <- not the headline metric",
            f"  precision : {self.precision:.4f}",
            f"  recall    : {self.recall:.4f}",
            f"  f1        : {self.f1:.4f}",
            f"  roc_auc   : {self.roc_auc:.4f}",
            "",
            "  confusion matrix",
            f"    TN = {self.tn:<8} FP = {self.fp:<8}",
            f"    FN = {self.fn:<8} TP = {self.tp:<8}",
            "",
            f"  support   : {self.support_positive} vulnerable / "
            f"{self.support_negative} benign",
            f"  predicted positive: {self.predicted_positive}",
        ]

        # A collapsed model is the failure mode most worth shouting about.
        if self.predicted_positive == 0:
            lines.append(
                "  WARNING: model predicted the negative class for every sample."
            )
        elif self.predicted_positive == self.support_positive + self.support_negative:
            lines.append(
                "  WARNING: model predicted the positive class for every sample."
            )

        return "\n".join(lines)


def compute_metrics(
    probabilities,
    targets,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> BinaryMetrics:
    """
    Compute all binary metrics at a given decision threshold.

    ``probabilities`` are sigmoid outputs in [0, 1]; ``targets`` are 0/1.
    """

    from sklearn.metrics import confusion_matrix, roc_auc_score

    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: probabilities {probabilities.shape} "
            f"vs targets {targets.shape}"
        )

    predictions = (probabilities >= threshold).astype(np.int64)

    # labels=[0,1] forces a 2x2 matrix even when a class is absent from
    # either the targets or the predictions, so unpacking never raises.
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # ROC-AUC is undefined with a single class present.
    if len(np.unique(targets)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(targets, probabilities))

    return BinaryMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        support_positive=int(targets.sum()),
        support_negative=int((targets == 0).sum()),
        predicted_positive=int(predictions.sum()),
    )


def find_best_threshold(
    probabilities,
    targets,
    steps: int = config.THRESHOLD_SEARCH_STEPS,
    low: float = config.THRESHOLD_SEARCH_MIN,
    high: float = config.THRESHOLD_SEARCH_MAX,
):
    """
    Grid-search the decision threshold that maximises F1.

    MUST be called with VALIDATION predictions only. Tuning the threshold on
    test data leaks the test set into model selection and invalidates the
    final numbers.

    Returns ``(best_threshold, best_metrics, sweep)`` where ``sweep`` is a
    list of ``(threshold, f1)`` pairs for logging or plotting.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    best_threshold = config.DEFAULT_THRESHOLD
    best_f1 = -1.0
    best_metrics = None
    sweep = []

    for threshold in np.linspace(low, high, steps):
        metrics = compute_metrics(probabilities, targets, float(threshold))
        sweep.append((float(threshold), metrics.f1))

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_metrics(
            probabilities, targets, config.DEFAULT_THRESHOLD
        )

    return best_threshold, best_metrics, sweep
