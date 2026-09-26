"""
Binary classification metrics and validation-only threshold optimisation.

Accuracy is reported but deliberately de-emphasised: on a ~97% benign split a
model that always predicts "benign" scores 97% accuracy and is worthless.
F1, precision, recall, PR-AUC and ROC-AUC are the metrics that matter here.

Why PR-AUC is the headline ranking metric
-----------------------------------------
ROC-AUC is computed from TPR and FPR. FPR has the negative count in its
denominator, and on a 30:1 benign:vulnerable split that denominator is huge,
so a large absolute number of false positives still moves FPR very little.
ROC-AUC therefore stays flatteringly high on exactly the models that would
drown a reviewer in false alarms. Precision-recall curves use precision,
whose denominator is the *predicted* positives, so PR-AUC degrades honestly
when the model over-predicts. Both are reported; PR-AUC is what is optimised.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from backend.ml import config


# ----------------------------------------------------------------------
# Threshold-free scores
# ----------------------------------------------------------------------


def _ranking_scores(probabilities, targets) -> tuple:
    """
    ``(roc_auc, pr_auc, brier, positive_rate)`` - none of which depend on the
    decision threshold, so they are computed once rather than per grid point.

    ROC-AUC and PR-AUC are undefined when only one class is present; they come
    back as NaN in that case rather than as a misleading 0.0 or 1.0.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    positives = int(targets.sum())
    total = int(targets.size)
    positive_rate = positives / total if total else 0.0

    # Brier score is well defined with a single class; the AUCs are not.
    brier = float(np.mean((probabilities - targets) ** 2)) if total else float("nan")

    if len(np.unique(targets)) < 2:
        return float("nan"), float("nan"), brier, positive_rate

    roc_auc = float(roc_auc_score(targets, probabilities))
    pr_auc = float(average_precision_score(targets, probabilities))

    return roc_auc, pr_auc, brier, positive_rate


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
    # Added after the original six: defaults keep every existing positional
    # construction and every stored checkpoint payload loadable.
    pr_auc: float = float("nan")
    brier: float = float("nan")
    mcc: float = float("nan")
    positive_rate: float = 0.0
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
            f"  pr_auc    : {self.pr_auc:.4f}   <- headline ranking metric",
            f"  roc_auc   : {self.roc_auc:.4f}   <- optimistic under imbalance",
            f"  brier     : {self.brier:.4f}   <- lower is better",
            f"  mcc       : {self.mcc:.4f}",
            "",
            "  confusion matrix",
            f"    TN = {self.tn:<8} FP = {self.fp:<8}",
            f"    FN = {self.fn:<8} TP = {self.tp:<8}",
            "",
            f"  support   : {self.support_positive} vulnerable / "
            f"{self.support_negative} benign",
            f"  predicted positive: {self.predicted_positive}",
        ]

        # The baseline a PR-AUC must beat is the positive rate, not 0.5.
        if self.positive_rate:
            lines.append(
                f"  PR-AUC baseline (always-positive): {self.positive_rate:.4f}"
            )

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

    from sklearn.metrics import confusion_matrix

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

    roc_auc, pr_auc, brier, positive_rate = _ranking_scores(probabilities, targets)

    return BinaryMetrics(
        threshold=float(threshold),
        **_rates(tn, fp, fn, tp),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier=brier,
        positive_rate=positive_rate,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        support_positive=int(targets.sum()),
        support_negative=int((targets == 0).sum()),
        predicted_positive=int(predictions.sum()),
    )


def _rates(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Accuracy / precision / recall / F1 / MCC from a confusion matrix."""

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # MCC stays informative when one class dominates, where accuracy does not.
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (
        (tp * tn - fp * fn) / np.sqrt(denominator) if denominator > 0 else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }


# ----------------------------------------------------------------------
# Threshold sweep
# ----------------------------------------------------------------------


def sweep_thresholds(
    probabilities,
    targets,
    steps: int = config.THRESHOLD_SEARCH_STEPS,
    low: float = config.THRESHOLD_SEARCH_MIN,
    high: float = config.THRESHOLD_SEARCH_MAX,
) -> list:
    """
    Evaluate the confusion matrix at every threshold on the grid.

    Vectorised: the probabilities are sorted once and each threshold becomes a
    binary search, instead of re-scanning the array (and recomputing the
    threshold-independent AUCs) 91 times as the original implementation did.

    Returns a list of dicts, one per grid point.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: probabilities {probabilities.shape} "
            f"vs targets {targets.shape}"
        )

    total = probabilities.size
    total_positive = int(targets.sum())

    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_targets = targets[order]

    # positive_suffix[i] = number of positives at sorted index >= i.
    positive_suffix = np.zeros(total + 1, dtype=np.int64)
    if total:
        positive_suffix[:-1] = np.cumsum(sorted_targets[::-1])[::-1]

    rows = []

    for threshold in np.linspace(low, high, steps):
        # side="left" -> every index from here on has probability >= threshold,
        # which matches the `probabilities >= threshold` rule in compute_metrics.
        cut = int(np.searchsorted(sorted_probabilities, threshold, side="left"))

        predicted_positive = total - cut
        tp = int(positive_suffix[cut])
        fp = predicted_positive - tp
        fn = total_positive - tp
        tn = total - tp - fp - fn

        row = {
            "threshold": float(threshold),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "predicted_positive": int(predicted_positive),
        }
        row.update(_rates(tn, fp, fn, tp))
        rows.append(row)

    return rows


def _objective_value(row: dict, objective: str, beta: float, min_precision: float):
    """Score one sweep row under the named selection objective."""

    if objective == "f1":
        return row["f1"]

    if objective == "fbeta":
        precision, recall = row["precision"], row["recall"]
        denominator = (beta * beta * precision) + recall
        if denominator == 0:
            return 0.0
        return (1 + beta * beta) * precision * recall / denominator

    if objective == "mcc":
        return row["mcc"]

    if objective == "recall_at_precision":
        # Infeasible points score below every feasible one, but are still
        # ordered by precision so the search degrades gracefully rather than
        # returning an arbitrary threshold when the constraint is unreachable.
        if row["precision"] >= min_precision:
            return 1.0 + row["recall"]
        return row["precision"] / max(min_precision, 1e-12)

    raise ValueError(f"Unknown threshold objective: {objective}")


def find_best_threshold(
    probabilities,
    targets,
    steps: int = config.THRESHOLD_SEARCH_STEPS,
    low: float = config.THRESHOLD_SEARCH_MIN,
    high: float = config.THRESHOLD_SEARCH_MAX,
    objective: str = "f1",
    beta: float = 2.0,
    min_precision: float = 0.5,
):
    """
    Grid-search the decision threshold that maximises ``objective``.

    MUST be called with VALIDATION predictions only. Tuning the threshold on
    test data leaks the test set into model selection and invalidates the
    final numbers.

    Objectives:
        "f1"                  - balanced default
        "fbeta"               - weights recall beta times more than precision;
                                beta=2 suits vulnerability triage, where a
                                missed vulnerability costs more than a false alarm
        "mcc"                 - robust to imbalance
        "recall_at_precision" - maximise recall subject to precision >= min_precision

    Returns ``(best_threshold, best_metrics, sweep)``. ``sweep`` is a list of
    ``(threshold, f1)`` pairs, unchanged from the original signature.
    """

    rows = sweep_thresholds(probabilities, targets, steps=steps, low=low, high=high)

    best_row = None
    best_score = -np.inf

    for row in rows:
        score = _objective_value(row, objective, beta, min_precision)

        # Strict > keeps the lowest threshold among ties, which is the
        # recall-favouring choice for vulnerability detection.
        if score > best_score:
            best_score = score
            best_row = row

    best_threshold = (
        best_row["threshold"] if best_row else config.DEFAULT_THRESHOLD
    )

    # Recomputed through compute_metrics so the returned object is built by
    # exactly the same code path as every other reported metric.
    best_metrics = compute_metrics(probabilities, targets, best_threshold)

    sweep = [(row["threshold"], row["f1"]) for row in rows]

    return best_threshold, best_metrics, sweep
