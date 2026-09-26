"""
Stage O - PROBABILITY CALIBRATION.

A model can rank well and still be badly calibrated: if the functions it scores
0.9 are vulnerable only 40% of the time, the number cannot be read as a
probability, and any downstream policy that thresholds on "70% confident" is
built on sand. Ranking quality (PR-AUC) and calibration are separate properties
and are measured separately here.

Measured
--------
Brier score            mean squared error of the probabilities. Lower is better.
                       Decomposes into calibration + refinement, so on its own it
                       does not prove miscalibration - hence the two below.
Expected Calibration    weighted mean |confidence - accuracy| over bins.
Error (ECE)             Reads directly as "how far off, on average".
Maximum Calibration     worst single bin. Catches a model that is fine overall
Error (MCE)             but wildly wrong in the high-confidence region that
                        actually gets acted on.
Reliability curve       per-bin predicted vs observed frequency.

Leakage rule
------------
A calibrator is a fitted model. Fitting it on the test set and then reporting
test metrics is the same mistake as tuning a threshold on test. So:

  * the calibrator is FIT on validation data only;
  * whether calibration *helps* is judged by K-fold cross-validation **inside**
    the validation set, so the improvement estimate is measured on data the
    calibrator did not see;
  * the final calibrator is refit on the whole validation set and frozen before
    the test set is touched once.

``ImbalanceAwareBins`` note: with a 30:1 split most probability mass sits near
zero, so equal-width bins leave the upper bins nearly empty and ECE becomes
dominated by noise. Quantile binning is offered for that reason and is the
default.
"""

import numpy as np

# Sub-1% of the probability range; keeps logit() finite without visibly
# moving any probability that matters.
_EPSILON = 1e-6


def _as_arrays(probabilities, targets):
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.int64).ravel()

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: probabilities {probabilities.shape} "
            f"vs targets {targets.shape}"
        )

    return probabilities, targets


def brier_score(probabilities, targets) -> float:
    probabilities, targets = _as_arrays(probabilities, targets)

    if probabilities.size == 0:
        return float("nan")

    return float(np.mean((probabilities - targets) ** 2))


def reliability_curve(probabilities, targets, bins: int = 10, strategy: str = "quantile"):
    """
    Per-bin predicted confidence vs observed positive rate.

    Returns a list of dicts. Empty bins are omitted rather than reported as
    zero, which would otherwise drag ECE towards a flattering number.
    """

    probabilities, targets = _as_arrays(probabilities, targets)

    if probabilities.size == 0:
        return []

    if strategy == "quantile":
        # Deduplicated: heavily tied scores produce repeated quantile edges,
        # which would create zero-width bins.
        edges = np.unique(
            np.quantile(probabilities, np.linspace(0.0, 1.0, bins + 1))
        )
        if edges.size < 2:
            edges = np.array([probabilities.min(), probabilities.max() + _EPSILON])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    # right=True everywhere except the first bin, so the lowest value is included.
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []

    for index in range(len(edges) - 1):
        mask = indices == index
        count = int(mask.sum())

        if not count:
            continue

        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_rate": float(targets[mask].mean()),
                "gap": float(targets[mask].mean() - probabilities[mask].mean()),
            }
        )

    return rows


def calibration_errors(probabilities, targets, bins: int = 10, strategy: str = "quantile"):
    """``(ece, mce)`` from the reliability curve."""

    rows = reliability_curve(probabilities, targets, bins=bins, strategy=strategy)

    if not rows:
        return float("nan"), float("nan")

    total = sum(row["count"] for row in rows)

    ece = sum(row["count"] * abs(row["gap"]) for row in rows) / total
    mce = max(abs(row["gap"]) for row in rows)

    return float(ece), float(mce)


def ece_noise_floor(rows) -> float:
    """
    The ECE a *perfectly calibrated* model of this size would still show.

    ECE is a biased estimator: each bin's observed rate is a binomial mean, so
    it differs from the predicted rate by pure sampling noise even when the
    model is exactly right, and the absolute value in the ECE sum turns that
    noise into a positive number that never cancels.

    For a bin of n samples at probability p, the observed rate has standard
    deviation sqrt(p(1-p)/n), and the expected absolute deviation of a roughly
    normal variable is sqrt(2/pi) times its standard deviation. Averaging that
    over the bins gives the floor below which an "improvement" in ECE is
    indistinguishable from noise - and, worse, is reachable simply by shrinking
    predictions toward the base rate, which any 2-parameter calibrator will do.

    Without this floor the adoption rule fires on perfectly calibrated models.
    """

    if not rows:
        return float("nan")

    total = sum(row["count"] for row in rows)

    if not total:
        return float("nan")

    expected = 0.0

    for row in rows:
        count = row["count"]
        probability = min(max(row["mean_predicted"], 0.0), 1.0)
        variance = probability * (1.0 - probability) / max(count, 1)
        expected += count * np.sqrt(2.0 / np.pi) * np.sqrt(variance)

    return float(expected / total)


def calibration_report(probabilities, targets, bins: int = 10, strategy: str = "quantile"):
    """All calibration diagnostics for one set of predictions."""

    rows = reliability_curve(probabilities, targets, bins=bins, strategy=strategy)
    ece, mce = calibration_errors(probabilities, targets, bins=bins, strategy=strategy)

    return {
        "brier": brier_score(probabilities, targets),
        "ece": ece,
        "mce": mce,
        "ece_noise_floor": ece_noise_floor(rows),
        "bins": rows,
    }


# ----------------------------------------------------------------------
# Calibrators
# ----------------------------------------------------------------------


class PlattCalibrator:
    """
    Logistic regression on the logit of the score (Platt scaling).

    Fitting on the logit rather than the raw probability makes this a pure
    temperature-and-shift correction, which preserves the ranking exactly - so
    PR-AUC and ROC-AUC cannot change, only the probability values do.
    """

    name = "platt"

    def __init__(self):
        self.model = None

    @staticmethod
    def _logit(probabilities):
        clipped = np.clip(probabilities, _EPSILON, 1.0 - _EPSILON)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, probabilities, targets):
        from sklearn.linear_model import LogisticRegression

        probabilities, targets = _as_arrays(probabilities, targets)

        self.model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self.model.fit(self._logit(probabilities), targets)

        return self

    def predict(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
        return self.model.predict_proba(self._logit(probabilities))[:, 1]

    def to_dict(self) -> dict:
        return {
            "method": self.name,
            "coefficient": float(self.model.coef_.ravel()[0]),
            "intercept": float(self.model.intercept_.ravel()[0]),
        }

    @classmethod
    def from_dict(cls, payload: dict):
        instance = cls()
        instance._coefficient = payload["coefficient"]
        instance._intercept = payload["intercept"]

        # Rebuilt arithmetically rather than by unpickling an estimator, so a
        # frozen policy stays loadable across scikit-learn versions.
        def predict(probabilities, a=instance._coefficient, b=instance._intercept):
            logit = cls._logit(np.asarray(probabilities, dtype=np.float64).ravel())
            return 1.0 / (1.0 + np.exp(-(a * logit.ravel() + b)))

        instance.predict = predict
        return instance


class IsotonicCalibrator:
    """
    Isotonic regression: a free-form monotone map from score to probability.

    More flexible than Platt and able to fix non-sigmoidal distortion, but it
    has far more effective parameters and will happily overfit a small or
    sparsely-positive validation set. Monotone, so ranking is preserved up to
    ties; it can flatten distinct scores onto one value, which slightly coarsens
    the threshold grid.
    """

    name = "isotonic"

    def __init__(self):
        self.model = None

    def fit(self, probabilities, targets):
        from sklearn.isotonic import IsotonicRegression

        probabilities, targets = _as_arrays(probabilities, targets)

        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(probabilities, targets)

        return self

    def predict(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
        return self.model.predict(probabilities)

    def to_dict(self) -> dict:
        # The fitted step function, stored as knots. Small, human-readable, and
        # re-applied with np.interp rather than a pickled estimator.
        return {
            "method": self.name,
            "x": [float(v) for v in self.model.X_thresholds_],
            "y": [float(v) for v in self.model.y_thresholds_],
        }

    @classmethod
    def from_dict(cls, payload: dict):
        instance = cls()
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)

        def predict(probabilities, x=x, y=y):
            # Clipped at the ends, matching out_of_bounds="clip" at fit time.
            return np.interp(
                np.asarray(probabilities, dtype=np.float64).ravel(), x, y
            )

        instance.predict = predict
        return instance


CALIBRATORS = {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def calibrator_from_dict(payload: dict):
    """Rebuild a frozen calibrator from its serialised parameters."""

    method = payload.get("method")

    if method not in CALIBRATORS:
        raise ValueError(f"Unknown calibration method in policy: {method!r}")

    return CALIBRATORS[method].from_dict(payload)


def _make(method: str):
    if method not in CALIBRATORS:
        raise ValueError(
            f"Unknown calibration method: {method}. "
            f"Choose from {sorted(CALIBRATORS)}"
        )

    return CALIBRATORS[method]()


def cross_validated_calibration(
    probabilities,
    targets,
    method: str = "platt",
    folds: int = 5,
    seed: int = 42,
    bins: int = 10,
):
    """
    Honest estimate of whether calibration helps, without touching test data.

    Stratified K-fold **within the validation set**: fit on K-1 folds, predict
    the held-out fold, and assemble out-of-fold calibrated probabilities for
    every validation sample. Scoring those is fair, because no sample was
    scored by a calibrator that had seen it.

    Returns ``(out_of_fold_probabilities, before_report, after_report)``.
    """

    from sklearn.model_selection import StratifiedKFold

    probabilities, targets = _as_arrays(probabilities, targets)

    positives = int(targets.sum())
    negatives = int(targets.size - positives)

    # StratifiedKFold needs at least `folds` members of each class.
    usable_folds = max(2, min(folds, positives, negatives))

    if positives < 2 or negatives < 2:
        raise ValueError(
            "Calibration needs at least two samples of each class; got "
            f"{positives} positive / {negatives} negative."
        )

    splitter = StratifiedKFold(
        n_splits=usable_folds, shuffle=True, random_state=seed
    )

    out_of_fold = np.zeros_like(probabilities)

    for train_index, test_index in splitter.split(probabilities, targets):
        calibrator = _make(method)
        calibrator.fit(probabilities[train_index], targets[train_index])
        out_of_fold[test_index] = calibrator.predict(probabilities[test_index])

    before = calibration_report(probabilities, targets, bins=bins)
    after = calibration_report(out_of_fold, targets, bins=bins)

    before["folds"] = usable_folds
    after["folds"] = usable_folds
    after["method"] = method

    return out_of_fold, before, after


def fit_final_calibrator(probabilities, targets, method: str = "platt"):
    """
    Fit the calibrator that will be frozen and applied to the test set.

    Call this ONLY with validation data, and only after
    ``cross_validated_calibration`` has justified using it.
    """

    return _make(method).fit(probabilities, targets)


def format_calibration(report: dict, title: str) -> str:
    lines = [
        "=" * 66,
        title,
        "=" * 66,
        f"  brier : {report['brier']:.6f}   (lower is better)",
        f"  ECE   : {report['ece']:.6f}   (mean |confidence - accuracy|)",
        f"  MCE   : {report['mce']:.6f}   (worst bin)",
        "",
        f"  {'bin range':<24}{'n':>8}{'predicted':>12}{'observed':>11}{'gap':>10}",
    ]

    for row in report["bins"]:
        span = f"[{row['lower']:.4f}, {row['upper']:.4f}]"
        lines.append(
            f"  {span:<24}{row['count']:>8}"
            f"{row['mean_predicted']:>12.4f}"
            f"{row['observed_rate']:>11.4f}"
            f"{row['gap']:>+10.4f}"
        )

    return "\n".join(lines)


def decide(
    before: dict,
    after: dict,
    min_relative_gain: float = 0.10,
    noise_multiple: float = 2.0,
) -> dict:
    """
    Recommend whether to adopt calibration.

    Adopted only when all three hold:

      * the measured ECE is more than ``noise_multiple`` times the sampling
        noise floor, so there is genuine miscalibration to fix;
      * out-of-fold ECE improves by at least ``min_relative_gain``;
      * the Brier score does not regress.

    The noise-floor condition is what stops a calibrator being bolted onto an
    already-calibrated model: see ``ece_noise_floor``.
    """

    ece_before, ece_after = before["ece"], after["ece"]
    brier_before, brier_after = before["brier"], after["brier"]
    floor = before.get("ece_noise_floor", 0.0)

    if not np.isfinite(ece_before) or not np.isfinite(ece_after):
        return {"adopt": False, "reason": "calibration error could not be computed"}

    relative_gain = (
        (ece_before - ece_after) / ece_before if ece_before > 0 else 0.0
    )
    brier_worse = brier_after > brier_before

    # Three independent conditions, all of which must hold:
    #   1. the miscalibration is real, i.e. clearly above the sampling-noise
    #      floor a perfectly calibrated model would still show;
    #   2. the out-of-fold gain is large enough to be worth another moving part;
    #   3. probability accuracy (Brier) does not regress.
    real_miscalibration = (
        np.isfinite(floor) and ece_before > noise_multiple * floor
    )

    adopt = real_miscalibration and relative_gain >= min_relative_gain and not brier_worse

    if adopt:
        reason = (
            f"ECE improved {ece_before:.4f} -> {ece_after:.4f} "
            f"({relative_gain:.1%} relative), starting from "
            f"{ece_before / floor:.1f}x the noise floor, with no Brier regression"
        )
    elif not real_miscalibration:
        reason = (
            f"ECE {ece_before:.4f} is within sampling noise for this sample size "
            f"(floor {floor:.4f}); the model is already calibrated and any "
            f"apparent gain is an artifact of shrinking toward the base rate"
        )
    elif brier_worse:
        reason = (
            f"Brier regressed {brier_before:.6f} -> {brier_after:.6f}; "
            f"calibration is not free here"
        )
    else:
        reason = (
            f"ECE gain {relative_gain:.1%} is below the {min_relative_gain:.0%} "
            f"bar; the model is already close enough to calibrated"
        )

    return {
        "adopt": adopt,
        "reason": reason,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_noise_floor": float(floor),
        "brier_before": brier_before,
        "brier_after": brier_after,
        "relative_ece_gain": float(relative_gain),
    }
