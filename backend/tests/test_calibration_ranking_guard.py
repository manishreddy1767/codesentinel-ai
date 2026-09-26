"""
The calibration adopt rule must not trade ranking for probability quality.

Why this exists
---------------
`decide()` originally adopted a calibrator when ECE was above its sampling-noise
floor, ECE improved by at least 10% out-of-fold, and Brier did not regress. All
three can hold while the *ordering* gets worse, and PR-AUC is this project's
declared headline metric.

Platt scaling cannot cause that: it is a monotone transform of the logit, so it
preserves the ranking exactly. Isotonic regression can, and isotonic is the
method that was actually adopted. It is a step function fitted to one score
distribution, so it maps distinct scores onto the same output; each tie discards
ranking information. On the real epoch-1 validation data the effect was
negligible (PR-AUC +0.000115), but the adopted fit has only 23 distinct output
levels from 37 knots, so a different checkpoint's score distribution could
collapse into far fewer.

These tests pin the guard's behaviour and demonstrate the mechanism it defends
against.
"""

import numpy as np
import pytest

from backend.ml.evaluation.calibration import calibration_report, decide


def _reports(pr_auc_before, pr_auc_after):
    """
    A pair of reports where ECE and Brier both clearly improve, so the ONLY
    thing that can change the verdict is the ranking.
    """

    before = {
        "ece": 0.0300,
        "brier": 0.0340,
        "ece_noise_floor": 0.0024,
        "pr_auc": pr_auc_before,
    }
    after = {
        "ece": 0.0020,
        "brier": 0.0250,
        "pr_auc": pr_auc_after,
    }
    return before, after


def test_adopts_when_ranking_is_preserved():
    """The baseline case: this must still be adopted, or the guard is too strict."""

    verdict = decide(*_reports(0.1114, 0.1115))

    assert verdict["adopt"] is True, verdict["reason"]
    assert verdict["ranking_checked"] is True
    assert verdict["relative_ranking_loss"] <= 0


def test_rejects_a_calibrator_that_destroys_ranking():
    """ECE and Brier both improve, but PR-AUC falls 10% — must be rejected."""

    verdict = decide(*_reports(0.1114, 0.1003))

    assert verdict["adopt"] is False
    assert "PR-AUC" in verdict["reason"], verdict["reason"]
    assert verdict["relative_ranking_loss"] == pytest.approx(0.0997, abs=1e-3)


def test_tolerates_ranking_loss_inside_the_bar():
    """A 1% loss is under the 2% bar, so ECE and Brier gains should still win."""

    verdict = decide(*_reports(0.1114, 0.1103))

    assert verdict["adopt"] is True, verdict["reason"]
    assert 0 < verdict["relative_ranking_loss"] < 0.02


def test_bar_is_configurable_in_both_directions():
    before, after = _reports(0.1114, 0.1003)   # a 9.97% loss

    assert decide(before, after, max_relative_ranking_loss=0.50)["adopt"] is True
    assert decide(before, after, max_relative_ranking_loss=0.001)["adopt"] is False


def test_ranking_check_is_skipped_when_not_measured():
    """
    Backward compatibility: a caller that supplies no pr_auc must behave exactly
    as before rather than being rejected for missing data.
    """

    before = {"ece": 0.0300, "brier": 0.0340, "ece_noise_floor": 0.0024}
    after = {"ece": 0.0020, "brier": 0.0250}

    verdict = decide(before, after)

    assert verdict["adopt"] is True, verdict["reason"]
    assert verdict["ranking_checked"] is False
    assert not np.isfinite(verdict["relative_ranking_loss"])


def test_ranking_check_skipped_on_degenerate_pr_auc():
    """A zero or non-finite PR-AUC must not make the loss ratio explode."""

    for bad in (0.0, float("nan")):
        verdict = decide(*_reports(bad, 0.10))
        assert verdict["ranking_checked"] is False
        assert verdict["adopt"] is True


# ----------------------------------------------------------------------
# calibration_report now carries the ranking metrics the guard reads
# ----------------------------------------------------------------------


def test_calibration_report_carries_finite_ranking_metrics():
    rng = np.random.default_rng(0)
    targets = (rng.random(2000) < 0.05).astype(np.int64)
    probabilities = np.clip(rng.beta(1.2, 30.0, 2000) + 0.05 * targets, 0.0, 1.0)

    report = calibration_report(probabilities, targets)

    assert np.isfinite(report["pr_auc"])
    assert np.isfinite(report["roc_auc"])
    # A weakly informative score must beat the base rate but not be perfect.
    assert float(targets.mean()) < report["pr_auc"] < 1.0


def test_calibration_report_ranking_is_nan_for_single_class():
    """
    Ranking is undefined with one class present. It must be NaN rather than 0.0,
    because 0.0 would look like a catastrophic ranking loss to the guard.
    """

    targets = np.zeros(200, dtype=np.int64)
    probabilities = np.linspace(0.01, 0.5, 200)

    report = calibration_report(probabilities, targets)

    assert not np.isfinite(report["pr_auc"])
    assert not np.isfinite(report["roc_auc"])


# ----------------------------------------------------------------------
# The mechanism the guard defends against
# ----------------------------------------------------------------------


def test_isotonic_loses_ranking_to_ties_while_platt_does_not():
    """
    Demonstrates why the guard is needed at all: fit both calibrators on one
    score distribution, then apply them to a DIFFERENT one, as happens when a
    policy is carried over to another checkpoint.

    Platt must preserve PR-AUC exactly. Isotonic must not be assumed to.
    """

    from sklearn.metrics import average_precision_score

    from backend.ml.evaluation.calibration import fit_final_calibrator

    rng = np.random.default_rng(3)

    # Distribution the calibrators are FITTED on.
    fit_targets = (rng.random(4000) < 0.05).astype(np.int64)
    fit_scores = np.clip(rng.beta(1.2, 30.0, 4000) + 0.05 * fit_targets, 0.0, 1.0)

    # A DIFFERENT distribution they are then APPLIED to.
    new_targets = (rng.random(4000) < 0.05).astype(np.int64)
    new_scores = np.clip(rng.beta(1.1, 8.0, 4000) + 0.05 * new_targets, 0.0, 1.0)

    reference = average_precision_score(new_targets, new_scores)

    platt = fit_final_calibrator(fit_scores, fit_targets, method="platt")
    isotonic = fit_final_calibrator(fit_scores, fit_targets, method="isotonic")

    platt_pr = average_precision_score(new_targets, platt.predict(new_scores))
    isotonic_pr = average_precision_score(new_targets, isotonic.predict(new_scores))

    # Platt is strictly monotone, so the ordering — and therefore PR-AUC — is
    # untouched no matter what distribution it is applied to.
    assert platt_pr == pytest.approx(reference, abs=1e-9), (
        "Platt changed the ranking, which contradicts its documented property"
    )

    # Isotonic collapses distinct scores onto shared levels. Assert the tie
    # creation directly, since that is the mechanism; the PR-AUC consequence
    # depends on where the ties land.
    distinct_before = np.unique(new_scores).size
    distinct_after = np.unique(isotonic.predict(new_scores)).size

    assert distinct_after < distinct_before, (
        f"isotonic produced {distinct_after} distinct values from "
        f"{distinct_before}; no ties were created, so this fixture no longer "
        f"exercises the mechanism the guard exists for"
    )
    assert isotonic_pr <= reference + 1e-9
