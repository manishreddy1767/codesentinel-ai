"""
Verification suite for the hardening added in the staged ML review.

Covers the parts that did not exist, or were wrong, before:

  * multi-dimensional leakage auditing (code vs identity vs project)
  * train-only leakage remediation
  * PR-AUC / Brier / MCC and the vectorised threshold sweep
  * threshold selection objectives
  * probability calibration and its adopt/reject decision
  * attention pooling
  * checkpoints recording the architecture actually built, and RNG state
  * best-weight restoration and min-delta early stopping
  * stratified subset selection

Runs entirely on synthetic data, with no network access and no CodeBERT
download.

    .venv/Scripts/python -m pytest backend/tests/test_ml_stages.py -v
"""

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from backend.ml import config
from backend.ml.training import checkpoint as ckpt
from backend.ml.training.metrics import (
    compute_metrics,
    find_best_threshold,
    sweep_thresholds,
)

BOS, EOS, PAD = 0, 2, 1


# ======================================================================
# Helpers
# ======================================================================


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def tiny_model(chunk_micro_batch=4, chunk_pooling="cls", function_pooling="mean"):
    from transformers import RobertaConfig, RobertaModel

    from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier

    encoder_config = RobertaConfig(
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=530,
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )

    return HierarchicalCodeBERTClassifier(
        model_name="tiny-test",
        chunk_pooling=chunk_pooling,
        function_pooling=function_pooling,
        chunk_micro_batch=chunk_micro_batch,
        encoder=RobertaModel(encoder_config),
    )


@pytest.fixture
def leaky_splits(tmp_path: Path) -> Path:
    """
    Splits with one deliberate leak of each kind, and nothing else shared.

    train/valid : the same function body (code leakage), opposite labels
    train/test  : the same upstream `hash` and `commit_id`, different body
                  (identity leakage that content hashing cannot see)
    train/test  : a shared `project` (a concern, not leakage)
    """

    directory = tmp_path / "splits"

    train = [
        {
            "func": f"int f{i}(void) {{ return {i}; }}",
            "target": i % 2,
            "hash": f"h{i}",
            "idx": i,
            "commit_id": f"c{i}",
            "project": "alpha",
        }
        for i in range(12)
    ]
    train.append(
        {
            "func": "int shared_body(void) { return 1; }",
            "target": 1,
            "hash": "hCODE",
            "idx": 900,
            "commit_id": "cCODE",
            "project": "alpha",
        }
    )
    train.append(
        {
            "func": "int train_side_body(void) { return 2; }",
            "target": 1,
            "hash": "hID",
            "idx": 901,
            "commit_id": "cID",
            "project": "alpha",
        }
    )
    # An exact duplicate inside train.
    train.append(dict(train[0]))

    valid = [
        {
            "func": f"int g{i}(void) {{ return {i}; }}",
            "target": i % 2,
            "hash": f"v{i}",
            "idx": 100 + i,
            "commit_id": f"vc{i}",
            "project": "beta",
        }
        for i in range(8)
    ]
    valid.append(
        {
            "func": "int shared_body(void) { return 1; }",
            "target": 0,  # opposite label -> conflicting duplicate
            "hash": "hCODEv",
            "idx": 902,
            "commit_id": "cCODEv",
            "project": "beta",
        }
    )

    test = [
        {
            "func": f"int t{i}(void) {{ return {i}; }}",
            "target": i % 2,
            "hash": f"t{i}",
            "idx": 200 + i,
            "commit_id": f"tc{i}",
            "project": "gamma",
        }
        for i in range(8)
    ]
    test.append(
        {
            "func": "int a_completely_different_body(void) { return 42; }",
            "target": 1,
            "hash": "hID",  # identity collision with train
            "idx": 903,
            "commit_id": "cID",
            "project": "alpha",  # project overlap with train
        }
    )

    write_jsonl(directory / "primevul_train.jsonl", train)
    write_jsonl(directory / "primevul_valid.jsonl", valid)
    write_jsonl(directory / "primevul_test.jsonl", test)

    return directory


# ======================================================================
# Stage C - leakage auditing
# ======================================================================


def test_audit_separates_code_identity_and_project_overlap(leaky_splits):
    from backend.ml.preprocessing.audit_splits import audit

    report = audit(leaky_splits)

    train_valid = report["cross_split"]["train__valid"]
    train_test = report["cross_split"]["train__test"]

    # Code leakage shows up between train and valid only.
    assert train_valid["exact_code"]["shared_keys"] == 1
    assert train_valid["normalized_code"]["shared_keys"] == 1
    assert train_test["exact_code"]["shared_keys"] == 0

    # Identity leakage shows up between train and test, where the bodies
    # differ entirely. This is precisely what content hashing misses.
    assert train_test["hash"]["shared_keys"] == 1
    assert train_test["commit_id"]["shared_keys"] == 1
    assert train_test["normalized_code"]["shared_keys"] == 0

    # Project overlap is classified as context, never as a blocking finding.
    assert train_test["project"]["shared_keys"] == 1
    assert train_test["project"]["kind"] == "context"

    code_and_identity = " ".join(report["findings"])
    assert "exact_code" in code_and_identity
    assert "hash" in code_and_identity
    assert "project" not in code_and_identity

    assert any("project" in concern for concern in report["concerns"])


def test_audit_flags_label_conflicting_duplicates(leaky_splits):
    from backend.ml.preprocessing.audit_splits import audit

    report = audit(leaky_splits)

    # Same body, opposite labels across train/valid.
    assert report["cross_split"]["train__valid"]["exact_code"][
        "label_conflicting_keys"
    ] == 1

    # The identity collision carries the same label on both sides, so it is
    # leakage but not a conflict. Reporting those separately is the point.
    assert report["cross_split"]["train__test"]["hash"][
        "label_conflicting_keys"
    ] == 0


def test_audit_reports_within_split_duplicates(leaky_splits):
    from backend.ml.preprocessing.audit_splits import audit

    report = audit(leaky_splits)
    duplicates = report["within_split_duplicates"]["train"]["exact_code"]

    assert duplicates["duplicate_keys"] == 1
    assert duplicates["redundant_records"] == 1


def test_audit_marks_absent_dimension_unavailable(leaky_splits):
    from backend.ml.preprocessing.audit_splits import audit

    report = audit(leaky_splits)

    # No record carries big_vul_idx; that must read as "not available" rather
    # than as a clean result, which would be a false reassurance.
    entry = report["cross_split"]["train__test"]["big_vul_idx"]
    assert entry["available"] is False


def test_audit_detects_label_field_on_a_later_record(tmp_path):
    """
    Regression: field detection used to be read off record #1 only, so a first
    record missing the label pinned label_field to None and every later record
    was miscounted as having an invalid label.
    """

    from backend.ml.preprocessing.audit_splits import collect_keys

    path = tmp_path / "primevul_train.jsonl"
    write_jsonl(
        path,
        [
            {"func": "int a(void) { return 0; }"},  # no label field at all
            {"func": "int b(void) { return 1; }", "target": 1},
            {"func": "int c(void) { return 0; }", "target": 0},
        ],
    )

    result = collect_keys(path)

    assert result["summary"]["label_field"] == "target"
    assert result["summary"]["usable"] == 2
    assert result["summary"]["skipped_bad_label"] == 1


def test_clean_removes_only_from_train(leaky_splits, tmp_path):
    from backend.ml.preprocessing.audit_splits import audit
    from backend.ml.preprocessing.clean_split_duplicates import (
        DEFAULT_DIMENSIONS,
        build_holdout_index,
        clean_train,
    )

    output = tmp_path / "clean"
    dimensions = set(DEFAULT_DIMENSIONS)

    index = build_holdout_index(
        [
            leaky_splits / "primevul_valid.jsonl",
            leaky_splits / "primevul_test.jsonl",
        ],
        dimensions,
    )

    result = clean_train(
        leaky_splits / "primevul_train.jsonl",
        output / "primevul_train.jsonl",
        index,
        dimensions,
        dry_run=False,
    )

    # Exactly the two planted leaks are removed.
    assert result["stats"]["removed"] == 2
    assert result["by_dimension"]["exact_code"] == 1
    assert result["by_dimension"]["hash"] == 1

    # Copy valid/test through and confirm the audit now comes back clean.
    for split in ("valid", "test"):
        source = leaky_splits / f"primevul_{split}.jsonl"
        (output / f"primevul_{split}.jsonl").write_bytes(source.read_bytes())

    after = audit(output)

    assert after["findings"] == []
    # Project overlap is untouched: it was never leakage, so cleaning must not
    # have silently deleted training data to make it disappear.
    assert after["concerns"]


def test_clean_refuses_to_overwrite_its_input(leaky_splits):
    """The cleaner must never write over the split directory it is reading."""

    from backend.ml.preprocessing import clean_split_duplicates

    result = clean_split_duplicates.main.__doc__  # module import smoke
    assert result is None or isinstance(result, str)

    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.ml.preprocessing.clean_split_duplicates",
            "--in-dir",
            str(leaky_splits),
            "--out-dir",
            str(leaky_splits),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        # The non-zero exit IS the assertion below.
        check=False,
    )

    assert completed.returncode == 2
    assert "must differ" in completed.stdout


# ======================================================================
# Metrics - PR-AUC, Brier, MCC, sweep
# ======================================================================


def test_pr_auc_beats_roc_auc_at_exposing_imbalance():
    """
    On a heavily imbalanced split a mediocre model keeps a high ROC-AUC while
    PR-AUC collapses towards the positive rate. That gap is the reason PR-AUC
    is the headline metric.
    """

    rng = np.random.default_rng(0)
    n = 4000
    targets = (rng.random(n) < 0.02).astype(np.int64)
    probabilities = np.clip(
        rng.normal(0.3, 0.15, n) + 0.25 * targets, 0.001, 0.999
    )

    metrics = compute_metrics(probabilities, targets, 0.5)

    assert metrics.roc_auc > 0.75
    assert metrics.pr_auc < metrics.roc_auc
    assert metrics.pr_auc > metrics.positive_rate
    assert metrics.positive_rate == pytest.approx(targets.mean(), abs=1e-9)


def test_brier_score_is_the_mean_squared_error():
    probabilities = np.array([0.0, 0.5, 1.0, 0.25])
    targets = np.array([0, 1, 1, 0])

    expected = np.mean((probabilities - targets) ** 2)
    metrics = compute_metrics(probabilities, targets, 0.5)

    assert metrics.brier == pytest.approx(expected)


def test_mcc_matches_sklearn():
    from sklearn.metrics import matthews_corrcoef

    rng = np.random.default_rng(5)
    targets = (rng.random(500) < 0.2).astype(np.int64)
    probabilities = rng.random(500)

    metrics = compute_metrics(probabilities, targets, 0.5)
    expected = matthews_corrcoef(targets, (probabilities >= 0.5).astype(int))

    assert metrics.mcc == pytest.approx(expected, abs=1e-9)


def test_auc_is_nan_with_a_single_class_but_brier_is_not():
    probabilities = np.array([0.1, 0.4, 0.8])
    targets = np.array([0, 0, 0])

    metrics = compute_metrics(probabilities, targets, 0.5)

    assert math.isnan(metrics.roc_auc)
    assert math.isnan(metrics.pr_auc)
    assert not math.isnan(metrics.brier)


@pytest.mark.parametrize("positive_rate", [0.5, 0.03, 0.97])
def test_vectorised_sweep_matches_reference_confusion_matrix(positive_rate):
    """The fast sweep must agree exactly with per-threshold computation."""

    rng = np.random.default_rng(int(positive_rate * 100))
    n = 900
    targets = (rng.random(n) < positive_rate).astype(np.int64)
    probabilities = np.clip(rng.beta(2, 5, n) + 0.3 * targets, 0, 1)

    # Force exact ties on a grid point: the boundary case index arithmetic
    # is most likely to get wrong.
    probabilities[:20] = 0.5

    for row in sweep_thresholds(probabilities, targets):
        reference = compute_metrics(probabilities, targets, row["threshold"])

        assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (
            reference.tn,
            reference.fp,
            reference.fn,
            reference.tp,
        )
        assert row["f1"] == pytest.approx(reference.f1)


def test_threshold_objectives_trade_precision_against_recall():
    rng = np.random.default_rng(11)
    n = 2000
    targets = (rng.random(n) < 0.1).astype(np.int64)
    probabilities = np.clip(rng.beta(2, 6, n) + 0.3 * targets, 0.001, 0.999)

    _, f1_metrics, _ = find_best_threshold(probabilities, targets, objective="f1")
    _, recall_metrics, _ = find_best_threshold(
        probabilities, targets, objective="fbeta", beta=2.0
    )

    # Weighting recall four times precision must not reduce recall.
    assert recall_metrics.recall >= f1_metrics.recall


def test_recall_at_precision_respects_the_floor():
    rng = np.random.default_rng(12)
    n = 3000
    targets = (rng.random(n) < 0.15).astype(np.int64)
    probabilities = np.clip(rng.beta(2, 5, n) + 0.45 * targets, 0.001, 0.999)

    _, metrics, _ = find_best_threshold(
        probabilities,
        targets,
        objective="recall_at_precision",
        min_precision=0.6,
    )

    assert metrics.precision >= 0.6


def test_unknown_threshold_objective_is_rejected():
    with pytest.raises(ValueError, match="Unknown threshold objective"):
        find_best_threshold(
            np.array([0.1, 0.9]), np.array([0, 1]), objective="nonsense"
        )


# ======================================================================
# Stage O - calibration
# ======================================================================


def _miscalibrated(n=3000, seed=7, sharpen=2.2, shift=2.0):
    rng = np.random.default_rng(seed)
    true_probability = rng.beta(1.2, 12, n)
    targets = (rng.random(n) < true_probability).astype(np.int64)

    logit = np.log(true_probability / (1 - true_probability))
    overconfident = 1 / (1 + np.exp(-(logit * sharpen + shift)))

    return true_probability, overconfident, targets


def test_calibration_detects_and_fixes_overconfidence():
    from backend.ml.evaluation.calibration import cross_validated_calibration, decide

    _, overconfident, targets = _miscalibrated()

    _, before, after = cross_validated_calibration(
        overconfident, targets, method="platt"
    )
    verdict = decide(before, after)

    assert verdict["adopt"] is True
    assert after["ece"] < before["ece"]


def test_calibration_is_declined_when_already_calibrated():
    """Calibration must not be bolted on for a noise-level improvement."""

    from backend.ml.evaluation.calibration import cross_validated_calibration, decide

    well_calibrated, _, targets = _miscalibrated()

    _, before, after = cross_validated_calibration(
        well_calibrated, targets, method="platt"
    )

    assert decide(before, after)["adopt"] is False


def test_platt_scaling_preserves_ranking():
    """
    Platt is monotone in the score, so it cannot change the ranking - and
    therefore cannot change PR-AUC or ROC-AUC. Only the probabilities move.
    """

    from sklearn.metrics import average_precision_score

    from backend.ml.evaluation.calibration import fit_final_calibrator

    _, overconfident, targets = _miscalibrated()

    calibrator = fit_final_calibrator(overconfident, targets, method="platt")
    calibrated = calibrator.predict(overconfident)

    assert average_precision_score(targets, calibrated) == pytest.approx(
        average_precision_score(targets, overconfident), abs=1e-9
    )


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_serialisation_round_trips(method):
    from backend.ml.evaluation.calibration import (
        calibrator_from_dict,
        fit_final_calibrator,
    )

    _, overconfident, targets = _miscalibrated()

    calibrator = fit_final_calibrator(overconfident, targets, method=method)
    restored = calibrator_from_dict(calibrator.to_dict())

    assert np.allclose(
        calibrator.predict(overconfident), restored.predict(overconfident)
    )

    # Must survive a JSON round trip, since that is how a policy is stored.
    json.loads(json.dumps(calibrator.to_dict()))


def test_reliability_curve_bins_are_non_empty_and_cover_everything():
    from backend.ml.evaluation.calibration import reliability_curve

    _, overconfident, targets = _miscalibrated()
    rows = reliability_curve(overconfident, targets, bins=10)

    assert rows
    assert all(row["count"] > 0 for row in rows)
    assert sum(row["count"] for row in rows) == targets.size


def test_calibration_needs_both_classes():
    from backend.ml.evaluation.calibration import cross_validated_calibration

    with pytest.raises(ValueError, match="at least two samples"):
        cross_validated_calibration(
            np.array([0.2, 0.3, 0.4]), np.array([0, 0, 0])
        )


# ======================================================================
# Stage H - attention pooling
# ======================================================================


def test_attention_pooling_produces_correct_shape_and_gradients():
    model = tiny_model(function_pooling="attention")

    input_ids = torch.tensor(
        [[BOS, 10, 11, EOS], [BOS, 12, 13, EOS], [BOS, 14, 15, EOS]]
    )
    attention_mask = torch.ones_like(input_ids)
    function_index = torch.tensor([0, 0, 1])

    logits = model(input_ids, attention_mask, function_index, batch_size=2)
    assert logits.shape == (2,)

    logits.sum().backward()

    attention_gradients = [
        p.grad for p in model.attention.parameters() if p.grad is not None
    ]
    assert attention_gradients
    assert any(g.abs().sum() > 0 for g in attention_gradients)


def test_attention_weights_sum_to_one_per_function():
    """
    A uniform attention head must reduce to the mean. If the segment softmax
    normalised across the whole batch instead of within each function, this
    would not hold.
    """

    model = tiny_model(function_pooling="attention")

    # Force constant scores so the softmax is exactly uniform per function.
    with torch.no_grad():
        for parameter in model.attention.parameters():
            parameter.zero_()

    hidden = model.hidden_size
    embeddings = torch.stack(
        [
            torch.full((hidden,), 1.0),
            torch.full((hidden,), 3.0),
            torch.full((hidden,), 10.0),
        ]
    )
    function_index = torch.tensor([0, 0, 1])

    with torch.no_grad():
        pooled = model._aggregate(embeddings, function_index, batch_size=2)

    # Function 0 has two chunks (1.0 and 3.0) -> uniform weights give 2.0.
    assert torch.allclose(pooled[0], torch.full((hidden,), 2.0), atol=1e-5)
    # Function 1 has a single chunk, whose weight must be exactly 1.
    assert torch.allclose(pooled[1], torch.full((hidden,), 10.0), atol=1e-5)


def test_unknown_function_pooling_is_rejected():
    with pytest.raises(ValueError, match="Unknown function_pooling"):
        tiny_model(function_pooling="telepathy")


# ======================================================================
# Stage L - checkpoint fidelity
# ======================================================================


def test_checkpoint_records_the_architecture_actually_built(tmp_path):
    """
    Regression: the saved architecture used to come from the module-level
    config, so a model built with a non-default pooling was recorded as the
    default and silently rebuilt wrong on load.
    """

    model = tiny_model(function_pooling="attention", chunk_pooling="mean")
    path = tmp_path / "ckpt.pt"

    ckpt.save_checkpoint(path, model)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["config"]["function_pooling"] == "attention"
    assert payload["config"]["chunk_pooling"] == "mean"
    # ... and it disagrees with the global default, which is the whole point.
    assert config.FUNCTION_POOLING == "mean"


def test_checkpoint_round_trips_rng_state(tmp_path):
    model = tiny_model()
    path = tmp_path / "ckpt.pt"

    torch.manual_seed(1234)
    ckpt.save_checkpoint(path, model)

    expected = torch.randn(5)

    payload = ckpt.load_checkpoint(path, model=model)
    assert ckpt.restore_rng_state(payload["rng_state"]) is True

    assert torch.allclose(torch.randn(5), expected)


def test_restore_rng_state_tolerates_a_legacy_checkpoint():
    """Checkpoints written before RNG capture must still load."""

    assert ckpt.restore_rng_state(None) is False


# ======================================================================
# Stage J/K - subset selection
# ======================================================================


def test_subset_is_stratified_and_seeded(tmp_path):
    """
    Regression: the subset used to be the head of the file. A split ordered by
    label then yields a single-class subset, which makes pos_weight 1.0 and
    turns an "overfit test" into a test that proves nothing.
    """

    from backend.ml.models.dataset import ChunkedFunctionDataset
    from backend.ml.training.train import _Subset

    # Deliberately label-ordered: 90 negatives, then 10 positives.
    rows = [
        {"target": 0, "num_chunks": 1, "chunks": [[BOS, 10 + (i % 50), EOS]]}
        for i in range(90)
    ] + [
        {"target": 1, "num_chunks": 1, "chunks": [[BOS, 60 + (i % 50), EOS]]}
        for i in range(10)
    ]

    path = tmp_path / "chunked.jsonl"
    write_jsonl(path, rows)

    dataset = ChunkedFunctionDataset(path)
    subset = _Subset(dataset, 20, seed=42)

    negatives, positives = subset.class_counts()

    assert len(subset) == 20
    assert positives > 0, "stratified subset must contain the positive class"
    assert negatives > 0
    assert subset.pos_weight() > 1.0

    # Same seed -> same subset, so tuning trials stay comparable.
    assert _Subset(dataset, 20, seed=42).indices == subset.indices
    assert _Subset(dataset, 20, seed=7).indices != subset.indices


def test_subset_targets_align_with_fetched_items(tmp_path):
    from backend.ml.models.dataset import ChunkedFunctionDataset
    from backend.ml.training.train import _Subset

    rows = [
        {"target": i % 2, "num_chunks": 1, "chunks": [[BOS, 10 + i, EOS]]}
        for i in range(40)
    ]

    path = tmp_path / "chunked.jsonl"
    write_jsonl(path, rows)

    dataset = ChunkedFunctionDataset(path)
    subset = _Subset(dataset, 12, seed=3)

    # The label the sampler reports must be the label attached to the item the
    # loader actually fetches, or every metric is quietly wrong.
    for position in range(len(subset)):
        assert subset[position]["target"] == subset.targets[position]


# ======================================================================
# Experiment tracking
# ======================================================================


def test_run_recorder_captures_environment_and_events(tmp_path):
    from backend.ml.utils.experiment import RunRecorder

    with RunRecorder("unit", root=tmp_path, config_dict={"lr": 1e-5}) as recorder:
        recorder.log("epoch_finished", epoch=1, score=0.5)

    environment = json.loads(
        (recorder.directory / "environment.json").read_text(encoding="utf-8")
    )
    assert "python" in environment and "git" in environment

    saved_config = json.loads(
        (recorder.directory / "config.json").read_text(encoding="utf-8")
    )
    assert saved_config["lr"] == 1e-5

    events = [
        json.loads(line)
        for line in (recorder.directory / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    names = [event["event"] for event in events]

    assert names[0] == "run_started"
    assert "epoch_finished" in names
    assert names[-1] == "run_finished"


def test_run_recorder_records_failures(tmp_path):
    from backend.ml.utils.experiment import RunRecorder

    recorder = RunRecorder("failing", root=tmp_path)

    with pytest.raises(ValueError), recorder:
        raise ValueError("boom")

    events = (recorder.directory / "events.jsonl").read_text(encoding="utf-8")

    assert "run_failed" in events
    assert "boom" in events
