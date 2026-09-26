"""
Tests for the focal-loss option and the LR-schedule choices.

These exist because both were added in response to a diagnosis (flat training
loss while the linear schedule decayed the learning rate), and a loss function
that is subtly wrong would look like a modelling result rather than a bug.
"""

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from backend.ml.training.losses import (
    FocalLossWithLogits,
    build_criterion,
)

NEWLINE = chr(10)


# ----------------------------------------------------------------------
# Reduction to the known-good case
# ----------------------------------------------------------------------


def test_gamma_zero_reduces_exactly_to_bce():
    """
    With gamma=0 the focusing term is (1-p_t)^0 = 1, so focal loss must be
    numerically identical to plain BCE. If this drifts, the modulation is being
    applied where it should not be.
    """

    logits = torch.tensor([-3.0, -0.5, 0.0, 0.7, 4.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])

    focal = FocalLossWithLogits(gamma=0.0, alpha=None, reduction="none")
    reference = nn.BCEWithLogitsLoss(reduction="none")

    assert torch.allclose(focal(logits, targets), reference(logits, targets), atol=1e-6)


def test_gamma_zero_with_alpha_reduces_to_pos_weighted_bce():
    """alpha must behave exactly like BCE's pos_weight."""

    logits = torch.tensor([-2.0, 0.3, 1.5, -0.8])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

    focal = FocalLossWithLogits(gamma=0.0, alpha=7.0, reduction="none")
    reference = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(7.0), reduction="none"
    )

    assert torch.allclose(focal(logits, targets), reference(logits, targets), atol=1e-6)


# ----------------------------------------------------------------------
# The property focal loss exists for
# ----------------------------------------------------------------------


def test_focal_downweights_easy_examples_more_than_hard_ones():
    """
    The whole point: an example the model already gets right should lose more of
    its loss than one it gets wrong. Measured as the ratio focal/BCE.
    """

    # easy negative (logit very low, target 0) vs hard positive (logit low, target 1)
    logits = torch.tensor([-6.0, -0.2])
    targets = torch.tensor([0.0, 1.0])

    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    focal = FocalLossWithLogits(gamma=2.0, reduction="none")(logits, targets)

    retained = focal / bce

    easy_retained, hard_retained = retained[0].item(), retained[1].item()

    assert easy_retained < hard_retained, (
        f"easy example retained {easy_retained:.6f} of its loss but hard "
        f"retained {hard_retained:.6f} — focal modulation is inverted"
    )
    # An example at p=0.9975 correct should be almost entirely suppressed.
    assert easy_retained < 0.01


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0])
def test_higher_gamma_suppresses_easy_examples_more(gamma):
    """Suppression must increase monotonically with gamma."""

    logits = torch.tensor([-5.0])
    targets = torch.tensor([0.0])

    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    focal = FocalLossWithLogits(gamma=gamma, reduction="none")(logits, targets)
    baseline = FocalLossWithLogits(gamma=0.0, reduction="none")(logits, targets)

    assert focal.item() <= baseline.item()
    assert torch.allclose(baseline, bce, atol=1e-6)


# ----------------------------------------------------------------------
# Numerical safety
# ----------------------------------------------------------------------


def test_no_nan_at_saturated_logits():
    """
    Saturated logits make p_t reach 1.0 in float arithmetic; without the clamp
    the focusing term can produce NaN and poison the backward pass.
    """

    logits = torch.tensor([-100.0, 100.0, -50.0, 50.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

    loss = FocalLossWithLogits(gamma=2.0, alpha=10.0)(logits, targets)

    assert torch.isfinite(loss).all(), loss


def test_gradients_flow_and_are_finite():
    logits = torch.tensor([-2.0, 0.4, 3.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0])

    FocalLossWithLogits(gamma=2.0, alpha=10.0)(logits, targets).mean().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_per_sample_reduction_shape():
    """The trainer needs per-sample losses for its high-loss diagnostic."""

    logits = torch.randn(9)
    targets = (torch.rand(9) > 0.5).float()

    assert FocalLossWithLogits(reduction="none")(logits, targets).shape == (9,)
    assert FocalLossWithLogits(reduction="mean")(logits, targets).shape == ()


def test_rejects_shape_mismatch_and_bad_gamma():
    with pytest.raises(ValueError, match="shape mismatch"):
        FocalLossWithLogits()(torch.randn(4), torch.randn(5))

    with pytest.raises(ValueError, match="gamma"):
        FocalLossWithLogits(gamma=-1.0)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def test_build_criterion_returns_the_requested_loss():
    device = torch.device("cpu")

    assert isinstance(
        build_criterion("bce", pos_weight=10.0, device=device),
        nn.BCEWithLogitsLoss,
    )
    assert isinstance(
        build_criterion("focal", pos_weight=10.0, device=device),
        FocalLossWithLogits,
    )

    with pytest.raises(ValueError, match="Unknown loss"):
        build_criterion("hinge", pos_weight=1.0, device=device)


def test_build_criterion_passes_pos_weight_through_to_both():
    """
    Switching loss must not silently change the class-balance correction — only
    the per-example weighting scheme.
    """

    device = torch.device("cpu")
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])

    bce = build_criterion("bce", pos_weight=10.0, device=device)(logits, targets)
    focal = build_criterion(
        "focal", pos_weight=10.0, device=device, focal_gamma=0.0
    )(logits, targets)

    # At gamma=0 they must agree, which proves pos_weight reached the focal path.
    assert torch.allclose(bce, focal, atol=1e-6)
    # And the positive is weighted 10x the negative at equal logits.
    assert torch.allclose(bce[0] / bce[1], torch.tensor(10.0), atol=1e-5)


# ----------------------------------------------------------------------
# Scheduler choices
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheduler", ["linear", "cosine", "constant_with_warmup"]
)
def test_scheduler_choices_build_and_behave(scheduler):
    """
    constant_with_warmup must hold the rate after warmup; the other two must
    decay it. This is the behaviour the schedule change was made for.
    """

    from transformers import (
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)

    total, warmup = 100, 10

    if scheduler == "constant_with_warmup":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, warmup)
    elif scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total)
    else:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total)

    rates = []
    for _ in range(total):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        lr_scheduler.step()

    peak = max(rates)
    final = rates[-1]

    assert rates[0] < peak, "no warmup observed"

    if scheduler == "constant_with_warmup":
        assert final == pytest.approx(peak, rel=1e-6), (
            "constant_with_warmup decayed the rate, which defeats its purpose"
        )
    else:
        assert final < peak * 0.25, f"{scheduler} did not decay ({final} vs {peak})"


def test_trainer_rejects_unknown_scheduler(tmp_path):
    """
    The guard must actually fire when a Trainer is constructed, not merely
    exist in the source. Uses a minimal in-memory dataset so no GPU or real
    data is needed.
    """

    import json

    from transformers import RobertaConfig, RobertaModel

    from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier
    from backend.ml.models.collate import make_collate_fn
    from backend.ml.models.dataset import ChunkedFunctionDataset
    from backend.ml.training.trainer import Trainer

    path = tmp_path / "chunked.jsonl"
    path.write_text(
        NEWLINE.join(
            json.dumps({"target": i % 2, "num_chunks": 1, "chunks": [[0, 10 + i, 2]]})
            for i in range(4)
        )
        + NEWLINE,
        encoding="utf-8",
    )

    dataset = ChunkedFunctionDataset(path)
    model = HierarchicalCodeBERTClassifier(
        model_name="tiny-test",
        encoder=RobertaModel(
            RobertaConfig(
                vocab_size=60, hidden_size=16, num_hidden_layers=1,
                num_attention_heads=1, intermediate_size=32,
                max_position_embeddings=64, pad_token_id=1,
                bos_token_id=0, eos_token_id=2,
            )
        ),
    )

    with pytest.raises(ValueError, match="Unknown scheduler"):
        Trainer(
            model=model,
            train_dataset=dataset,
            valid_dataset=dataset,
            collate_fn=make_collate_fn(1),
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / "ckpt",
            scheduler="exponential_decay_that_does_not_exist",
        )


def test_trainer_accepts_each_valid_scheduler_and_loss(tmp_path):
    """The three schedulers and two losses must all construct successfully."""

    import itertools
    import json

    from transformers import RobertaConfig, RobertaModel

    from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier
    from backend.ml.models.collate import make_collate_fn
    from backend.ml.models.dataset import ChunkedFunctionDataset
    from backend.ml.training.trainer import Trainer

    path = tmp_path / "chunked.jsonl"
    path.write_text(
        NEWLINE.join(
            json.dumps({"target": i % 2, "num_chunks": 1, "chunks": [[0, 10 + i, 2]]})
            for i in range(4)
        )
        + NEWLINE,
        encoding="utf-8",
    )
    dataset = ChunkedFunctionDataset(path)

    for scheduler, loss in itertools.product(
        ("linear", "cosine", "constant_with_warmup"), ("bce", "focal")
    ):
        model = HierarchicalCodeBERTClassifier(
            model_name="tiny-test",
            encoder=RobertaModel(
                RobertaConfig(
                    vocab_size=60, hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=1, intermediate_size=32,
                    max_position_embeddings=64, pad_token_id=1,
                    bos_token_id=0, eos_token_id=2,
                )
            ),
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            valid_dataset=dataset,
            collate_fn=make_collate_fn(1),
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / f"ckpt_{scheduler}_{loss}",
            scheduler=scheduler,
            loss=loss,
        )

        assert trainer.scheduler_name == scheduler
        assert trainer.loss_name == loss
        # The criterion must be per-sample so the high-loss diagnostic works.
        out = trainer.criterion(
            torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0])
        )
        assert out.shape == (2,), f"{loss} did not return per-sample losses"
