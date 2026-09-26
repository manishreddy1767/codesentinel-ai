"""
Loss functions for extreme class imbalance.

Two options, both returning **per-sample** losses (`reduction="none"`) so the
trainer can take the mean itself and still surface per-sample values to the
high-loss diagnostic.

Why a second option exists
--------------------------
`BCEWithLogitsLoss(pos_weight=w)` multiplies the loss of *every* positive by
the same constant. At a 3% positive rate that fixes the gradient imbalance
between classes, but it does nothing about the imbalance *within* a class: the
thousands of easy negatives the model already classifies confidently still
contribute the bulk of the summed gradient, because there are so many of them.

Focal loss (Lin et al., 2017, "Focal Loss for Dense Object Detection") instead
scales each sample by ``(1 - p_t) ** gamma``, where ``p_t`` is the probability
assigned to the *correct* class. A negative the model already scores at 0.01
gets its loss multiplied by ``0.01 ** gamma`` - effectively removed - while a
positive the model scores at 0.1 keeps almost all of its loss. The capacity
goes to the examples that are still wrong.

This is a hypothesis to be tested on validation, not an assumed improvement.
The vulnerability-detection literature reports mixed results for focal loss,
and on this dataset the diagnostic pointing to it is underfitting (flat
training loss across an epoch boundary) rather than easy-negative swamping. It
is offered as a tunable choice so the data decides.
"""

import torch
from torch import nn


class FocalLossWithLogits(nn.Module):
    """
    Binary focal loss on raw logits, with an optional positive-class weight.

    Args:
        gamma: focusing exponent. 0.0 reduces exactly to weighted BCE; the
            paper's default for dense detection is 2.0. Higher values suppress
            easy examples more aggressively.
        alpha: weight applied to the positive class, playing the same role as
            ``pos_weight`` in BCE. Passed as a raw ratio (e.g. 10.0), not the
            paper's [0, 1] convention, so it is directly comparable with the
            ``pos_weight`` the rest of the pipeline computes.
        reduction: "none" (default, required by the trainer), "mean" or "sum".
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 reduction: str = "none"):
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unknown reduction: {reduction}")

        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"shape mismatch: logits {tuple(logits.shape)} vs targets "
                f"{tuple(targets.shape)}"
            )

        targets = targets.to(logits.dtype)

        # Unreduced BCE, computed in a numerically stable way by PyTorch rather
        # than by taking log(sigmoid(x)) ourselves.
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = probability assigned to the correct class.
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)

        # Clamp keeps (1 - p_t) ** gamma finite when p_t saturates at 1.0 under
        # fp16 autocast; without it a perfectly-classified sample can produce
        # 0 ** negative-ish and propagate NaN through the backward pass.
        focal = (1.0 - p_t).clamp(min=1e-7) ** self.gamma

        loss = focal * bce

        if self.alpha is not None:
            # Same semantics as BCE's pos_weight: scale positives only.
            weight = torch.where(
                targets > 0.5,
                torch.as_tensor(self.alpha, dtype=loss.dtype, device=loss.device),
                torch.ones((), dtype=loss.dtype, device=loss.device),
            )
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()

        return loss


def build_criterion(
    name: str,
    pos_weight: float,
    device,
    focal_gamma: float = 2.0,
    reduction: str = "none",
):
    """
    Build the per-sample loss the trainer expects.

    `name` is "bce" or "focal". Both apply the positive-class weight the caller
    computed from the training split, so switching between them changes the
    per-example weighting scheme without silently changing the class balance
    correction.
    """

    name = (name or "bce").strip().lower()

    if name == "bce":
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device),
            reduction=reduction,
        )

    if name == "focal":
        return FocalLossWithLogits(
            gamma=focal_gamma, alpha=pos_weight, reduction=reduction
        ).to(device)

    raise ValueError(f"Unknown loss: {name!r}. Choose 'bce' or 'focal'.")
