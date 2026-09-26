"""
Hyperparameter search space for the CodeBERT vulnerability classifier.

Kept separate from the search driver so the space is reviewable on its own and
identical whether Optuna or the built-in random search is running it.

Why the classification threshold is NOT in here
-----------------------------------------------
Tuning the threshold jointly with the model hyperparameters lets the search
pick a configuration that only looks good at one lucky operating point, and
makes the reported validation score an optimistically-biased maximum over two
nested searches. So the objective is PR-AUC, which is threshold-free: it scores
the model's *ranking*, and cannot be gamed by moving a cutoff. The threshold is
chosen afterwards, once, from the winning model's validation predictions
(Stage N).

Ranges are deliberately conservative. The published CodeBERT fine-tuning
recipes live around 1e-5 to 5e-5, and the low end here exists to let the search
back off rather than to be selected.
"""

# name -> (kind, specification)
#
#   ("loguniform", (low, high))    sampled in log space; right for learning rates
#   ("uniform",    (low, high))
#   ("categorical", [choices...])
SEARCH_SPACE = {
    # The single most important knob for transformer fine-tuning.
    "learning_rate": ("loguniform", (1e-6, 5e-5)),
    # Regularises the classification head, which is the part most able to
    # memorise on a small positive class.
    "dropout": ("uniform", (0.1, 0.5)),
    "weight_decay": ("uniform", (0.0, 0.1)),
    "warmup_ratio": ("uniform", (0.0, 0.15)),
    # Effective batch = batch_size * grad_accum. Both are searched because the
    # same effective batch behaves differently depending on how it is reached:
    # BatchNorm-free transformers are mostly indifferent, but memory is not.
    "batch_size": ("categorical", [2, 4, 8]),
    "grad_accum_steps": ("categorical", [1, 2, 4]),
}

# Architecture variants, searched only when explicitly enabled. The baseline
# ("mean") is always a candidate so a comparison can never silently discard it.
ARCHITECTURE_SPACE = {
    "function_pooling": ("categorical", ["mean", "max", "attention"]),
    "chunk_pooling": ("categorical", ["cls", "mean"]),
}

# Memory-shaping parameters. Searching these trades compute for VRAM rather
# than for accuracy, so they are opt-in and default to the measured-safe value.
MEMORY_SPACE = {
    "chunk_micro_batch": ("categorical", [2, 4, 8, 16]),
}


def build_space(include_architecture: bool = False, include_memory: bool = False) -> dict:
    space = dict(SEARCH_SPACE)

    if include_architecture:
        space.update(ARCHITECTURE_SPACE)

    if include_memory:
        space.update(MEMORY_SPACE)

    return space


def sample(space: dict, rng) -> dict:
    """Draw one configuration using a numpy Generator. Used by the fallback search."""

    import numpy as np

    parameters = {}

    for name, (kind, specification) in space.items():
        if kind == "loguniform":
            low, high = specification
            parameters[name] = float(
                np.exp(rng.uniform(np.log(low), np.log(high)))
            )
        elif kind == "uniform":
            low, high = specification
            parameters[name] = float(rng.uniform(low, high))
        elif kind == "categorical":
            # .item() so the value is a plain Python scalar and stays
            # JSON-serialisable.
            parameters[name] = specification[int(rng.integers(len(specification)))]
        else:
            raise ValueError(f"Unknown parameter kind: {kind}")

    return parameters


def suggest_optuna(space: dict, trial) -> dict:
    """Draw one configuration from an Optuna trial."""

    parameters = {}

    for name, (kind, specification) in space.items():
        if kind == "loguniform":
            low, high = specification
            parameters[name] = trial.suggest_float(name, low, high, log=True)
        elif kind == "uniform":
            low, high = specification
            parameters[name] = trial.suggest_float(name, low, high)
        elif kind == "categorical":
            parameters[name] = trial.suggest_categorical(name, specification)
        else:
            raise ValueError(f"Unknown parameter kind: {kind}")

    return parameters
