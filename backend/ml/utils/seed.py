"""Reproducible seeding across Python, NumPy, PyTorch and CUDA."""

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every RNG the training pipeline touches.

    Full bit-for-bit reproducibility is NOT guaranteed even with this call:
    several cuDNN/cuBLAS kernels are non-deterministic by design, atomics in
    reduction kernels change summation order between runs, and mixed precision
    makes that ordering visible in the low bits. `deterministic=True` sets the
    documented flags, but PyTorch may still select a non-deterministic kernel
    unless CUBLAS_WORKSPACE_CONFIG is also exported before the process starts.
    Treat runs as statistically, not exactly, reproducible.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
