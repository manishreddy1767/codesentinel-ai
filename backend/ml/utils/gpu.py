"""GPU memory reporting helpers."""


def memory_report(device=None) -> str:
    """
    One-line CUDA memory summary.

    Returns a CPU notice when CUDA is unavailable so callers never branch.
    """

    try:
        import torch
    except ImportError:
        return "GPU memory | torch not installed"

    if not torch.cuda.is_available():
        return "GPU memory | CUDA not available (running on CPU)"

    mib = 1024 * 1024

    allocated = torch.cuda.memory_allocated(device) / mib
    reserved = torch.cuda.memory_reserved(device) / mib
    peak = torch.cuda.max_memory_allocated(device) / mib

    return (
        f"GPU memory | Allocated: {allocated:7.0f} MiB"
        f" | Reserved: {reserved:7.0f} MiB"
        f" | Peak: {peak:7.0f} MiB"
    )


def reset_peak_memory(device=None) -> None:
    """Reset the peak-allocation counter, e.g. at the start of an epoch."""

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def resolve_device(prefer_cuda: bool = True):
    """Return the training device, falling back to CPU when CUDA is absent."""

    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
