"""
Experiment tracking and reproducibility metadata.

Every run - smoke test, tuning trial, or final training - gets its own
directory under ``artifacts/`` containing enough information to explain the
numbers months later:

    artifacts/experiments/<name>_<timestamp>/
        environment.json    versions, GPU, git commit, dirty-tree flag
        config.json         the exact hyperparameters used
        metrics.json        final metrics
        events.jsonl        append-only log of everything that happened

Deliberately dependency-free: no MLflow or W&B server to stand up, nothing to
authenticate against, and the output is plain JSON that survives the tooling
that wrote it. ``RunRecorder`` is the only thing other modules import.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.ml import config


def _git_metadata() -> dict:
    """Commit and dirty-tree state, so a result can be tied back to code."""

    def run(*args):
        try:
            return subprocess.run(
                args,
                cwd=config.PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")

    return {
        "commit": commit or "unknown",
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # A dirty tree means the commit hash does NOT fully describe the code
        # that produced these numbers. Worth recording rather than hiding.
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def environment_snapshot() -> dict:
    """Everything about the machine that could change a number."""

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": _git_metadata(),
        "env_overrides": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("CODESENTINEL_")
        },
    }

    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda_build"] = torch.version.cuda
        snapshot["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            snapshot["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_mib": round(properties.total_memory / 1024**2),
                "capability": f"{properties.major}.{properties.minor}",
                "device_count": torch.cuda.device_count(),
            }
        else:
            snapshot["gpu"] = None
    except ImportError:
        snapshot["torch"] = None

    for module_name in ("transformers", "sklearn", "numpy", "optuna"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            snapshot[module_name] = None

    return snapshot


class RunRecorder:
    """
    One experiment run's directory and its append-only event log.

    Used as a context manager so that a crash still records the failure and the
    wall-clock duration instead of leaving a half-written directory with no
    explanation.
    """

    def __init__(self, name: str, root: Path | None = None, config_dict: dict | None = None):
        root = Path(root) if root else config.EXPERIMENTS_DIR

        # Local time, deliberately: this string is a directory name a human
        # reads. The UTC instant is recorded inside environment.json.
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.directory = root / f"{name}_{stamp}"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.events_path = self.directory / "events.jsonl"
        self.started = time.time()

        self.write_json("environment.json", environment_snapshot())

        if config_dict is not None:
            self.write_json("config.json", config_dict)

        self.log("run_started", name=name)

    # -- writing --------------------------------------------------------

    def write_json(self, filename: str, payload) -> Path:
        path = self.directory / filename
        path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return path

    def log(self, event: str, **fields) -> None:
        """Append one event. Never raises: logging must not kill a run."""

        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "event": event,
        }
        record.update(fields)

        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def record_metrics(self, metrics, split: str = "valid") -> None:
        payload = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)

        self.write_json(f"metrics_{split}.json", payload)
        self.log("metrics", split=split, **_scalars(payload))

    # -- context manager -------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.log("run_finished", status="ok")
        else:
            self.log("run_failed", status="error", error=f"{exc_type.__name__}: {exc}")

        # Never suppress the exception; the caller still needs to see it.
        return False


def _scalars(payload: dict) -> dict:
    """Flatten to log-safe scalars, dropping nested confusion structures."""

    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
