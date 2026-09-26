"""
Every script in `scripts/` must be runnable the way its own docstring says.

This exists because two of them were not. `scripts/error_analysis.py` and
`scripts/sensitivity_valid_test_overlap.py` both documented
`python scripts/<name>.py ...` and both died with ModuleNotFoundError, because
running a script *by path* puts the script's own directory on sys.path rather
than the repository root. The failure was invisible to the rest of the suite:
pytest imports these modules with the root already on the path, so a unit test
that merely imported them would pass while the documented command was broken.

The only way to catch that class of bug is to launch a real subprocess with a
clean environment, which is what these tests do. They use `--help`, so nothing
is computed, no data is read and no GPU is touched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py"))


def _clean_env():
    """
    Strip PYTHONPATH so the test cannot be rescued by an inherited path, which
    is precisely how the original bug hid. Everything else is inherited so the
    virtualenv is still found.
    """

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_scripts_directory_is_not_empty():
    """Guards against the parametrisation below silently testing nothing."""

    assert SCRIPTS, "no scripts found - the glob or the directory moved"


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_as_documented(script):
    """
    `python scripts/<name>.py --help` must exit 0 from the repository root.

    Run from ROOT rather than the test's directory because that is the path the
    docstrings document.
    """

    result = subprocess.run(
        [sys.executable, str(Path("scripts") / script), "--help"],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"scripts/{script} --help exited {result.returncode} with a clean "
        f"PYTHONPATH, so its documented invocation is broken.\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "usage" in result.stdout.lower(), (
        f"scripts/{script} --help produced no usage text: {result.stdout[:400]!r}"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_from_an_unrelated_directory(script):
    """
    The bootstrap must key off __file__, not the current directory, so the
    scripts work when invoked by absolute path from elsewhere (how a scheduler
    or a CI job would call them).
    """

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT.parent,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"scripts/{script} failed when run by absolute path from "
        f"{ROOT.parent}, so its path bootstrap depends on the cwd.\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
