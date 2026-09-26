"""
Supervisor for long training runs on unstable GPU hardware.

    python scripts/train_supervised.py -- <all normal train.py arguments>

Why this exists
---------------
This project's GPU (RTX 3050 Laptop, 4 GB) has failed three times under
sustained load, with a *different* CUDA error each time:

    tuning trial 3      Exception in pinned allocator free()
    training attempt 1  CUBLAS_STATUS_EXECUTION_FAILED       (batch 2,500)
    training attempt 2  cudaErrorIllegalAddress              (batch 3,501)

Each time the same code had already run thousands of batches, and each time a
plain CUDA matmul worked again once the process died. That pattern is hardware
or driver instability, not a defect in the training loop, and it cannot be
fixed from Python.

What it does NOT do
-------------------
It does not hide the problem. Every crash is counted, timestamped and printed,
and the final summary reports how many restarts were needed so the run can be
described accurately.

Crucially, it **requires forward progress**: if training dies twice without
`global_step` advancing, the supervisor stops. A crash that always happens at
the same step is a deterministic bug, and restarting it forever would burn
hours while hiding a real defect. Only genuinely intermittent failures are
retried.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DONE_MARKER = "TRAINING COMPLETE"


def read_global_step(checkpoint_dir: Path) -> int:
    """Progress so far, or -1 when there is no readable checkpoint yet."""

    path = checkpoint_dir / "latest_checkpoint.pt"

    if not path.exists():
        return -1

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("global_step", -1))
    except Exception:  # noqa: BLE001 - a half-written checkpoint is "unknown"
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Seconds to let the GPU settle after a crash before retrying",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Everything after -- is passed to backend.ml.training.train",
    )
    args = parser.parse_args()

    train_args = [a for a in args.train_args if a != "--"]

    if not train_args:
        raise SystemExit("No training arguments given after --")

    crashes = []
    last_step = read_global_step(args.checkpoint_dir)

    print("=" * 74)
    print("SUPERVISED TRAINING")
    print("=" * 74)
    print(f"  checkpoint dir : {args.checkpoint_dir}")
    print(f"  starting step  : {last_step}")
    print(f"  max attempts   : {args.max_attempts}")
    print(f"  log            : {args.log}")

    for attempt in range(1, args.max_attempts + 1):
        # Attempt 1 resumes too: the caller is expected to have a checkpoint,
        # and --resume is a no-op when none exists.
        command = [
            sys.executable,
            "-m",
            "backend.ml.training.train",
            *train_args,
            "--resume",
        ]

        print(f"\n--- attempt {attempt}/{args.max_attempts} "
              f"at {datetime.now().astimezone().strftime('%H:%M:%S')} (step {last_step}) ---",
              flush=True)

        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== supervisor attempt {attempt} =====\n")
            handle.flush()
            completed = subprocess.run(
                command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                check=False,
            )

        text = args.log.read_text(encoding="utf-8", errors="replace")
        step = read_global_step(args.checkpoint_dir)

        if DONE_MARKER in text:
            print(f"\nTRAINING COMPLETE after {attempt} attempt(s), "
                  f"{len(crashes)} crash(es).")
            break

        # Did not finish -> treat as a crash.
        crashes.append(
            {
                "attempt": attempt,
                "time": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "global_step": step,
            }
        )

        print(f"  crashed (rc={completed.returncode}) at step {step}", flush=True)

        # Forward-progress guard: two consecutive crashes with no advance means
        # the failure is deterministic, and retrying is just burning time.
        if step <= last_step and len(crashes) >= 2:
            previous = crashes[-2]["global_step"]
            if previous == step:
                print(
                    "\nABORTING: two consecutive crashes with no progress "
                    f"(stuck at global_step={step}). This looks deterministic "
                    "rather than intermittent - investigate before retrying."
                )
                break

        last_step = max(last_step, step)

        if attempt < args.max_attempts:
            print(f"  cooling down {args.cooldown}s before retry...", flush=True)
            time.sleep(args.cooldown)
    else:
        print(f"\nGAVE UP after {args.max_attempts} attempts.")

    summary = {
        "attempts": len(crashes) + (1 if DONE_MARKER in args.log.read_text(
            encoding="utf-8", errors="replace") else 0),
        "crashes": crashes,
        "completed": DONE_MARKER in args.log.read_text(encoding="utf-8", errors="replace"),
        "final_global_step": read_global_step(args.checkpoint_dir),
    }

    out = args.checkpoint_dir / "supervisor_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  crashes: {len(crashes)}   summary -> {out}")

    if not summary["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
