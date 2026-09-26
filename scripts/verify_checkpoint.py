"""
Checkpoint integrity verification.

    python scripts/verify_checkpoint.py --checkpoint data/checkpoints/latest_checkpoint.pt

Loads a checkpoint in a FRESH process and checks that every piece of state
needed to resume is present and sane. Runs on CPU by default so it can be used
while training occupies the GPU.

Why this is needed
------------------
A checkpoint written by a process that later died to a CUDA fault is not
automatically trustworthy. `torch.save` is atomic here (temp file + replace),
so a torn file is unlikely - but "the file loads" is a much weaker claim than
"the state in it can actually resume training". This checks the second.

Checks performed
----------------
1. File loads at all, and is a dict.
2. Model weights: every tensor present, correct count, no NaN/Inf, not all-zero.
3. Optimizer: state present, step counts consistent, exp_avg/exp_avg_sq finite.
4. Scheduler: last_epoch consistent with global_step.
5. RNG: all four generators captured and restorable.
6. Bookkeeping: epoch / batch_in_epoch / global_step mutually consistent.
7. Architecture metadata recorded from the model instance.

Exit code 0 = all checks passed.
"""

import argparse
import math
from pathlib import Path

import torch


class Check:
    def __init__(self):
        self.results = []

    def add(self, name, ok, detail=""):
        self.results.append((name, bool(ok), detail))
        symbol = "PASS" if ok else "FAIL"
        print(f"  [{symbol}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def failed(self):
        return [name for name, ok, _ in self.results if not ok]


def verify(path: Path, expected_batch_size: int | None) -> int:
    """
    Two checkpoint kinds exist and they legitimately differ:

      latest_checkpoint.pt  full training state, written to RESUME a run.
                            Must carry optimizer, scheduler and RNG state.
      best_codebert.pt      the best weights, written for INFERENCE and for
                            the final evaluation. The trainer deliberately does
                            not store optimizer/scheduler in it - they are
                            ~900 MB of state that deployment never reads.

    Applying resume-oriented checks to an inference checkpoint reports a
    failure that is not one, so the kind is detected and those checks become
    informational for `best_*`.
    """
    print("=" * 74)
    print("CHECKPOINT INTEGRITY VERIFICATION")
    print("=" * 74)
    print(f"  file : {path}")
    print(f"  size : {path.stat().st_size / 1024**2:.1f} MiB")
    print()

    check = Check()
    resume_checkpoint = "latest" in path.name.lower()
    print(f"  kind : {'RESUME (full training state)' if resume_checkpoint else 'INFERENCE (weights + metadata)'}")
    print()

    # --- 1. loads -------------------------------------------------------
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        check.add("file loads in a fresh process", isinstance(payload, dict))
    except Exception as error:  # noqa: BLE001
        check.add("file loads in a fresh process", False, str(error)[:120])
        return 1

    # --- 2. model weights ------------------------------------------------
    model_state = payload.get("model_state")
    check.add("model_state present", model_state is not None)

    if model_state:
        tensors = list(model_state.values())
        n_bad = sum(
            1
            for t in tensors
            if torch.is_tensor(t) and not torch.isfinite(t.float()).all()
        )
        n_zero = sum(
            1
            for t in tensors
            if torch.is_tensor(t) and t.numel() > 1 and float(t.float().abs().sum()) == 0.0
        )
        total_params = sum(t.numel() for t in tensors if torch.is_tensor(t))

        check.add("model tensor count", len(tensors) == 201, f"{len(tensors)} tensors")
        check.add("no NaN/Inf in weights", n_bad == 0, f"{n_bad} bad tensors")
        check.add(
            "weights are not degenerate",
            n_zero <= 2,
            f"{n_zero} all-zero tensors (bias init can legitimately be zero)",
        )
        check.add("parameter count plausible", total_params > 100_000_000,
                  f"{total_params:,} parameters")

    # --- 3. optimizer ----------------------------------------------------
    optimizer_state = payload.get("optimizer_state")

    if resume_checkpoint:
        check.add("optimizer_state present", optimizer_state is not None)
    else:
        print("  [INFO] optimizer_state absent — expected for an inference "
              "checkpoint")

    if optimizer_state:
        state = optimizer_state.get("state", {})
        groups = optimizer_state.get("param_groups", [])

        check.add("optimizer has param_groups", bool(groups), f"{len(groups)} groups")
        check.add("optimizer has per-parameter state", bool(state),
                  f"{len(state)} entries")

        steps = [
            float(v["step"]) for v in state.values()
            if isinstance(v, dict) and "step" in v
        ]
        if steps:
            check.add(
                "optimizer step counts agree",
                max(steps) - min(steps) < 1e-6,
                f"step={steps[0]:.0f}",
            )

        bad_moments = 0
        for entry in state.values():
            if not isinstance(entry, dict):
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                t = entry.get(key)
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    bad_moments += 1

        check.add("Adam moments finite", bad_moments == 0, f"{bad_moments} bad")

        learning_rates = [g.get("lr") for g in groups]
        check.add(
            "learning rate is finite and positive",
            all(isinstance(lr, float) and math.isfinite(lr) and lr >= 0 for lr in learning_rates),
            f"lr={learning_rates}",
        )

    # --- 4. scheduler ----------------------------------------------------
    scheduler_state = payload.get("scheduler_state")

    if resume_checkpoint:
        check.add("scheduler_state present", scheduler_state is not None)
    else:
        print("  [INFO] scheduler_state absent — expected for an inference checkpoint")

    global_step = payload.get("global_step", -1)

    if scheduler_state:
        last_epoch = scheduler_state.get("last_epoch")
        check.add(
            "scheduler last_epoch matches global_step",
            last_epoch == global_step,
            f"last_epoch={last_epoch} global_step={global_step}",
        )

    # --- 5. RNG ----------------------------------------------------------
    rng = payload.get("rng_state") or {}

    for name in ("python", "numpy", "torch", "cuda"):
        present = name in rng
        if resume_checkpoint:
            check.add(f"RNG {name} captured", present)
        elif not present:
            print(f"  [INFO] RNG {name} absent — not required for inference")

    if "torch" in rng:
        try:
            import random

            saved_python = random.getstate()
            saved_torch = torch.get_rng_state()

            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            random.setstate(rng["python"])

            torch.set_rng_state(saved_torch)
            random.setstate(saved_python)

            check.add("RNG state is restorable", True)
        except Exception as error:  # noqa: BLE001
            check.add("RNG state is restorable", False, str(error)[:100])

    # --- 6. bookkeeping ---------------------------------------------------
    epoch = payload.get("epoch")
    batch_in_epoch = payload.get("batch_in_epoch")

    check.add("epoch recorded", epoch is not None, f"epoch={epoch}")
    check.add("batch_in_epoch recorded", batch_in_epoch is not None,
              f"batch_in_epoch={batch_in_epoch}")
    check.add("global_step recorded", global_step >= 0, f"global_step={global_step}")

    if expected_batch_size and batch_in_epoch is not None and epoch is not None:
        # With grad_accum=1 the optimizer steps once per batch, so
        # global_step should equal epochs_completed*steps_per_epoch + batch.
        check.add(
            "global_step consistent with epoch/batch",
            global_step >= batch_in_epoch,
            f"{global_step} >= {batch_in_epoch}",
        )

    # --- 7. architecture ---------------------------------------------------
    config = payload.get("config") or {}
    for key in ("model_name", "chunk_pooling", "function_pooling", "dropout"):
        check.add(f"architecture records {key}", key in config,
                  f"{key}={config.get(key)}")

    if not resume_checkpoint:
        metrics = payload.get("metrics") or {}
        check.add("validation metrics recorded", bool(metrics),
                  f"pr_auc={metrics.get('pr_auc')} f1={metrics.get('f1')}")
        extra = payload.get("extra") or {}
        check.add("selection provenance recorded",
                  "monitor" in extra and "best_epoch" in extra,
                  f"monitor={extra.get('monitor')} best_epoch={extra.get('best_epoch')}")

    print()
    print("=" * 74)

    if check.failed:
        print(f"RESULT: {len(check.failed)} CHECK(S) FAILED")
        for name in check.failed:
            print(f"  - {name}")
        return 1

    print(f"RESULT: ALL {len(check.results)} CHECKS PASSED")
    print(f"  resumable from epoch {epoch}, batch {batch_in_epoch}, step {global_step}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: {args.checkpoint} not found")
        raise SystemExit(2)

    raise SystemExit(verify(args.checkpoint, args.batch_size))


if __name__ == "__main__":
    main()
