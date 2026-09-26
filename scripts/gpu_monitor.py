"""
GPU and training-reliability telemetry recorder.

    python scripts/gpu_monitor.py --out artifacts/reports/gpu_telemetry.jsonl \
        --checkpoint-dir data/checkpoints --interval 30

Samples one JSON line per interval so that, if training crashes again, there
is a record of what the GPU was doing in the minutes before it happened rather
than a guess afterwards.

Recorded per sample: temperature, utilisation, memory, power, SM clock,
decoded throttle reasons, the training process's uptime and CPU time, and the
checkpoint's global_step (so progress rate is derivable).

Read-only. It runs `nvidia-smi` and stats a file; it never touches the GPU
context and cannot disturb the training job.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# nvidia-smi clocks_throttle_reasons.active bitmask.
THROTTLE_BITS = {
    0x1: "GpuIdle",
    0x2: "AppClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrake",
}

FIELDS = (
    "temperature.gpu,utilization.gpu,memory.used,memory.total,"
    "power.draw,clocks.sm,clocks_throttle_reasons.active"
)


def decode_throttle(value: str):
    try:
        bits = int(value, 16)
    except (TypeError, ValueError):
        return []
    return [name for bit, name in THROTTLE_BITS.items() if bits & bit]


def sample_gpu() -> dict:
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return {"gpu_error": str(error)[:120]}

    if not output:
        return {"gpu_error": "no output"}

    parts = [p.strip() for p in output.split(",")]

    def num(index):
        try:
            return float(parts[index])
        except (IndexError, ValueError):
            return None

    return {
        "temp_c": num(0),
        "util_pct": num(1),
        "mem_used_mib": num(2),
        "mem_total_mib": num(3),
        "power_w": num(4),
        "sm_clock_mhz": num(5),
        "throttle": decode_throttle(parts[6] if len(parts) > 6 else ""),
    }


def sample_process() -> dict:
    """Uptime and CPU time of the largest python process (the trainer)."""

    script = (
        "Get-Process python* -EA SilentlyContinue | "
        "Sort-Object WorkingSet64 -Descending | Select-Object -First 1 "
        "Id,StartTime,CPU,WorkingSet64 | ConvertTo-Json -Compress"
    )

    try:
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout.strip()

        if not output:
            return {"process": None}

        data = json.loads(output)
        start = data.get("StartTime")
        uptime = None

        if isinstance(start, str) and start.startswith("/Date("):
            millis = int(start[6:].split(")")[0].split("+")[0])
            uptime = round(time.time() - millis / 1000.0, 1)

        return {
            "pid": data.get("Id"),
            "cpu_s": round(data.get("CPU") or 0, 1),
            "rss_mib": round((data.get("WorkingSet64") or 0) / 1024**2, 1),
            "uptime_s": uptime,
        }
    except Exception as error:  # noqa: BLE001
        return {"process_error": str(error)[:120]}


def sample_checkpoint(checkpoint_dir: Path) -> dict:
    """global_step without loading the (1.4 GB) tensors more than necessary."""

    path = checkpoint_dir / "latest_checkpoint.pt"

    if not path.exists():
        return {"global_step": None}

    try:
        stat = path.stat()
        return {
            "ckpt_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "ckpt_age_s": round(time.time() - stat.st_mtime, 1),
            "ckpt_mib": round(stat.st_size / 1024**2, 1),
        }
    except OSError as error:
        return {"ckpt_error": str(error)[:120]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"sampling every {args.interval}s -> {args.out}")

    for _ in range(args.max_samples):
        record = {"time": datetime.now(timezone.utc).isoformat()}
        record.update(sample_gpu())
        record.update(sample_process())
        record.update(sample_checkpoint(args.checkpoint_dir))

        try:
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            pass

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
