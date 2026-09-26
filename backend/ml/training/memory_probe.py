"""
Stage I - GPU MEMORY PROBE.

    python -m backend.ml.training.memory_probe
    python -m backend.ml.training.memory_probe --max-chunks 8 --headroom 0.15

Measures peak GPU memory for a real forward+backward+optimizer step across a
grid of batch sizes and chunk micro-batch sizes, and reports the largest
configuration that actually fits. Nothing here is estimated from parameter
counts: every number printed was observed.

Why measure rather than calculate
---------------------------------
The usual back-of-envelope (weights + gradients + Adam moments) predicts steady
state but not the peak, and the peak is what triggers OOM. Activation memory
depends on how many chunks the batch happened to expand into, which varies per
batch; gradient checkpointing trades activation memory for recomputation in a
way that is hard to predict; the caching allocator fragments; and AMP keeps an
fp32 master copy alongside fp16 activations. On a 4 GB laptop GPU the gap
between "should fit" and "fits" is exactly where a run dies six hours in.

The probe deliberately uses the WORST case - every function expanding to
``--max-chunks`` chunks of full length - because that is the batch that will
OOM, not the average one.

Escalation order when a TRAINING step does not fit
--------------------------------------------------
The brief lists chunk micro-batch first. Measurement on this hardware says
otherwise, and the measurement wins:

  micro_batch      1     2     4     8    16    32
  training peak  1260  1213  1167  1225  1290  1619   MiB   (~1.4x range)
  inference peak   81    79   108   168   288   528   MiB   (6.7x range)

During training autograd retains what it needs for backward across every
micro-batch, so peak memory tracks the *number of chunks in the batch*, not how
they were fed through the encoder. Micro-batching is a genuine lever for
evaluation, where ``no_grad`` frees each micro-batch immediately, and only a
mild one for training.

So, for training:
  1. reduce batch size              (directly reduces the chunk count)
  2. raise gradient accumulation    (keeps the effective batch unchanged)
  3. reduce max chunks per function (directly reduces the chunk count)
  4. set chunk micro-batch to ~4    (below that costs throughput for nothing)
  5. reduce max sequence length     (last resort - it changes the data)
"""

import argparse
import gc

import torch
from torch import nn

from backend.ml import config
from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier


def _reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def probe(
    model,
    optimizer,
    criterion,
    device,
    batch_size: int,
    chunks_per_function: int,
    sequence_length: int,
    use_amp: bool,
) -> dict:
    """One real training step at the worst-case shape. Returns peak memory."""

    _reset()

    total_chunks = batch_size * chunks_per_function

    # Real token ids, real attention (all-ones = no padding shortcut), and a
    # function_index that spreads chunks across the batch exactly as collate would.
    input_ids = torch.randint(
        5, 50000, (total_chunks, sequence_length), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)
    function_index = torch.arange(batch_size, device=device).repeat_interleave(
        chunks_per_function
    )
    targets = torch.randint(
        0, 2, (batch_size,), device=device, dtype=torch.float32
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    try:
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                function_index=function_index,
                batch_size=batch_size,
            )
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2

        return {"ok": True, "peak_mib": peak, "reserved_mib": reserved}

    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "peak_mib": float("nan"), "reserved_mib": float("nan")}
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return {"ok": False, "peak_mib": float("nan"), "reserved_mib": float("nan")}
    finally:
        del input_ids, attention_mask, function_index, targets
        _reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--max-chunks", type=int, default=config.MAX_CHUNKS_PER_FUNCTION)
    parser.add_argument("--seq-len", type=int, default=config.MAX_LENGTH)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--micro-batches", nargs="+", type=int, default=[2, 4, 8, 16]
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.15,
        help="Fraction of VRAM to leave free for fragmentation and eval peaks",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("CODESENTINEL - STAGE I: GPU MEMORY PROBE")
    print("=" * 78)

    if not torch.cuda.is_available():
        print("\nBLOCKED: CUDA is not available; there is nothing to probe.")
        raise SystemExit(1)

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    total_mib = properties.total_memory / 1024**2
    budget = total_mib * (1 - args.headroom)

    use_amp = not args.no_amp
    gradient_checkpointing = not args.no_grad_checkpointing

    print(f"  device            : {torch.cuda.get_device_name(0)}")
    print(f"  total VRAM        : {total_mib:.0f} MiB")
    print(f"  usable budget     : {budget:.0f} MiB ({1 - args.headroom:.0%})")
    print(f"  worst-case shape  : {args.max_chunks} chunks x {args.seq_len} tokens per function")
    print(f"  mixed precision   : {use_amp}")
    print(f"  grad checkpointing: {gradient_checkpointing}")

    print("\nBuilding model...")
    model = HierarchicalCodeBERTClassifier(
        model_name=args.model,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # One step first, so the optimizer's Adam moments exist. Without them the
    # first measured configuration looks ~2x cheaper than steady state and the
    # recommendation comes out too optimistic.
    print("Warming up (allocating optimizer state)...")
    probe(model, optimizer, criterion, device, 1, 1, args.seq_len, use_amp)

    baseline = torch.cuda.memory_allocated() / 1024**2
    print(f"  steady-state weights + optimizer: {baseline:.0f} MiB\n")

    print(f"  {'batch':>6}{'micro':>7}{'chunks':>8}{'peak MiB':>11}{'reserved':>10}  verdict")
    print("  " + "-" * 62)

    results = []

    for batch_size in args.batch_sizes:
        for micro_batch in args.micro_batches:
            model.chunk_micro_batch = micro_batch

            result = probe(
                model,
                optimizer,
                criterion,
                device,
                batch_size,
                args.max_chunks,
                args.seq_len,
                use_amp,
            )

            total_chunks = batch_size * args.max_chunks
            fits = result["ok"] and result["peak_mib"] <= budget

            if not result["ok"]:
                verdict = "OOM"
                peak_text = "-"
                reserved_text = "-"
            else:
                verdict = "fits" if fits else "over budget"
                peak_text = f"{result['peak_mib']:.0f}"
                reserved_text = f"{result['reserved_mib']:.0f}"

            print(
                f"  {batch_size:>6}{micro_batch:>7}{total_chunks:>8}"
                f"{peak_text:>11}{reserved_text:>10}  {verdict}"
            )

            results.append(
                {
                    "batch_size": batch_size,
                    "chunk_micro_batch": micro_batch,
                    "total_chunks": total_chunks,
                    "fits": fits,
                    **result,
                }
            )

    viable = [r for r in results if r["fits"]]

    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)

    if not viable:
        print(
            "\n  Nothing fits at this worst-case shape. Escalate in order:\n"
            "    1. --micro-batches 1\n"
            f"    2. reduce CODESENTINEL_MAX_CHUNKS below {args.max_chunks}\n"
            "    3. reduce CODESENTINEL_MAX_LENGTH (changes the data - document it)"
        )
        raise SystemExit(1)

    # Prefer the largest real batch; break ties by the lowest peak memory, which
    # leaves the most room for the eval pass and for fragmentation over a long run.
    best = max(viable, key=lambda r: (r["batch_size"], -r["peak_mib"]))

    target_effective = 8
    grad_accum = max(1, round(target_effective / best["batch_size"]))

    print(
        f"\n  Largest configuration that fits the worst-case batch:\n\n"
        f"    --batch-size {best['batch_size']} "
        f"--chunk-micro-batch {best['chunk_micro_batch']} "
        f"--grad-accum {grad_accum}\n\n"
        f"  measured peak   : {best['peak_mib']:.0f} MiB of {total_mib:.0f} MiB\n"
        f"  effective batch : {best['batch_size'] * grad_accum} functions per optimizer step"
    )

    print(
        "\n  This is the worst case (every function at the chunk cap). Typical\n"
        "  batches use less, so the headroom is real rather than nominal."
    )


if __name__ == "__main__":
    main()
