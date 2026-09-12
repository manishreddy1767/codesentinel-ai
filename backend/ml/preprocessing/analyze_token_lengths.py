"""
CodeBERT token-length profile of the processed splits.

    python -m backend.ml.preprocessing.analyze_token_lengths

Reports how many functions exceed the 512-token limit and how many chunks the
current chunking settings will produce. Run this before chunking to sanity
check CODESENTINEL_MAX_CHUNKS.
"""

import argparse
import math
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import read_jsonl


def percentile(sorted_values, fraction: float):
    if not sorted_values:
        return 0

    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def expected_chunks(token_count: int) -> int:
    """Mirror the window arithmetic in chunk_dataset.build_chunks."""

    body = config.MAX_LENGTH - 2
    overlap = min(config.CHUNK_OVERLAP, body - 1)
    step = body - overlap

    if token_count <= body:
        return 1

    return 1 + math.ceil((token_count - body) / step)


def analyze_file(name: str, path: Path, tokenizer) -> None:
    print("\n" + "=" * 60)
    print(f"{name.upper()} SPLIT")
    print("=" * 60)

    if not path.exists():
        print(f"MISSING: {path}")
        return

    lengths = []
    over_limit = 0
    over_max_chunks = 0
    total_chunks = 0

    for index, (_, record) in enumerate(read_jsonl(path), start=1):
        # add_special_tokens=False matches how chunk_dataset.py tokenizes.
        token_count = len(
            tokenizer(
                record.get("func", ""),
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )

        lengths.append(token_count)

        if token_count + 2 > config.MAX_LENGTH:
            over_limit += 1

        chunks = expected_chunks(token_count)
        total_chunks += min(chunks, config.MAX_CHUNKS_PER_FUNCTION)

        if chunks > config.MAX_CHUNKS_PER_FUNCTION:
            over_max_chunks += 1

        if index % 5000 == 0:
            print(f"  ...{index} samples")

    if not lengths:
        print("No records.")
        return

    total = len(lengths)
    lengths.sort()

    print(f"\nSamples          : {total}")
    print(f"Minimum tokens   : {lengths[0]}")
    print(f"Maximum tokens   : {lengths[-1]}")
    print(f"Average tokens   : {sum(lengths) / total:.2f}")

    print("\nToken length percentiles:")
    for fraction in (0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(fraction * 100):<3}: {percentile(lengths, fraction)}")

    print(f"\n{config.MAX_LENGTH}-token limit analysis:")
    print(f"  functions needing >1 chunk : {over_limit} ({over_limit / total * 100:.2f}%)")
    print(
        f"  functions exceeding {config.MAX_CHUNKS_PER_FUNCTION} chunks"
        f" : {over_max_chunks} ({over_max_chunks / total * 100:.2f}%)"
    )
    print(f"  total chunks after capping : {total_chunks}")
    print(f"  mean chunks per function   : {total_chunks / total:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--model", default=config.MODEL_NAME)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("=" * 60)
    print("CODEBERT TOKEN LENGTH ANALYSIS")
    print("=" * 60)
    print(f"\nLoading tokenizer: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for split in config.SPLITS:
        analyze_file(split, args.dir / f"primevul_{split}.jsonl", tokenizer)


if __name__ == "__main__":
    main()
