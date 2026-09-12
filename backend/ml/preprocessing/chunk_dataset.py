"""
Stage 5 - TOKENIZATION and CHUNKING.

CodeBERT accepts at most 512 positions, and plenty of PrimeVul functions are
longer than that. This stage splits every function into one or more valid
CodeBERT inputs and stores them in data/chunked/.

    python -m backend.ml.preprocessing.chunk_dataset
    python -m backend.ml.preprocessing.chunk_dataset --use-balanced-train

Correctness rules this implementation enforces
----------------------------------------------
1. The function is tokenized with ``add_special_tokens=False`` and the
   special tokens are re-attached *per chunk*. The previous implementation
   sliced the sequence after adding them, so only the first chunk had <s>
   and only the last had </s>; every middle chunk was a bare token run that
   CodeBERT was never trained to see.
2. Each chunk body is at most ``max_length - 2`` tokens, leaving exactly room
   for <s> and </s>, so no chunk can exceed the positional embedding table.
3. Consecutive chunks overlap by ``CHUNK_OVERLAP`` tokens, so a pattern that
   straddles a boundary is still fully contained in one chunk.
4. Every function yields at least one chunk. An empty or whitespace-only
   function still produces the degenerate ``[<s>, </s>]`` chunk rather than
   an empty list, which would crash the collate function.
5. No chunk is ever empty.
6. Functions longer than ``MAX_CHUNKS_PER_FUNCTION`` chunks are truncated to
   that many chunks, and the record records that it was truncated.
"""

import argparse
from pathlib import Path

from backend.ml import config
from backend.ml.utils.io import read_jsonl, write_jsonl


def build_chunks(
    token_ids,
    bos_id: int,
    eos_id: int,
    max_length: int = config.MAX_LENGTH,
    overlap: int = config.CHUNK_OVERLAP,
    max_chunks: int = config.MAX_CHUNKS_PER_FUNCTION,
):
    """
    Split a bare token-id sequence into CodeBERT-ready chunks.

    ``token_ids`` must come from a tokenizer call with
    ``add_special_tokens=False``.

    Returns ``(chunks, truncated)`` where each chunk is a list of ids that
    starts with ``bos_id`` and ends with ``eos_id``.
    """

    body_size = max_length - 2

    if body_size < 1:
        raise ValueError(f"max_length must be at least 3, got {max_length}")

    # Guard against a nonsensical overlap that would make the window not advance.
    if overlap >= body_size:
        overlap = body_size - 1

    step = body_size - overlap

    # A function with no tokens still has to produce exactly one valid chunk,
    # otherwise it would silently vanish from the dataset and break batching.
    if not token_ids:
        return [[bos_id, eos_id]], False

    chunks = []
    start = 0
    truncated = False

    while start < len(token_ids):
        body = token_ids[start:start + body_size]

        if not body:
            break

        # Stop as soon as the cap is reached rather than building every chunk
        # of a 100k-token function and throwing all but eight away.
        if len(chunks) >= max_chunks:
            truncated = True
            break

        chunks.append([bos_id] + list(body) + [eos_id])

        # The window covered the tail, so stop instead of emitting a chunk
        # that repeats only overlap tokens.
        if start + body_size >= len(token_ids):
            break

        start += step

    return chunks, truncated


def chunk_function(code: str, tokenizer, **kwargs):
    """Tokenize one function and split it into chunks."""

    token_ids = tokenizer(
        code,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    # RoBERTa-family tokenizers define bos/eos; fall back to cls/sep otherwise.
    if bos_id is None:
        bos_id = tokenizer.cls_token_id
    if eos_id is None:
        eos_id = tokenizer.sep_token_id

    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer exposes no BOS/EOS (or CLS/SEP) token ids")

    return build_chunks(token_ids, bos_id, eos_id, **kwargs)


def process_split(split: str, input_path: Path, output_path: Path, tokenizer):
    print("\n" + "-" * 70)
    print(f"CHUNKING: {split}")
    print(f"  in : {input_path}")
    print(f"  out: {output_path}")
    print("-" * 70)

    if not input_path.exists():
        print("  SKIPPED: input file not found.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_functions = 0
    total_chunks = 0
    max_chunks_seen = 0
    truncated_functions = 0
    single_chunk = 0
    multi_chunk = 0
    label_counts = {0: 0, 1: 0}

    def records():
        nonlocal total_functions, total_chunks, max_chunks_seen
        nonlocal truncated_functions, single_chunk, multi_chunk

        for _, record in read_jsonl(input_path):
            code = record.get("func", "")
            target = int(record.get("target", 0))

            chunks, truncated = chunk_function(code, tokenizer)

            # Invariants the training pipeline relies on.
            assert chunks, "chunking produced zero chunks"
            assert all(chunks), "chunking produced an empty chunk"
            assert all(
                len(chunk) <= config.MAX_LENGTH for chunk in chunks
            ), "chunk exceeds max_length"

            total_functions += 1
            total_chunks += len(chunks)
            max_chunks_seen = max(max_chunks_seen, len(chunks))
            label_counts[target] = label_counts.get(target, 0) + 1

            if truncated:
                truncated_functions += 1

            if len(chunks) == 1:
                single_chunk += 1
            else:
                multi_chunk += 1

            out = {
                "target": target,
                "num_chunks": len(chunks),
                "chunks": chunks,
                "truncated": truncated,
            }

            for field in ("project", "cwe", "idx", "hash"):
                if field in record:
                    out[field] = record[field]

            yield out

            if total_functions % 5000 == 0:
                print(f"    ...{total_functions} functions")

    write_jsonl(output_path, records())

    if total_functions == 0:
        print("  WARNING: input file contained no records.")
        return None

    print(f"  functions            : {total_functions}")
    print(f"  total chunks         : {total_chunks}")
    print(f"  mean chunks/function : {total_chunks / total_functions:.3f}")
    print(f"  max chunks/function  : {max_chunks_seen}")
    print(f"  single-chunk funcs   : {single_chunk}")
    print(f"  multi-chunk funcs    : {multi_chunk}")
    print(
        f"  truncated at {config.MAX_CHUNKS_PER_FUNCTION} chunks"
        f"  : {truncated_functions}"
    )
    print(f"  vulnerable / benign  : {label_counts.get(1, 0)} / {label_counts.get(0, 0)}")

    return {"functions": total_functions, "chunks": total_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument(
        "--use-balanced-train",
        action="store_true",
        help="Chunk primevul_train_balanced.jsonl instead of primevul_train.jsonl",
    )
    parser.add_argument("--model", default=config.MODEL_NAME)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("=" * 70)
    print("CODESENTINEL - TOKENIZATION AND CHUNKING")
    print("=" * 70)
    print(f"  model        : {args.model}")
    print(f"  max_length   : {config.MAX_LENGTH}")
    print(f"  overlap      : {config.CHUNK_OVERLAP}")
    print(f"  max chunks   : {config.MAX_CHUNKS_PER_FUNCTION}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  bos={tokenizer.bos_token_id}  eos={tokenizer.eos_token_id}")

    sources = {
        split: args.in_dir / f"primevul_{split}.jsonl" for split in config.SPLITS
    }

    if args.use_balanced_train:
        balanced = args.in_dir / "primevul_train_balanced.jsonl"

        if not balanced.exists():
            print(f"\nERROR: {balanced} not found.")
            print("Run: python -m backend.ml.preprocessing.balance_dataset --undersample")
            raise SystemExit(1)

        sources["train"] = balanced
        print(f"\nUsing undersampled training file: {balanced}")

    for split, input_path in sources.items():
        process_split(
            split,
            input_path,
            args.out_dir / f"primevul_{split}_chunked.jsonl",
            tokenizer,
        )

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
