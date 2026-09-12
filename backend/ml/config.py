"""
Central configuration for the CodeSentinel ML module.

Every value can be overridden with an environment variable so that the
same code runs unchanged on a laptop CPU, a single GPU box, or CI.
"""

import os
from pathlib import Path


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


DATA_DIR = _path_env("CODESENTINEL_DATA_DIR", PROJECT_ROOT / "data")

RAW_DIR = _path_env("CODESENTINEL_RAW_DIR", DATA_DIR / "raw")
PROCESSED_DIR = _path_env("CODESENTINEL_PROCESSED_DIR", DATA_DIR / "processed")
CHUNKED_DIR = _path_env("CODESENTINEL_CHUNKED_DIR", DATA_DIR / "chunked")
CHECKPOINT_DIR = _path_env("CODESENTINEL_CHECKPOINT_DIR", DATA_DIR / "checkpoints")

SPLITS = ("train", "valid", "test")

# Raw files are read-only. Nothing in this module ever writes into RAW_DIR.
RAW_FILES = {split: RAW_DIR / f"primevul_{split}.jsonl" for split in SPLITS}

PROCESSED_FILES = {
    split: PROCESSED_DIR / f"primevul_{split}.jsonl" for split in SPLITS
}

# Optional undersampled training file (see balance_dataset.py).
BALANCED_TRAIN_FILE = PROCESSED_DIR / "primevul_train_balanced.jsonl"

CHUNKED_FILES = {
    split: CHUNKED_DIR / f"primevul_{split}_chunked.jsonl" for split in SPLITS
}

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_codebert.pt"


# ----------------------------------------------------------------------
# Dataset schema
#
# PrimeVul field names are auto-detected from these candidate lists rather
# than hard-coded, so a renamed export does not silently produce garbage.
# ----------------------------------------------------------------------

CODE_FIELD_CANDIDATES = ("func", "code", "function", "source", "func_before")
LABEL_FIELD_CANDIDATES = ("target", "label", "vul", "is_vulnerable", "vulnerable")

# Metadata carried through preprocessing when present.
METADATA_FIELDS = (
    "project",
    "commit_id",
    "cwe",
    "cve",
    "idx",
    "big_vul_idx",
    "hash",
)


# ----------------------------------------------------------------------
# Tokenization / chunking
# ----------------------------------------------------------------------

MODEL_NAME = os.environ.get("CODESENTINEL_MODEL_NAME", "microsoft/codebert-base")

# CodeBERT's positional embedding table allows 512 positions.
MAX_LENGTH = int(os.environ.get("CODESENTINEL_MAX_LENGTH", 512))

# Tokens of overlap between consecutive chunks, so a vulnerability that
# straddles a chunk boundary is still visible inside a single chunk.
CHUNK_OVERLAP = int(os.environ.get("CODESENTINEL_CHUNK_OVERLAP", 128))

# Hard cap on chunks kept per function. Bounds worst-case memory for the
# handful of very long functions in PrimeVul.
MAX_CHUNKS_PER_FUNCTION = int(os.environ.get("CODESENTINEL_MAX_CHUNKS", 8))


# ----------------------------------------------------------------------
# Class imbalance
#
# Primary strategy: BCEWithLogitsLoss(pos_weight=n_negative / n_positive),
# computed from the training split that is actually used. See the ML README
# for why this was chosen over resampling.
# ----------------------------------------------------------------------

IMBALANCE_STRATEGY = os.environ.get("CODESENTINEL_IMBALANCE", "pos_weight")

# Only consulted by balance_dataset.py (an optional compute-budget tool).
BENIGN_TO_VULNERABLE_RATIO = float(
    os.environ.get("CODESENTINEL_BENIGN_RATIO", 5.0)
)

# Ceiling so a pathological split cannot produce an enormous pos_weight.
MAX_POS_WEIGHT = float(os.environ.get("CODESENTINEL_MAX_POS_WEIGHT", 50.0))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

# "cls"  -> use the <s> token representation of each chunk
# "mean" -> mask-aware mean over the chunk's tokens
CHUNK_POOLING = os.environ.get("CODESENTINEL_CHUNK_POOLING", "cls")

# How chunk embeddings of one function are combined. "mean" is the stable default.
FUNCTION_POOLING = os.environ.get("CODESENTINEL_FUNCTION_POOLING", "mean")

CLASSIFIER_DROPOUT = float(os.environ.get("CODESENTINEL_DROPOUT", 0.1))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

SEED = int(os.environ.get("CODESENTINEL_SEED", 42))

EPOCHS = int(os.environ.get("CODESENTINEL_EPOCHS", 5))

# Functions per optimizer step. Each function expands to 1..MAX_CHUNKS chunks,
# so the real encoder load is BATCH_SIZE * average_chunks.
BATCH_SIZE = int(os.environ.get("CODESENTINEL_BATCH_SIZE", 8))
EVAL_BATCH_SIZE = int(os.environ.get("CODESENTINEL_EVAL_BATCH_SIZE", 16))

# Chunks pushed through CodeBERT at once. Caps peak activation memory
# independently of BATCH_SIZE.
CHUNK_MICRO_BATCH = int(os.environ.get("CODESENTINEL_CHUNK_MICRO_BATCH", 16))

LEARNING_RATE = float(os.environ.get("CODESENTINEL_LR", 2e-5))
WEIGHT_DECAY = float(os.environ.get("CODESENTINEL_WEIGHT_DECAY", 0.01))
WARMUP_RATIO = float(os.environ.get("CODESENTINEL_WARMUP_RATIO", 0.06))
MAX_GRAD_NORM = float(os.environ.get("CODESENTINEL_MAX_GRAD_NORM", 1.0))

GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("CODESENTINEL_GRAD_ACCUM", 1)
)

USE_AMP = os.environ.get("CODESENTINEL_AMP", "1") == "1"
USE_GRADIENT_CHECKPOINTING = os.environ.get("CODESENTINEL_GRAD_CKPT", "1") == "1"

NUM_WORKERS = int(os.environ.get("CODESENTINEL_NUM_WORKERS", 0))

# Checkpoint cadence, in optimizer steps.
CHECKPOINT_EVERY = int(os.environ.get("CODESENTINEL_CHECKPOINT_EVERY", 500))
LOG_EVERY = int(os.environ.get("CODESENTINEL_LOG_EVERY", 50))
GPU_LOG_EVERY = int(os.environ.get("CODESENTINEL_GPU_LOG_EVERY", 200))

EARLY_STOPPING_PATIENCE = int(os.environ.get("CODESENTINEL_PATIENCE", 2))

# Per-sample loss above this prints a diagnostic line. Purely informational:
# a high loss never stops training.
HIGH_LOSS_THRESHOLD = float(os.environ.get("CODESENTINEL_HIGH_LOSS", 2.0))
HIGH_LOSS_MAX_REPORTS = int(os.environ.get("CODESENTINEL_HIGH_LOSS_MAX", 5))


# ----------------------------------------------------------------------
# Thresholding
# ----------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.5

# Grid searched on validation only. The test split is never used for tuning.
THRESHOLD_SEARCH_MIN = 0.05
THRESHOLD_SEARCH_MAX = 0.95
THRESHOLD_SEARCH_STEPS = 91


def describe() -> str:
    """Human-readable dump of the active configuration."""

    lines = [
        "CodeSentinel ML configuration",
        f"  model                 : {MODEL_NAME}",
        f"  max_length            : {MAX_LENGTH}",
        f"  chunk_overlap         : {CHUNK_OVERLAP}",
        f"  max_chunks/function   : {MAX_CHUNKS_PER_FUNCTION}",
        f"  imbalance strategy    : {IMBALANCE_STRATEGY}",
        f"  chunk pooling         : {CHUNK_POOLING}",
        f"  function pooling      : {FUNCTION_POOLING}",
        f"  seed                  : {SEED}",
        f"  epochs                : {EPOCHS}",
        f"  batch_size            : {BATCH_SIZE}",
        f"  chunk_micro_batch     : {CHUNK_MICRO_BATCH}",
        f"  learning_rate         : {LEARNING_RATE}",
        f"  amp                   : {USE_AMP}",
        f"  gradient_checkpointing: {USE_GRADIENT_CHECKPOINTING}",
        f"  data dir              : {DATA_DIR}",
    ]
    return "\n".join(lines)
