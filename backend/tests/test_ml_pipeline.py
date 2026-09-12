"""
Verification suite for the CodeSentinel ML module.

Covers the full pipeline on synthetic PrimeVul-shaped data: inspection,
validation, preprocessing, class distribution, chunking, dataset/collate,
model shapes, loss, backward pass, checkpointing, resume, metrics and
threshold selection.

Tests that need CodeBERT weights are marked ``slow`` and skipped unless the
model can be loaded (offline CI, no HF cache). Everything else runs with no
network access.

    .venv/Scripts/python -m pytest backend/tests/test_ml_pipeline.py -v
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from backend.ml import config
from backend.ml.models.collate import make_collate_fn
from backend.ml.models.dataset import ChunkedFunctionDataset
from backend.ml.preprocessing.chunk_dataset import build_chunks
from backend.ml.training import checkpoint as ckpt
from backend.ml.training.metrics import compute_metrics, find_best_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BOS, EOS, PAD = 0, 2, 1


# ======================================================================
# Synthetic data fixtures
# ======================================================================


def _make_function(index: int, vulnerable: bool, lines: int) -> str:
    body = "\n".join(
        f"    int v{index}_{n} = compute({n});" for n in range(lines)
    )

    if vulnerable:
        body += f"\n    strcpy(buffer, user_input_{index});"
    else:
        body += f"\n    strncpy(buffer, safe_input_{index}, sizeof(buffer) - 1);"

    return f"int handler_{index}(char *buffer) {{\n{body}\n    return 0;\n}}"


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A synthetic data/raw/ with PrimeVul's schema and its class imbalance."""

    rng = random.Random(1234)
    directory = tmp_path / "raw"
    directory.mkdir()

    counter = 0

    for split, size, positive_rate in (
        ("train", 120, 0.10),
        ("valid", 40, 0.10),
        ("test", 40, 0.10),
    ):
        records = []

        for _ in range(size):
            counter += 1
            vulnerable = rng.random() < positive_rate

            records.append(
                {
                    "project": f"proj{counter % 5}",
                    "commit_id": f"c{counter:06d}",
                    "target": 1 if vulnerable else 0,
                    "func": _make_function(
                        counter, vulnerable, rng.choice([2, 5, 60, 200])
                    ),
                    "cwe": ["CWE-120"] if vulnerable else [],
                    "idx": counter,
                    "hash": counter * 7919,
                }
            )

        # Guarantee both classes exist in every split.
        records[0]["target"] = 1
        records[0]["func"] = _make_function(90000 + size, True, 3)
        records[1]["target"] = 0
        records[1]["func"] = _make_function(91000 + size, False, 3)

        path = directory / f"primevul_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    # A malformed line and an empty-code record, both in train only.
    train_path = directory / "primevul_train.jsonl"
    with train_path.open("a", encoding="utf-8") as handle:
        handle.write("{ this is not valid json\n")
        handle.write(json.dumps({"target": 1, "func": "   ", "cwe": []}) + "\n")

    return directory


@pytest.fixture
def chunked_file(tmp_path: Path) -> Path:
    """Chunked records with 1, 2 and 3 chunks per function, both classes."""

    path = tmp_path / "chunked.jsonl"

    records = [
        {"target": 0, "num_chunks": 1, "chunks": [[BOS, 10, 11, EOS]]},
        {
            "target": 1,
            "num_chunks": 2,
            "chunks": [[BOS, 20, 21, 22, EOS], [BOS, 23, EOS]],
        },
        {
            "target": 1,
            "num_chunks": 3,
            "chunks": [[BOS, 30, EOS], [BOS, 31, 32, EOS], [BOS, 33, EOS]],
        },
        {"target": 0, "num_chunks": 1, "chunks": [[BOS, EOS]]},
    ]

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return path


def run_module(module: str, *args, env_extra=None):
    """Invoke a pipeline stage exactly the way a user would."""

    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})

    return subprocess.run(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )


# ======================================================================
# 1-5. Raw loading, inspection, validation, labels, class distribution
# ======================================================================


def test_inspect_dataset_runs_on_raw(raw_dir):
    result = run_module("backend.ml.preprocessing.inspect_dataset", "--dir", raw_dir)

    assert result.returncode == 0, result.stderr
    assert "Detected code field    : func" in result.stdout
    assert "Detected label field   : target" in result.stdout
    assert "Invalid JSON lines     : 1" in result.stdout
    assert "suggested pos_weight" in result.stdout


def test_validate_dataset_passes_and_reports_leakage(raw_dir):
    result = run_module("backend.ml.preprocessing.validate_dataset", "--dir", raw_dir)

    assert result.returncode == 0, result.stderr
    assert "PASSED" in result.stdout
    assert "CROSS-SPLIT LEAKAGE CHECK" in result.stdout
    assert "CLEAN" in result.stdout


def test_validate_dataset_detects_leakage(raw_dir, tmp_path):
    """A function copied from train into test must be flagged."""

    train_first = json.loads(
        (raw_dir / "primevul_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    test_path = raw_dir / "primevul_test.jsonl"
    with test_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(train_first) + "\n")

    result = run_module(
        "backend.ml.preprocessing.validate_dataset", "--dir", raw_dir, "--strict"
    )

    assert result.returncode == 1
    assert "LEAKAGE" in result.stdout


def test_validate_dataset_rejects_single_class(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()

    for split in config.SPLITS:
        path = directory / f"primevul_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(5):
                handle.write(
                    json.dumps({"func": f"int f{index}(){{}}", "target": 0}) + "\n"
                )

    result = run_module("backend.ml.preprocessing.validate_dataset", "--dir", directory)

    assert result.returncode == 1
    assert "contains no vulnerable samples" in result.stdout


# ======================================================================
# 6-7. Preprocessing, balancing, and untouched valid/test
# ======================================================================


def test_preprocess_and_balance_leave_valid_test_untouched(raw_dir, tmp_path):
    processed = tmp_path / "processed"

    result = run_module(
        "backend.ml.preprocessing.preprocess_primevul",
        "--raw-dir", raw_dir,
        "--out-dir", processed,
    )
    assert result.returncode == 0, result.stderr

    # Raw files must be byte-identical after the run.
    for split in config.SPLITS:
        assert (processed / f"primevul_{split}.jsonl").exists()

    # Malformed JSON and empty code dropped from train.
    assert "dropped empty code  : 1" in result.stdout

    valid_before = (processed / "primevul_valid.jsonl").read_bytes()
    test_before = (processed / "primevul_test.jsonl").read_bytes()

    balance = run_module(
        "backend.ml.preprocessing.balance_dataset",
        "--dir", processed,
        "--undersample",
        "--ratio", 2,
    )
    assert balance.returncode == 0, balance.stderr

    # balance_dataset must never rewrite validation or test.
    assert (processed / "primevul_valid.jsonl").read_bytes() == valid_before
    assert (processed / "primevul_test.jsonl").read_bytes() == test_before

    balanced = processed / "primevul_train_balanced.jsonl"
    assert balanced.exists()

    targets = [
        json.loads(line)["target"]
        for line in balanced.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    positives = sum(targets)
    negatives = len(targets) - positives

    assert positives > 0
    # ratio 2 means at most 2 benign per vulnerable.
    assert negatives <= positives * 2


def test_preprocess_never_modifies_raw(raw_dir, tmp_path):
    before = {
        path.name: path.read_bytes() for path in sorted(raw_dir.glob("*.jsonl"))
    }

    run_module(
        "backend.ml.preprocessing.preprocess_primevul",
        "--raw-dir", raw_dir,
        "--out-dir", tmp_path / "processed",
    )

    after = {
        path.name: path.read_bytes() for path in sorted(raw_dir.glob("*.jsonl"))
    }

    assert before == after, "raw dataset files were modified"


def test_preprocess_deduplicates_train_only(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()

    duplicate = "int f(){ strcpy(a, b); }"

    for split in config.SPLITS:
        path = directory / f"primevul_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for _ in range(3):
                handle.write(json.dumps({"func": duplicate, "target": 1}) + "\n")
            handle.write(json.dumps({"func": "int g(){}", "target": 0}) + "\n")

    processed = tmp_path / "processed"
    run_module(
        "backend.ml.preprocessing.preprocess_primevul",
        "--raw-dir", directory,
        "--out-dir", processed,
    )

    def count(split):
        text = (processed / f"primevul_{split}.jsonl").read_text(encoding="utf-8")
        return len([line for line in text.splitlines() if line.strip()])

    assert count("train") == 2, "train should be de-duplicated"
    assert count("valid") == 4, "valid must keep its duplicates"
    assert count("test") == 4, "test must keep its duplicates"


# ======================================================================
# 8-11. Chunking
# ======================================================================


def test_chunking_single_chunk():
    chunks, truncated = build_chunks(list(range(50)), BOS, EOS, max_length=512)

    assert len(chunks) == 1
    assert not truncated
    assert chunks[0][0] == BOS
    assert chunks[0][-1] == EOS
    assert len(chunks[0]) == 52


def test_chunking_multi_chunk_has_special_tokens_on_every_chunk():
    """The bug in the old implementation: middle chunks lacked BOS/EOS."""

    chunks, _ = build_chunks(
        list(range(3000)), BOS, EOS, max_length=512, overlap=128
    )

    assert len(chunks) > 2

    for chunk in chunks:
        assert chunk[0] == BOS, "chunk missing BOS"
        assert chunk[-1] == EOS, "chunk missing EOS"
        assert len(chunk) <= 512, "chunk exceeds max_length"
        assert len(chunk) >= 3


def test_chunking_respects_overlap():
    body = 512 - 2
    overlap = 128
    step = body - overlap

    tokens = list(range(2000))
    chunks, _ = build_chunks(tokens, BOS, EOS, max_length=512, overlap=overlap)

    # Second chunk body must start `step` tokens into the sequence.
    assert chunks[1][1] == tokens[step]


def test_chunking_empty_input_yields_one_chunk():
    chunks, truncated = build_chunks([], BOS, EOS)

    assert len(chunks) == 1
    assert chunks[0] == [BOS, EOS]
    assert not truncated


def test_chunking_never_yields_empty_chunks():
    for size in (0, 1, 509, 510, 511, 512, 1019, 1020, 1021, 5000):
        chunks, _ = build_chunks(list(range(size)), BOS, EOS, max_length=512)

        assert chunks, f"zero chunks for size {size}"
        assert all(chunks), f"empty chunk for size {size}"
        assert all(len(chunk) <= 512 for chunk in chunks)


def test_chunking_truncates_at_max_chunks():
    chunks, truncated = build_chunks(
        list(range(100_000)), BOS, EOS, max_length=512, overlap=128, max_chunks=8
    )

    assert len(chunks) == 8
    assert truncated


def test_chunking_handles_degenerate_overlap():
    """An overlap >= body size would otherwise make the window never advance."""

    chunks, _ = build_chunks(
        list(range(2000)), BOS, EOS, max_length=512, overlap=9999, max_chunks=100
    )

    assert len(chunks) > 1


# ======================================================================
# 12-13. Dataset and collate, batch size 1 and larger
# ======================================================================


def test_dataset_loads_and_reports_class_stats(chunked_file):
    dataset = ChunkedFunctionDataset(chunked_file)

    assert len(dataset) == 4

    negatives, positives = dataset.class_counts()
    assert (negatives, positives) == (2, 2)
    assert dataset.pos_weight() == pytest.approx(1.0)
    assert dataset.total_chunks() == 7


def test_dataset_lazy_and_in_memory_agree(chunked_file):
    """Lazy offset reads must return byte-identical items to eager loading."""

    lazy = ChunkedFunctionDataset(chunked_file, in_memory=False)
    eager = ChunkedFunctionDataset(chunked_file, in_memory=True)

    assert len(lazy) == len(eager)
    assert lazy.targets == eager.targets
    assert lazy.chunk_counts == eager.chunk_counts
    assert lazy.total_chunks() == eager.total_chunks()

    for index in range(len(lazy)):
        assert lazy[index] == eager[index]


def test_dataset_respects_max_chunks_cap(chunked_file):
    """A stale file with more chunks than the config allows must be clipped."""

    capped = ChunkedFunctionDataset(chunked_file, max_chunks=2)

    assert capped.chunk_counts == [1, 2, 2, 1]
    assert len(capped[2]["chunks"]) == 2
    assert capped.total_chunks() == 6


def test_dataset_rejects_zero_chunk_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"target": 1, "chunks": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="zero chunks"):
        ChunkedFunctionDataset(path)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_collate_shapes(chunked_file, batch_size):
    from torch.utils.data import DataLoader

    dataset = ChunkedFunctionDataset(chunked_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(PAD),
    )

    for batch in loader:
        functions = batch["targets"].size(0)
        chunks = batch["input_ids"].size(0)

        assert functions <= batch_size
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["function_index"].shape == (chunks,)
        assert batch["num_chunks"].shape == (functions,)
        assert batch["targets"].dtype == torch.float32

        # function_index must address exactly the functions in this batch.
        assert int(batch["function_index"].max()) == functions - 1
        assert int(batch["num_chunks"].sum()) == chunks

        # Padding positions must be masked out.
        padded = batch["input_ids"] == PAD
        assert torch.all(batch["attention_mask"][padded] == 0)


def test_collate_pads_to_longest_in_batch(chunked_file):
    dataset = ChunkedFunctionDataset(chunked_file)
    collate = make_collate_fn(PAD)

    batch = collate([dataset[0], dataset[1]])

    # Longest chunk across both functions is [BOS,20,21,22,EOS] = 5.
    assert batch["input_ids"].shape[1] == 5
    assert batch["input_ids"].shape[0] == 3


# ======================================================================
# 14-17. Model shapes, forward, loss, backward  (needs CodeBERT weights)
# ======================================================================


def tiny_model(chunk_micro_batch=4, chunk_pooling="cls", function_pooling="mean"):
    """
    The real classifier wrapped around a 2-layer randomly initialised
    RoBERTa encoder. Exercises the genuine forward/aggregation/head code
    path without downloading CodeBERT.
    """

    from transformers import RobertaConfig, RobertaModel

    from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier

    encoder_config = RobertaConfig(
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=530,
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )

    return HierarchicalCodeBERTClassifier(
        model_name="tiny-test",
        chunk_pooling=chunk_pooling,
        function_pooling=function_pooling,
        chunk_micro_batch=chunk_micro_batch,
        encoder=RobertaModel(encoder_config),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_model_output_shape_matches_targets(chunked_file, batch_size):
    from torch.utils.data import DataLoader

    model = tiny_model()
    dataset = ChunkedFunctionDataset(chunked_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(PAD),
    )

    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            function_index=batch["function_index"],
            batch_size=batch["targets"].size(0),
        )

        # The contract the whole training loop depends on.
        assert logits.shape == batch["targets"].shape
        assert logits.dim() == 1
        assert torch.isfinite(logits).all()


def test_model_aggregates_chunks_not_concatenates(chunked_file):
    """3 functions producing 7 chunks must still yield exactly 3 logits."""

    dataset = ChunkedFunctionDataset(chunked_file)
    collate = make_collate_fn(PAD)
    batch = collate([dataset[1], dataset[2], dataset[0]])

    assert batch["input_ids"].size(0) == 6  # 2 + 3 + 1 chunks

    logits = tiny_model()(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        function_index=batch["function_index"],
        batch_size=3,
    )

    assert logits.shape == (3,)


def test_mean_aggregation_is_actually_the_mean():
    """Two identical chunks must aggregate to the same embedding as one."""

    model = tiny_model()
    model.eval()

    chunk = torch.tensor([[BOS, 5, 6, 7, EOS]])
    mask = torch.ones_like(chunk)

    with torch.no_grad():
        single = model(chunk, mask, torch.tensor([0]), batch_size=1)
        doubled = model(
            chunk.repeat(2, 1),
            mask.repeat(2, 1),
            torch.tensor([0, 0]),
            batch_size=1,
        )

    assert torch.allclose(single, doubled, atol=1e-5)


@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_chunk_pooling_modes(chunked_file, pooling):
    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[i] for i in range(4)])

    logits = tiny_model(chunk_pooling=pooling)(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        function_index=batch["function_index"],
        batch_size=4,
    )

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_max_function_pooling(chunked_file):
    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[i] for i in range(4)])

    logits = tiny_model(function_pooling="max")(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        function_index=batch["function_index"],
        batch_size=4,
    )

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_chunk_micro_batching_is_equivalent(chunked_file):
    """Micro-batch size must change memory use, not the result."""

    torch.manual_seed(0)
    model = tiny_model(chunk_micro_batch=1)
    model.eval()

    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[i] for i in range(4)])

    with torch.no_grad():
        small = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["function_index"],
            batch_size=4,
        )

        model.chunk_micro_batch = 64
        large = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["function_index"],
            batch_size=4,
        )

    assert torch.allclose(small, large, atol=1e-5)


def test_loss_and_backward(chunked_file):
    import torch.nn as nn

    model = tiny_model()
    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[i] for i in range(4)])

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["function_index"],
        batch_size=4,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    loss = criterion(logits, batch["targets"])

    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()

    grads = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grads, "no gradients were produced"
    assert any(g.abs().sum() > 0 for g in grads), "all gradients are zero"
    assert all(torch.isfinite(g).all() for g in grads)


def test_model_returns_raw_logits_not_probabilities(chunked_file):
    """Logits must be unbounded; a sigmoid inside the model would break BCE."""

    model = tiny_model()

    # Push the head to a large output so a squashed value is detectable.
    with torch.no_grad():
        model.classifier.bias.fill_(12.0)

    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[0]])

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["function_index"],
        batch_size=1,
    )

    assert logits.item() > 1.0, "output looks like a probability, not a logit"


def test_gradient_checkpointing_still_produces_gradients(chunked_file):
    """Reentrant checkpointing silently kills gradients here; guard it."""

    import torch.nn as nn

    model = tiny_model()
    model.enable_gradient_checkpointing()
    model.train()

    dataset = ChunkedFunctionDataset(chunked_file)
    batch = make_collate_fn(PAD)([dataset[i] for i in range(4)])

    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["function_index"],
        batch_size=4,
    )

    nn.BCEWithLogitsLoss()(logits, batch["targets"]).backward()

    encoder_grads = [
        p.grad
        for name, p in model.named_parameters()
        if name.startswith("encoder.") and p.grad is not None
    ]

    assert encoder_grads, "gradient checkpointing produced no encoder gradients"
    assert any(g.abs().sum() > 0 for g in encoder_grads)


# ======================================================================
# 18-20. Device handling and memory
# ======================================================================


def test_cpu_fallback():
    from backend.ml.utils.gpu import resolve_device

    assert resolve_device(prefer_cuda=False).type == "cpu"


def test_memory_report_never_raises():
    from backend.ml.utils.gpu import memory_report, reset_peak_memory

    reset_peak_memory()
    report = memory_report()

    assert isinstance(report, str)
    assert "GPU memory" in report


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
def test_gpu_memory_does_not_grow_across_batches(chunked_file):
    import torch.nn as nn

    device = torch.device("cuda")
    model = tiny_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = ChunkedFunctionDataset(chunked_file)
    collate = make_collate_fn(PAD)

    readings = []

    for step in range(12):
        batch = collate([dataset[i % len(dataset)] for i in range(4)])
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["function_index"],
            batch_size=4,
        )
        loss = criterion(logits, batch["targets"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step >= 4:  # let allocator steady-state settle first
            readings.append(torch.cuda.memory_allocated(device))

    # Allow slack for allocator behaviour but catch a genuine per-step leak.
    assert max(readings) - min(readings) < 32 * 1024 * 1024, (
        f"allocated memory grew across steps: {readings}"
    )


# ======================================================================
# 21-23. Checkpoint save / load / resume
# ======================================================================


def test_checkpoint_roundtrip(tmp_path, chunked_file):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    path = tmp_path / "latest_checkpoint.pt"

    ckpt.save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        global_step=123,
        batch_in_epoch=45,
        best_f1=0.6789,
        best_threshold=0.37,
        epochs_without_improvement=1,
    )

    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists(), "temp file left behind"

    restored = tiny_model()

    # Confirm the weights genuinely differ before loading.
    original = model.classifier.weight.detach().clone()
    assert not torch.allclose(original, restored.classifier.weight)

    payload = ckpt.load_checkpoint(path, model=restored)

    assert torch.allclose(original, restored.classifier.weight)
    assert payload["epoch"] == 2
    assert payload["global_step"] == 123
    assert payload["batch_in_epoch"] == 45
    assert payload["best_f1"] == pytest.approx(0.6789)
    assert payload["best_threshold"] == pytest.approx(0.37)
    assert payload["epochs_without_improvement"] == 1
    assert payload["config"]["max_length"] == config.MAX_LENGTH


def test_load_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ckpt.load_checkpoint(tmp_path / "nope.pt")


def test_resumable_sampler_is_deterministic_per_epoch():
    data = list(range(50))

    a = ckpt.ResumableSampler(data, seed=7)
    b = ckpt.ResumableSampler(data, seed=7)

    a.set_epoch(3)
    b.set_epoch(3)

    assert list(a) == list(b), "same (seed, epoch) must give the same order"

    b.set_epoch(4)
    assert list(a) != list(b), "different epochs must shuffle differently"


def test_resumable_sampler_skip_resumes_exactly():
    """Resuming mid-epoch must continue the same permutation, not reshuffle."""

    data = list(range(50))
    batch_size = 4
    skip_batches = 3

    full = ckpt.ResumableSampler(data, seed=11)
    full.set_epoch(2)
    full_order = list(full)

    resumed = ckpt.ResumableSampler(data, seed=11)
    resumed.set_epoch(2)
    resumed.set_skip(skip_batches * batch_size)
    resumed_order = list(resumed)

    assert resumed_order == full_order[skip_batches * batch_size:]
    assert len(resumed) == len(data) - skip_batches * batch_size

    # No sample is replayed and none is lost.
    seen = full_order[: skip_batches * batch_size]
    assert set(seen).isdisjoint(resumed_order)
    assert sorted(seen + resumed_order) == data


def test_sampler_without_shuffle_is_sequential():
    sampler = ckpt.ResumableSampler(list(range(10)), shuffle=False)
    assert list(sampler) == list(range(10))


# ======================================================================
# Trainer integration: train -> validate -> checkpoint -> resume
# ======================================================================


@pytest.fixture
def trainable_chunked_file(tmp_path: Path) -> Path:
    """
    A learnable synthetic task: token 42 appears iff the function is vulnerable.

    Large enough to produce several batches, with a 4:1 imbalance so pos_weight
    is exercised, and a mix of 1-, 2- and 3-chunk functions.
    """

    rng = random.Random(99)
    path = tmp_path / "trainable.jsonl"

    with path.open("w", encoding="utf-8") as handle:
        for index in range(64):
            vulnerable = index % 5 == 0
            chunk_count = 1 + index % 3

            chunks = []
            for _ in range(chunk_count):
                body = [rng.randint(10, 39) for _ in range(6)]
                if vulnerable:
                    body[rng.randrange(len(body))] = 42
                chunks.append([BOS] + body + [EOS])

            handle.write(
                json.dumps(
                    {
                        "target": 1 if vulnerable else 0,
                        "num_chunks": len(chunks),
                        "chunks": chunks,
                    }
                )
                + "\n"
            )

    return path


def _make_trainer(dataset_path, tmp_path, **overrides):
    from backend.ml.training.trainer import Trainer

    dataset = ChunkedFunctionDataset(dataset_path)

    settings = dict(
        collate_fn=make_collate_fn(PAD),
        device=torch.device("cpu"),
        epochs=2,
        batch_size=8,
        eval_batch_size=8,
        learning_rate=5e-4,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=2,
        log_every=0,
        gpu_log_every=0,
        patience=5,
        use_amp=False,
        high_loss_threshold=0,
    )
    settings.update(overrides)

    return Trainer(
        model=tiny_model(),
        train_dataset=dataset,
        valid_dataset=dataset,
        **settings,
    ), dataset


def test_trainer_pos_weight_comes_from_training_split(
    trainable_chunked_file, tmp_path
):
    trainer, dataset = _make_trainer(trainable_chunked_file, tmp_path)

    negatives, positives = dataset.class_counts()
    assert trainer.pos_weight_value == pytest.approx(negatives / positives)
    assert trainer.pos_weight_value > 1.0


def test_trainer_full_loop_produces_checkpoints(trainable_chunked_file, tmp_path):
    trainer, _ = _make_trainer(trainable_chunked_file, tmp_path)

    result = trainer.fit()

    assert trainer.latest_path.exists()
    assert trainer.best_path.exists()

    assert len(trainer.history) == 2
    assert result["best_f1"] >= 0.0
    assert 0.0 < result["best_threshold"] < 1.0

    # The best checkpoint must carry the validation-selected threshold.
    payload = torch.load(trainer.best_path, map_location="cpu", weights_only=False)
    assert payload["best_threshold"] == pytest.approx(result["best_threshold"])
    assert payload["best_f1"] == pytest.approx(result["best_f1"])
    assert "metrics" in payload and payload["metrics"]["f1"] == pytest.approx(
        result["best_f1"]
    )


def test_trainer_learns_a_separable_signal(trainable_chunked_file, tmp_path):
    """Not an accuracy claim - just proof that gradients actually move the model."""

    trainer, dataset = _make_trainer(
        trainable_chunked_file, tmp_path, epochs=6, learning_rate=3e-3
    )

    before, targets = trainer.evaluate(dataset)
    baseline = compute_metrics(before, targets, 0.5)

    trainer.fit()

    after, _ = trainer.evaluate(dataset)
    trained = compute_metrics(after, targets, 0.5)

    assert trained.f1 > baseline.f1 or trainer.best_f1 > baseline.f1
    # The model must not collapse to a constant output.
    assert after.std() > 1e-4


def test_trainer_resume_restores_state(trainable_chunked_file, tmp_path):
    trainer, _ = _make_trainer(trainable_chunked_file, tmp_path, epochs=1)
    trainer.fit()

    weights = {
        name: p.detach().clone() for name, p in trainer.model.named_parameters()
    }
    expected_step = trainer.global_step
    expected_f1 = trainer.best_f1
    expected_threshold = trainer.best_threshold

    # A brand-new trainer with fresh weights and a fresh optimizer.
    resumed, _ = _make_trainer(trainable_chunked_file, tmp_path, epochs=1)

    assert not torch.allclose(
        resumed.model.classifier.weight, weights["classifier.weight"]
    )

    assert resumed.maybe_resume(resume=True) is True

    assert resumed.global_step == expected_step
    assert resumed.best_f1 == pytest.approx(expected_f1)
    assert resumed.best_threshold == pytest.approx(expected_threshold)
    assert resumed.start_epoch == 1
    assert resumed.start_batch == 0

    for name, parameter in resumed.model.named_parameters():
        assert torch.allclose(parameter, weights[name]), f"{name} not restored"

    # Optimizer moments must come back too, not just the weights.
    assert resumed.optimizer.state_dict()["state"], "optimizer state not restored"


def test_trainer_reports_no_resume_when_no_checkpoint(
    trainable_chunked_file, tmp_path
):
    trainer, _ = _make_trainer(trainable_chunked_file, tmp_path)

    # Must not silently claim a resume that did not happen.
    assert trainer.maybe_resume(resume=True) is False
    assert trainer.start_epoch == 0
    assert trainer.global_step == 0


def test_trainer_mid_epoch_checkpoint_records_batch(
    trainable_chunked_file, tmp_path
):
    trainer, _ = _make_trainer(
        trainable_chunked_file, tmp_path, epochs=1, batch_size=4, checkpoint_every=2
    )

    trainer.train_epoch(0, skip_batches=0)

    payload = torch.load(trainer.latest_path, map_location="cpu", weights_only=False)
    assert payload["batch_in_epoch"] > 0
    assert payload["epoch"] == 0


def test_trainer_aborts_on_non_finite_loss(trainable_chunked_file, tmp_path):
    """NaN loss is one of the few conditions that should stop training."""

    trainer, _ = _make_trainer(trainable_chunked_file, tmp_path, epochs=1)

    with torch.no_grad():
        trainer.model.classifier.weight.fill_(float("nan"))

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        trainer.train_epoch(0)


def test_trainer_evaluate_is_side_effect_free(trainable_chunked_file, tmp_path):
    trainer, dataset = _make_trainer(trainable_chunked_file, tmp_path)

    before = {
        name: p.detach().clone() for name, p in trainer.model.named_parameters()
    }

    probabilities, targets = trainer.evaluate(dataset)

    assert probabilities.shape == targets.shape == (len(dataset),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} changed during eval"
        assert parameter.grad is None or parameter.grad.abs().sum() == 0


def test_evaluator_predict_matches_trainer_evaluate(
    trainable_chunked_file, tmp_path
):
    from backend.ml.evaluation.evaluator import predict

    trainer, dataset = _make_trainer(trainable_chunked_file, tmp_path)

    from_trainer, targets_a = trainer.evaluate(dataset)
    from_evaluator, targets_b = predict(
        trainer.model,
        dataset,
        make_collate_fn(PAD),
        torch.device("cpu"),
        batch_size=8,
        progress_every=0,
    )

    assert (targets_a == targets_b).all()
    assert from_trainer == pytest.approx(from_evaluator, abs=1e-6)


# ======================================================================
# 24-25. Metrics and threshold selection
# ======================================================================


def test_metrics_match_hand_computed_values():
    probabilities = [0.9, 0.8, 0.2, 0.1, 0.6, 0.4]
    targets = [1, 0, 0, 0, 1, 1]

    metrics = compute_metrics(probabilities, targets, threshold=0.5)

    # predictions: 1, 1, 0, 0, 1, 0  -> tp=2 fp=1 fn=1 tn=2
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (2, 1, 1, 2)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_metrics_handle_all_negative_predictions():
    """The collapse case must produce zeros, not a divide-by-zero crash."""

    metrics = compute_metrics([0.01] * 10, [0] * 9 + [1], threshold=0.5)

    assert metrics.predicted_positive == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert "WARNING" in metrics.format()


def test_metrics_single_class_gives_nan_auc():
    import math

    metrics = compute_metrics([0.2, 0.8], [0, 0], threshold=0.5)
    assert math.isnan(metrics.roc_auc)


def test_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics([0.1, 0.2, 0.3], [0, 1])


def test_threshold_search_beats_default_on_imbalanced_scores():
    """A model whose positives sit at ~0.3 needs a threshold below 0.5."""

    probabilities = [0.05] * 90 + [0.30, 0.32, 0.28, 0.35, 0.31] * 2
    targets = [0] * 90 + [1] * 10

    at_default = compute_metrics(probabilities, targets, 0.5)
    assert at_default.f1 == 0.0

    threshold, tuned, sweep = find_best_threshold(probabilities, targets)

    assert threshold < 0.5
    assert tuned.f1 > at_default.f1
    assert tuned.f1 == pytest.approx(1.0)
    assert len(sweep) == config.THRESHOLD_SEARCH_STEPS


def test_threshold_search_returns_metrics_at_its_own_threshold():
    probabilities = [0.1, 0.4, 0.6, 0.9]
    targets = [0, 0, 1, 1]

    threshold, metrics, _ = find_best_threshold(probabilities, targets)

    assert metrics.threshold == pytest.approx(threshold)
    recomputed = compute_metrics(probabilities, targets, threshold)
    assert recomputed.f1 == pytest.approx(metrics.f1)


# ======================================================================
# 26-27. Reproducibility
# ======================================================================


def test_set_seed_makes_torch_reproducible():
    from backend.ml.utils.seed import set_seed

    set_seed(123)
    first = torch.randn(8)

    set_seed(123)
    second = torch.randn(8)

    assert torch.equal(first, second)


def test_config_describe_runs():
    text = config.describe()
    assert "microsoft/codebert-base" in text or config.MODEL_NAME in text

