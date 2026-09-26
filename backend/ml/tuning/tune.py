"""
Stage M - HYPERPARAMETER TUNING.

    # quick smoke search first - proves the harness works before spending compute
    python -m backend.ml.tuning.tune --trials 3 --epochs 1 \
        --limit-train 400 --limit-valid 200

    # real search
    python -m backend.ml.tuning.tune --trials 20 --epochs 3

    # include the architecture comparison (mean vs max vs attention pooling)
    python -m backend.ml.tuning.tune --trials 30 --epochs 3 --search-architecture

THE TEST SPLIT IS NEVER LOADED BY THIS MODULE.

Selection rule
--------------
Trials are scored by **validation PR-AUC**, which is threshold-free. That is
the whole reason the threshold is not tuned here: optimising a thresholded
metric (F1) jointly with the hyperparameters would make the reported best score
a maximum over two nested searches, and therefore biased upward. Ranking
quality is selected first; the operating threshold is chosen once, afterwards,
in Stage N.

Backends
--------
Optuna when installed - TPE sampling plus median pruning, so obviously bad
trials die after their first epoch instead of burning a full run. Otherwise a
seeded random search over the identical space, which is reproducible and needs
no new dependency. Random search is a genuinely strong baseline for this many
dimensions; the Optuna path mainly buys pruning.

Every trial writes its own record, so an interrupted search is still analysable.
"""

import argparse
import gc
import time
from pathlib import Path

from backend.ml import config
from backend.ml.tuning.space import build_space, sample, suggest_optuna
from backend.ml.utils.experiment import RunRecorder
from backend.ml.utils.gpu import resolve_device
from backend.ml.utils.seed import set_seed


def _peak_memory_mib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except ImportError:
        pass

    return 0.0


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except ImportError:
        pass


class TrialRunner:
    """Trains one configuration and returns its validation score."""

    def __init__(self, args, recorder: RunRecorder):
        self.args = args
        self.recorder = recorder
        self.device = resolve_device(prefer_cuda=not args.cpu)
        self.results = []

        from transformers import AutoTokenizer

        from backend.ml.models.collate import make_collate_fn
        from backend.ml.models.dataset import ChunkedFunctionDataset

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if tokenizer.pad_token_id is None:
            raise ValueError(f"{args.model} tokenizer has no pad token id")

        self.collate_fn = make_collate_fn(tokenizer.pad_token_id)

        train_file = args.train_file or (
            args.chunked_dir / "primevul_train_chunked.jsonl"
        )
        print(f"  train file: {train_file}")
        self.train_dataset = ChunkedFunctionDataset(train_file)
        self.valid_dataset = ChunkedFunctionDataset(
            args.chunked_dir / "primevul_valid_chunked.jsonl"
        )

        # Subsampling is stratified and seeded (see train._Subset), so every
        # trial sees the same subset and trials stay comparable.
        if args.limit_train or args.limit_valid:
            from backend.ml.training.train import _Subset

            if args.limit_train:
                self.train_dataset = _Subset(
                    self.train_dataset, args.limit_train, seed=args.seed
                )
            if args.limit_valid:
                self.valid_dataset = _Subset(
                    self.valid_dataset, args.limit_valid, seed=args.seed
                )

        print(
            f"  train: {len(self.train_dataset)} functions"
            f"  valid: {len(self.valid_dataset)} functions"
            f"  device: {self.device}"
        )

    def run(self, trial_number: int, parameters: dict, optuna_trial=None) -> float:
        parameters = {**getattr(self, "pinned", {}), **parameters}
        from backend.ml.models.codebert_classifier import HierarchicalCodeBERTClassifier
        from backend.ml.training.trainer import Trainer

        args = self.args

        print("\n" + "=" * 74)
        print(f"TRIAL {trial_number}")
        print("=" * 74)
        for key, value in sorted(parameters.items()):
            print(f"  {key:<20}: {value}")

        # Same seed every trial: the only thing that differs between trials is
        # the hyperparameters, not the initialisation or the data order.
        set_seed(args.seed)
        _reset_peak_memory()

        started = time.time()

        model = HierarchicalCodeBERTClassifier(
            model_name=args.model,
            chunk_pooling=parameters.get("chunk_pooling", config.CHUNK_POOLING),
            function_pooling=parameters.get(
                "function_pooling", config.FUNCTION_POOLING
            ),
            dropout=parameters.get("dropout", config.CLASSIFIER_DROPOUT),
            gradient_checkpointing=not args.no_grad_checkpointing,
            chunk_micro_batch=parameters.get(
                "chunk_micro_batch", args.chunk_micro_batch
            ),
        )

        # Each trial checkpoints into its own directory, so a later trial can
        # never overwrite an earlier trial's best model.
        trial_directory = self.recorder.directory / f"trial_{trial_number:03d}"

        trainer = Trainer(
            model=model,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            collate_fn=self.collate_fn,
            device=self.device,
            epochs=args.epochs,
            batch_size=parameters.get("batch_size", config.BATCH_SIZE),
            eval_batch_size=args.eval_batch_size,
            learning_rate=parameters["learning_rate"],
            weight_decay=parameters.get("weight_decay", config.WEIGHT_DECAY),
            warmup_ratio=parameters.get("warmup_ratio", config.WARMUP_RATIO),
            grad_accum_steps=parameters.get("grad_accum_steps", 1),
            use_amp=not args.no_amp,
            num_workers=args.num_workers,
            checkpoint_dir=trial_directory,
            # Mid-epoch checkpointing is pure overhead for short trials.
            checkpoint_every=0,
            log_every=args.log_every,
            gpu_log_every=0,
            patience=args.patience,
            seed=args.seed,
            # The per-sample high-loss diagnostic is noise at this volume.
            high_loss_threshold=0.0,
            monitor=args.monitor,
            restore_best=False,
            # A sweep builds a fresh model and DataLoader per trial. PyTorch's
            # pinned-host-memory allocator does not hand that memory back
            # between loader lifetimes, and the process eventually aborts inside
            # its free() with an unhandled C++ exception (observed: trial 3 of a
            # 4-trial run). Unpinned transfers cost a little throughput and buy
            # a sweep that survives to the end.
            pin_memory=False,
        )

        try:
            result = trainer.fit()
        except RuntimeError as error:
            # An OOM is a property of the configuration, not a crash of the
            # search: record it, score it worst, and keep going.
            if "out of memory" not in str(error).lower():
                raise

            print(f"  TRIAL {trial_number} OOM: {error}")
            _reset_peak_memory()

            record = {
                "trial": trial_number,
                "parameters": parameters,
                "status": "oom",
                "score": float("-inf"),
                "seconds": round(time.time() - started, 1),
            }
            self.results.append(record)
            self.recorder.log("trial_oom", **record)

            return float("-inf")

        score = result["best_score"]
        duration = time.time() - started

        record = {
            "trial": trial_number,
            "parameters": parameters,
            "status": "ok",
            "monitor": args.monitor,
            "score": float(score),
            "best_epoch": result["best_epoch"],
            "best_threshold_on_valid": result["best_threshold"],
            "seconds": round(duration, 1),
            "peak_memory_mib": _peak_memory_mib(),
            "history": result["history"],
        }

        self.results.append(record)
        self.recorder.log("trial_finished", **{
            k: v for k, v in record.items() if k not in ("history", "parameters")
        })

        # Written after every trial so an interrupted search is still usable.
        self.recorder.write_json("trials.json", self.results)

        print(
            f"\n  TRIAL {trial_number} {args.monitor}={score:.4f}"
            f"  best_epoch={result['best_epoch']}"
            f"  {duration:.1f}s"
            f"  peak={record['peak_memory_mib']}MiB"
        )

        # Drop every reference before the next trial builds its own model.
        # Without the explicit delete the previous trial's optimizer state and
        # graph stay alive through the local names while the next model is being
        # allocated, which on a 4 GB card is the difference between fitting and
        # not.
        del model, trainer
        gc.collect()
        _reset_peak_memory()

        return float(score)


def run_optuna(runner: TrialRunner, space: dict, args) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Seeded sampler + median pruner. n_warmup_steps=1 lets a trial prove
    # itself for one epoch before it becomes eligible for pruning.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        study_name=args.name,
    )

    counter = {"n": 0}

    def objective(trial):
        counter["n"] += 1
        parameters = suggest_optuna(space, trial)
        return runner.run(counter["n"], parameters, optuna_trial=trial)

    study.optimize(objective, n_trials=args.trials)

    runner.recorder.write_json(
        "optuna_best.json",
        {"best_value": study.best_value, "best_params": study.best_params},
    )


def run_random(runner: TrialRunner, space: dict, args) -> None:
    import numpy as np

    rng = np.random.default_rng(args.seed)

    for number in range(1, args.trials + 1):
        runner.run(number, sample(space, rng))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--name", default="tuning")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)

    parser.add_argument("--chunked-dir", type=Path, default=config.CHUNKED_DIR)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=None,
        help="Override the chunked training file (e.g. an undersampled one)",
    )
    parser.add_argument(
        "--fix",
        nargs="+",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Pin a search-space parameter instead of searching it, e.g. "
            "--fix batch_size=8. Useful when a value has already been settled "
            "by measurement and searching it would waste trials."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=config.TUNING_DIR)
    parser.add_argument("--model", default=config.MODEL_NAME)

    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)

    parser.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument(
        "--chunk-micro-batch", type=int, default=config.CHUNK_MICRO_BATCH
    )
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--monitor", default=config.EARLY_STOPPING_MONITOR)

    parser.add_argument(
        "--lr-min",
        type=float,
        default=None,
        help="Narrow the learning-rate search range (lower bound)",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=None,
        help="Narrow the learning-rate search range (upper bound)",
    )
    parser.add_argument("--search-architecture", action="store_true")
    parser.add_argument("--search-memory", action="store_true")

    parser.add_argument("--backend", choices=("auto", "optuna", "random"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    space = build_space(
        include_architecture=args.search_architecture,
        include_memory=args.search_memory,
    )

    # A narrowed learning-rate range, used to focus a follow-up search on the
    # region an earlier round pointed at. Recorded in config.json so the final
    # space is never inferred from the defaults.
    if args.lr_min is not None or args.lr_max is not None:
        low, high = space["learning_rate"][1]
        space["learning_rate"] = (
            "loguniform",
            (args.lr_min if args.lr_min is not None else low,
             args.lr_max if args.lr_max is not None else high),
        )

    # Pinned parameters leave the search space entirely and are merged back into
    # every trial's parameter dict, so they are still recorded in the results.
    pinned = {}

    for item in args.fix:
        if "=" not in item:
            raise SystemExit(f"--fix expects NAME=VALUE, got {item!r}")
        name, _, raw = item.partition("=")
        name = name.strip()
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw.strip()
        pinned[name] = value
        space.pop(name, None)

    backend = args.backend

    if backend == "auto":
        try:
            import optuna  # noqa: F401

            backend = "optuna"
        except ImportError:
            backend = "random"

    print("=" * 74)
    print("CODESENTINEL - HYPERPARAMETER TUNING")
    print("=" * 74)
    print(f"  backend     : {backend}")
    print(f"  trials      : {args.trials}")
    print(f"  epochs/trial: {args.epochs}")
    print(f"  objective   : validation {args.monitor} (threshold-free)")
    print(f"  seed        : {args.seed}")
    print("  test split  : NOT LOADED")
    print("\n  search space:")
    for name, (kind, specification) in sorted(space.items()):
        print(f"    {name:<20} {kind:<12} {specification}")

    print()

    recorder = RunRecorder(
        args.name,
        root=args.out_dir,
        config_dict={
            "args": vars(args),
            "space": {k: list(v) for k, v in space.items()},
            "pinned": pinned,
            "backend": backend,
        },
    )

    print(f"  artifacts -> {recorder.directory}\n")

    with recorder:
        runner = TrialRunner(args, recorder)
        runner.pinned = pinned

        if pinned:
            print(f"  pinned (not searched): {pinned}")

        if backend == "optuna":
            run_optuna(runner, space, args)
        else:
            run_random(runner, space, args)

        successful = [r for r in runner.results if r["status"] == "ok"]

        if not successful:
            print("\nNo trial completed successfully.")
            raise SystemExit(1)

        best = max(successful, key=lambda r: r["score"])

        recorder.write_json("best_trial.json", best)

        print("\n" + "=" * 74)
        print("TUNING COMPLETE")
        print("=" * 74)
        print(f"  trials run  : {len(runner.results)} ({len(successful)} successful)")
        print(f"  best trial  : {best['trial']}")
        print(f"  best {args.monitor:<7}: {best['score']:.4f}")
        print(f"  best epoch  : {best['best_epoch']}")
        print("\n  best parameters:")
        for key, value in sorted(best["parameters"].items()):
            print(f"    {key:<20}: {value}")

        print("\n  Leaderboard:")
        print(f"    {'trial':>6}{'score':>10}{'epoch':>7}{'secs':>8}  parameters")
        for record in sorted(successful, key=lambda r: -r["score"]):
            summary = ", ".join(
                f"{k}={v:.2e}" if isinstance(v, float) and v < 0.01 else f"{k}={v}"
                for k, v in sorted(record["parameters"].items())
            )
            print(
                f"    {record['trial']:>6}{record['score']:>10.4f}"
                f"{record['best_epoch']:>7}{record['seconds']:>8.0f}  {summary[:70]}"
            )

        print(f"\n  Full results: {recorder.directory}")
        print("\n  Next: train the final model with these parameters, then run")
        print("  Stage N threshold selection on its validation predictions.")


if __name__ == "__main__":
    main()
