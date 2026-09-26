"""
Stage 18 - BASELINES AND ABLATIONS.

    python -m backend.ml.evaluation.baselines --split test --limit 2000
    python -m backend.ml.evaluation.baselines --split test          # full split

Runs the non-ML detectors over a real split so the learned model has something
honest to be compared against. A vulnerability detector that cannot beat
"grep for strcpy" is not worth deploying, and that comparison only exists if
somebody actually runs the rules.

Baselines implemented here
--------------------------
rules_only    CodeSentinel's rule-based detector (`detect_vulnerabilities`)
taint_only    CodeSentinel's taint-flow detector (`detect_taint_flows`)
rules_taint   the full analyzer pipeline: rules + taint, deduplicated,
              confidence-scored, risk-aggregated

Each produces a per-function score so PR-AUC is computable, not just a single
operating point:

    score = min(1.0, weighted severity mass / SATURATION)

Severity weighting means a CRITICAL finding outranks three LOW ones, which is
closer to how a reviewer triages than a raw count would be.

What is NOT here, and why
-------------------------
ML-only, ML+rules and the full hybrid need the trained classifier's test
predictions, so they are computed by `compare_baselines.py` after Stage 17
rather than here - running the model again would risk a second, unlogged
evaluation of the test split.

GATv2 / CodeBERT+graph is NOT IMPLEMENTED in this repository:
`backend/ml/graphs/` and `backend/ml/embeddings/` contain empty `__init__.py`
files and nothing else. It is reported as unavailable rather than estimated.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from backend.ml import config
from backend.ml.training.metrics import compute_metrics, find_best_threshold
from backend.ml.utils.io import read_jsonl

# The analyzer lives in the FastAPI app package, which is rooted at backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SEVERITY_WEIGHT = {
    "CRITICAL": 1.0,
    "HIGH": 0.7,
    "MEDIUM": 0.4,
    "LOW": 0.2,
    "INFO": 0.05,
}

# Severity mass at which the score saturates to 1.0. One CRITICAL finding is
# already a confident "yes"; beyond that, more findings should not keep
# inflating a probability that is capped at 1 anyway.
SATURATION = 1.0


def _score(vulnerabilities) -> float:
    """Severity-weighted score in [0, 1] from a list of findings."""

    if not vulnerabilities:
        return 0.0

    mass = 0.0

    for vulnerability in vulnerabilities:
        severity = str(vulnerability.get("severity", "MEDIUM")).upper()
        mass += SEVERITY_WEIGHT.get(severity, 0.4)

    return min(1.0, mass / SATURATION)


def run_baseline(name: str, records, language: str) -> dict:
    """Score every record with one detector. Failures are counted, not hidden."""

    from app.services.taint_service import detect_taint_flows
    from app.services.vulnerability_service import detect_vulnerabilities

    if name == "rules_only":
        def detect(code):
            return detect_vulnerabilities(code, language)
    elif name == "taint_only":
        def detect(code):
            return detect_taint_flows(code, language)
    elif name == "rules_taint":
        from app.services.analyzer import analyze_code

        def detect(code):
            return analyze_code(code, language=language)["vulnerabilities"]
    else:
        raise ValueError(f"Unknown baseline: {name}")

    scores = []
    targets = []
    errors = 0
    started = time.time()

    for index, (code, target) in enumerate(records):
        try:
            findings = detect(code)
            scores.append(_score(findings))
        except Exception:  # noqa: BLE001 - see comment
            # Deliberately broad. The analyzer is third-party-ish code reached
            # through tree-sitter and can raise anything on a malformed or
            # unusual function. A parser failure is a real property of the
            # baseline on this input, not something to let abort a 25k-function
            # sweep: it scores 0 and is counted in the reported error column.
            errors += 1
            scores.append(0.0)

        targets.append(target)

        if index and index % 2000 == 0:
            print(f"    ...{index} functions ({time.time() - started:.0f}s)", flush=True)

    return {
        "name": name,
        "scores": scores,
        "targets": targets,
        "errors": errors,
        "seconds": round(time.time() - started, 1),
    }


def load_records(path: Path, limit: int, seed: int):
    """Load (code, target) pairs. A limit takes a seeded stratified sample."""

    records = [
        (record["func"], int(record["target"]))
        for _, record in read_jsonl(path, skip_invalid=True)
    ]

    if not limit or limit >= len(records):
        return records, False

    import random

    rng = random.Random(seed)
    positives = [r for r in records if r[1] == 1]
    negatives = [r for r in records if r[1] == 0]

    # Preserve the real class ratio so precision stays comparable to the
    # full-split numbers rather than being inflated by a balanced sample.
    share = limit / len(records)
    want_positive = max(1, round(len(positives) * share))
    want_negative = limit - want_positive

    rng.shuffle(positives)
    rng.shuffle(negatives)

    sample = positives[:want_positive] + negatives[:want_negative]
    rng.shuffle(sample)

    return sample, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=config.DATA_DIR / "processed_clean")
    parser.add_argument("--split", default="test", choices=list(config.SPLITS))
    parser.add_argument("--language", default="cpp")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Seeded, ratio-preserving sample. 0 = the whole split.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["rules_only", "taint_only", "rules_taint"],
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 78)
    print("CODESENTINEL - STAGE 18: NON-ML BASELINES")
    print("=" * 78)

    path = args.dir / f"primevul_{args.split}.jsonl"

    if not path.exists():
        print(f"\nERROR: {path} not found.")
        raise SystemExit(1)

    records, sampled = load_records(path, args.limit, args.seed)
    positives = sum(t for _, t in records)

    print(f"  split     : {args.split}  ({path})")
    print(f"  functions : {len(records)}" + ("  [SAMPLED]" if sampled else "  [FULL SPLIT]"))
    print(f"  positives : {positives} ({positives / len(records):.2%})")
    print(f"  language  : forced to {args.language}")
    print(
        "\n  Note: language is forced rather than auto-detected. PrimeVul stores\n"
        "  bare function bodies with no #include, so detect_language() cannot\n"
        "  recognise them as C/C++ and the rules would never fire."
    )

    results = {}

    for name in args.baselines:
        print(f"\n  running {name}...", flush=True)
        outcome = run_baseline(name, records, args.language)

        scores, targets = outcome["scores"], outcome["targets"]

        at_default = compute_metrics(scores, targets, 0.5)
        # A rule detector's natural operating point is "it found something",
        # which is any score > 0, not 0.5.
        at_any = compute_metrics(scores, targets, 1e-9)
        tuned_threshold, tuned, _ = find_best_threshold(scores, targets, objective="f1")

        results[name] = {
            # Per-sample scores are persisted so that compare_baselines.py and
            # error_analysis.py can join them against the ML predictions
            # arithmetically. Without them the hybrid comparison cannot be
            # built without re-running everything.
            "scores": [float(s) for s in scores],
            "errors": outcome["errors"],
            "seconds": outcome["seconds"],
            "flagged_any": int(at_any.predicted_positive),
            "metrics_any_finding": at_any.to_dict(),
            "metrics_at_0.5": at_default.to_dict(),
            "best_threshold": tuned_threshold,
            "metrics_at_best_threshold": tuned.to_dict(),
        }

        print(
            f"    {name}: {outcome['seconds']}s, {outcome['errors']} analyzer errors, "
            f"flagged {at_any.predicted_positive}/{len(records)}"
        )
        print(
            f"      any-finding : P={at_any.precision:.4f} R={at_any.recall:.4f} "
            f"F1={at_any.f1:.4f}  PR-AUC={at_any.pr_auc:.4f}"
        )

    print("\n" + "=" * 78)
    print("BASELINE SUMMARY  (operating point: any finding)")
    print("=" * 78)
    print(f"  {'baseline':<14}{'precision':>11}{'recall':>9}{'F1':>8}{'PR-AUC':>9}{'flagged':>9}{'errors':>8}")

    for name, result in results.items():
        m = result["metrics_any_finding"]
        print(
            f"  {name:<14}{m['precision']:>11.4f}{m['recall']:>9.4f}"
            f"{m['f1']:>8.4f}{m['pr_auc']:>9.4f}"
            f"{result['flagged_any']:>9}{result['errors']:>8}"
        )

    baseline_rate = positives / len(records)
    print(f"\n  A always-positive classifier scores PR-AUC = {baseline_rate:.4f} here.")
    print("  Any baseline at or below that is providing no ranking signal at all.")

    print("\n  NOT IMPLEMENTED in this repository (reported, not estimated):")
    print("    CodeBERT + GATv2 : backend/ml/graphs/ is an empty package")

    payload = {
        # Stored once (identical across baselines) so downstream joins can
        # assert alignment rather than assume it.
        "targets": [int(t) for t in targets],
        "split": args.split,
        "path": str(path),
        "functions": len(records),
        "positives": positives,
        "sampled": sampled,
        "seed": args.seed,
        "language": args.language,
        "positive_rate": baseline_rate,
        "results": results,
    }

    out_path = args.out or (config.REPORTS_DIR / f"baselines_{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  written to {out_path}")


if __name__ == "__main__":
    main()
