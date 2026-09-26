"""
Phase 8/10 - ERROR ANALYSIS on the frozen test predictions.

    python scripts/error_analysis.py --ml artifacts/reports/test_report.json

Reads the per-sample predictions saved by `test_model.py --save-predictions`
and joins them back to the source records. It does **not** run the model, so
it cannot consume the one-shot test budget no matter how often it is run.

Breakdowns produced
-------------------
  * confusion counts at the frozen threshold
  * performance on TRUNCATED vs NON-TRUNCATED functions
      - this is the one that matters most here: vulnerable functions truncate
        5.9x more often than benign ones, so if the 8-chunk cap is costing
        recall it should show up as a lower recall on the truncated subset
  * performance by function length decile
  * performance by chunk count
  * per-CWE recall (only 11.8% of records carry a CWE, so counts are small)
  * per-project performance for the largest projects
  * score distribution by class
  * rule/ML agreement matrix, if baseline scores are available

Every table reports raw counts alongside rates. A rate computed from a handful
of samples is labelled so it is not read as a stable estimate.

Causality is not claimed anywhere: these are associations in one test split of
one dataset.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Allow the documented invocation (`python scripts/<name>.py`) to work: running a
# script by path puts *its own* directory on sys.path, not the repository root,
# so `import backend...` would fail. Prepending the root keeps the form printed
# in the docstring above honest without requiring PYTHONPATH to be set.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ml.training.metrics import compute_metrics
from backend.ml.utils.io import read_jsonl

# Minimum group size before a rate is treated as meaningful rather than noise.
MIN_GROUP = 20


def bucket_metrics(probabilities, targets, mask, threshold, label):
    """Confusion counts for one subgroup, or None when the group is empty."""

    if mask.sum() == 0:
        return None

    p, t = probabilities[mask], targets[mask]
    predicted = (p >= threshold).astype(int)

    tp = int(((predicted == 1) & (t == 1)).sum())
    fp = int(((predicted == 1) & (t == 0)).sum())
    fn = int(((predicted == 0) & (t == 1)).sum())
    tn = int(((predicted == 0) & (t == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    return {
        "label": label,
        "n": int(mask.sum()),
        "positives": int(t.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall,
        "precision": precision,
        "mean_score_pos": float(p[t == 1].mean()) if (t == 1).any() else float("nan"),
        "mean_score_neg": float(p[t == 0].mean()) if (t == 0).any() else float("nan"),
    }


def print_table(rows, title, note=""):
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  {'group':<28}{'n':>8}{'pos':>7}{'TP':>6}{'FN':>6}{'FP':>7}{'recall':>9}{'prec':>8}")
    print("  " + "-" * 79)

    for row in rows:
        if row is None:
            continue
        small = "" if row["positives"] >= MIN_GROUP else "  *"
        recall = f"{row['recall']:.4f}" if row["recall"] == row["recall"] else "n/a"
        precision = f"{row['precision']:.4f}" if row["precision"] == row["precision"] else "n/a"
        print(
            f"  {row['label']:<28}{row['n']:>8}{row['positives']:>7}"
            f"{row['tp']:>6}{row['fn']:>6}{row['fp']:>7}{recall:>9}{precision:>8}{small}"
        )

    if any(r and r["positives"] < MIN_GROUP for r in rows):
        print(f"  * fewer than {MIN_GROUP} positives — rate is noisy, read the counts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml", type=Path, required=True)
    parser.add_argument("--processed", type=Path, default=Path("data/processed_clean"))
    parser.add_argument("--chunked", type=Path, default=Path("data/chunked"))
    parser.add_argument("--baselines", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.ml.read_text(encoding="utf-8"))

    if "probabilities" not in report:
        raise SystemExit(
            "ERROR: no stored predictions. Re-run test_model.py with "
            "--save-predictions."
        )

    probabilities = np.asarray(report["probabilities"], dtype=np.float64)
    targets = np.asarray(report["targets"], dtype=np.int64)
    threshold = float(report["threshold"])

    # ---- join back to the source records --------------------------------
    records = list(
        read_jsonl(args.processed / f"primevul_{args.split}.jsonl", skip_invalid=True)
    )
    chunk_counts, truncated_flags = [], []

    for _, record in read_jsonl(
        args.chunked / f"primevul_{args.split}_chunked.jsonl", skip_invalid=True
    ):
        chunk_counts.append(int(record.get("num_chunks", 1)))
        truncated_flags.append(bool(record.get("truncated", False)))

    if not (len(records) == len(chunk_counts) == probabilities.size):
        raise SystemExit(
            f"ERROR: misaligned sources — {len(records)} processed, "
            f"{len(chunk_counts)} chunked, {probabilities.size} predictions. "
            f"Refusing to report a join that may be wrong."
        )

    code_lengths = np.array([len(r["func"]) for _, r in records])
    chunk_counts = np.array(chunk_counts)
    truncated = np.array(truncated_flags, dtype=bool)
    projects = [r.get("project", "?") for _, r in records]
    cwes = [r.get("cwe") for _, r in records]

    print("=" * 88)
    print(f"ERROR ANALYSIS — {args.split} split, frozen threshold {threshold:.4f}")
    print("=" * 88)

    overall = compute_metrics(probabilities, targets, threshold)
    print(f"  n={probabilities.size:,}  positives={int(targets.sum())} "
          f"({targets.mean():.2%})")
    print(f"  precision={overall.precision:.4f} recall={overall.recall:.4f} "
          f"F1={overall.f1:.4f} PR-AUC={overall.pr_auc:.4f} ROC-AUC={overall.roc_auc:.4f}")
    print(f"  TP={overall.tp}  FP={overall.fp}  FN={overall.fn}  TN={overall.tn}")

    results = {"overall": overall.to_dict(), "threshold": threshold}

    # ---- 1. truncation --------------------------------------------------
    rows = [
        bucket_metrics(probabilities, targets, ~truncated, threshold, "not truncated"),
        bucket_metrics(probabilities, targets, truncated, threshold, "TRUNCATED (8-chunk cap)"),
    ]
    print_table(
        rows,
        "1. TRUNCATION — does the 8-chunk cap cost recall?",
        "vulnerable functions truncate 5.9x more often than benign ones",
    )
    results["truncation"] = [r for r in rows if r]

    both = [r for r in rows if r]
    if len(both) == 2 and all(r["recall"] == r["recall"] for r in both):
        delta = both[1]["recall"] - both[0]["recall"]
        print(f"\n  recall(truncated) - recall(not truncated) = {delta:+.4f}")
        print("  Association only. Truncated functions are also longer and more")
        print("  complex, so a deficit here is not proof the cap caused it.")

    # ---- 2. length deciles ----------------------------------------------
    edges = np.percentile(code_lengths, np.arange(0, 101, 10))
    rows = []
    for i in range(10):
        low, high = edges[i], edges[i + 1]
        mask = (code_lengths >= low) & (
            code_lengths <= high if i == 9 else code_lengths < high
        )
        rows.append(
            bucket_metrics(probabilities, targets, mask, threshold,
                           f"D{i+1} {int(low)}-{int(high)} chars")
        )
    print_table(rows, "2. FUNCTION LENGTH (character deciles)")
    results["length_deciles"] = [r for r in rows if r]

    # ---- 3. chunk count --------------------------------------------------
    rows = []
    for c in sorted(set(chunk_counts.tolist())):
        rows.append(
            bucket_metrics(probabilities, targets, chunk_counts == c, threshold,
                           f"{c} chunk{'s' if c != 1 else ''}")
        )
    print_table(rows, "3. CHUNK COUNT")
    results["chunk_counts"] = [r for r in rows if r]

    # ---- 4. CWE ----------------------------------------------------------
    cwe_index = defaultdict(list)
    for i, c in enumerate(cwes):
        if not c:
            continue
        for item in (c if isinstance(c, list) else [c]):
            if str(item).strip():
                cwe_index[str(item).strip()].append(i)

    top_cwes = sorted(cwe_index.items(), key=lambda kv: -len(kv[1]))[:12]
    rows = []
    for name, idx in top_cwes:
        mask = np.zeros(probabilities.size, dtype=bool)
        mask[idx] = True
        rows.append(bucket_metrics(probabilities, targets, mask, threshold, name))
    print_table(rows, "4. BY CWE (only 11.8% of records carry one)")
    results["cwe"] = [r for r in rows if r]

    # ---- 5. project -------------------------------------------------------
    counts = Counter(projects)
    rows = []
    for name, _ in counts.most_common(12):
        mask = np.array([p == name for p in projects])
        rows.append(bucket_metrics(probabilities, targets, mask, threshold, name[:26]))
    print_table(rows, "5. BY PROJECT (largest 12)")
    results["projects"] = [r for r in rows if r]

    # ---- 6. score distribution -------------------------------------------
    print("\n6. SCORE DISTRIBUTION")
    for name, mask in (("vulnerable", targets == 1), ("benign", targets == 0)):
        s = probabilities[mask]
        print(f"  {name:<11} n={len(s):>6}  mean={s.mean():.4f}  "
              f"p10={np.percentile(s,10):.4f} p50={np.percentile(s,50):.4f} "
              f"p90={np.percentile(s,90):.4f} max={s.max():.4f}")

    separation = probabilities[targets == 1].mean() - probabilities[targets == 0].mean()
    print(f"  mean separation (pos - neg) = {separation:+.4f}")
    results["score_separation"] = float(separation)

    # ---- 7. rule / ML agreement -------------------------------------------
    if args.baselines and args.baselines.exists():
        base = json.loads(args.baselines.read_text(encoding="utf-8"))
        rule_scores = base["results"].get("rules_only", {}).get("scores")

        if rule_scores and len(rule_scores) == probabilities.size:
            rule_flag = np.asarray(rule_scores) > 0
            ml_flag = probabilities >= threshold

            print("\n7. RULE / ML AGREEMENT")
            print(f"  {'cell':<26}{'n':>8}{'positives':>11}{'precision':>11}")
            print("  " + "-" * 56)

            agreement = {}
            for label, mask in (
                ("both flag", rule_flag & ml_flag),
                ("ML only", ~rule_flag & ml_flag),
                ("rules only", rule_flag & ~ml_flag),
                ("neither", ~rule_flag & ~ml_flag),
            ):
                n = int(mask.sum())
                pos = int(targets[mask].sum())
                prec = pos / n if n else float("nan")
                agreement[label] = {"n": n, "positives": pos, "precision": prec}
                print(f"  {label:<26}{n:>8}{pos:>11}{prec:>11.4f}")

            results["rule_ml_agreement"] = agreement
        else:
            print("\n7. RULE / ML AGREEMENT — skipped (baseline scores unavailable "
                  "or misaligned)")

    out = args.out or Path(f"artifacts/reports/error_analysis_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
