"""Paired significance tests on the RAGAS per-sample scores (offline).

evaluate_ragas.py stores per-sample metric scores keyed by qa_id for every
method. This script pairs dynamic_mlp against each baseline on the qa_ids they
share (samples can be skipped per-method on retrieval/generation failure) and
runs a paired t-test + Wilcoxon signed-rank per metric per condition.

Usage:
    uv run python scripts/ragas_significance.py \\
        --ragas-dir results/ragas_full \\
        --output results/ragas_full/significance.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONDS = [("vq_clean", "ViQuAD clean"), ("vq_noisy", "ViQuAD noisy"),
         ("dd_clean", "DANGDOCAO clean"), ("dd_noisy", "DANGDOCAO noisy")]
BASELINES = ["fixed_equal_4", "toneless_only"]


def paired_scores(ps: dict, method_a: str, method_b: str, metric: str):
    """Aligned (a, b) score arrays over qa_ids where BOTH methods have a value."""
    a_ids = ps[method_a]["qa_ids"]
    b_ids = ps[method_b]["qa_ids"]
    a_map = {qid: v for qid, v in zip(a_ids, ps[method_a]["scores"][metric], strict=True)}
    b_map = {qid: v for qid, v in zip(b_ids, ps[method_b]["scores"][metric], strict=True)}
    common = [q for q in a_map if q in b_map and a_map[q] is not None and b_map[q] is not None]
    return (np.array([a_map[q] for q in common], dtype=np.float64),
            np.array([b_map[q] for q in common], dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas-dir", default="results/ragas_full")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out: dict = {}
    for stem, label in CONDS:
        path = Path(args.ragas_dir) / f"{stem}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        ps = d["per_sample"]
        metrics = list(d["results"]["dynamic_mlp"].keys())
        out[stem] = {}
        print(f"\n=== {label} ===")
        for base in BASELINES:
            if base not in ps:
                continue
            out[stem][base] = {}
            for metric in metrics:
                a, b = paired_scores(ps, "dynamic_mlp", base, metric)
                if len(a) < 5:
                    continue
                diff = a - b
                _, t_p = sp_stats.ttest_rel(a, b)
                try:
                    _, w_p = sp_stats.wilcoxon(diff, alternative="two-sided")
                except ValueError:   # all-zero differences
                    w_p = 1.0
                entry = {
                    "n_pairs": int(len(a)),
                    "mean_delta": round(float(diff.mean()), 4),
                    "ttest_p": float(t_p),
                    "wilcoxon_p": float(w_p),
                }
                out[stem][base][metric] = entry
                star = ("***" if t_p < 1e-3 else "**" if t_p < 1e-2
                        else "*" if t_p < 5e-2 else "ns")
                print(f"  mlp vs {base:<14} {metric:<18} "
                      f"d={entry['mean_delta']:+.4f}  p={t_p:.2e} {star}  (n={len(a)})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
