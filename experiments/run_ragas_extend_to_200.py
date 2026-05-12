"""Extend the existing 50-sample RAGAS evaluation to a total of 200 samples.

Strategy: re-derive the 50 qa_ids from the previous run by re-running
`random.sample(qas, 50, seed=42)` against the same QA file (deterministic), then
sample 150 *new* queries disjoint from those 50, score them with the same
five methods and four metrics, and merge the two sets by a sample-count-weighted
mean to obtain final 200-sample averages.

Outputs:
  data/derived/ragas_excluded_ids_clean.json   — 50 ids re-derived from the 50-run
  data/derived/ragas_excluded_ids_noisy.json
  results/3way_full/ragas_clean_extra150.json  — fresh 150-sample run, disjoint
  results/3way_full/ragas_noisy_extra150.json
  results/3way_full/ragas_clean_200.json       — merged 50 + 150 means
  results/3way_full/ragas_noisy_200.json
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
INDEX_DIR  = ROOT / "indexes" / "viaquad"
DATA       = ROOT / "data" / "processed"
DERIVED    = ROOT / "data" / "derived"
RESULTS    = ROOT / "results" / "3way_full"
CHECKPOINT = ROOT / "checkpoints" / "fusion_mlp_3way_full.pt"
SEED       = 42
N_EXISTING = 50
N_TARGET   = 200
N_EXTRA    = N_TARGET - N_EXISTING   # 150


_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "PYTHONUNBUFFERED":     "1",
}


def run_step(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 78}\n{desc}\n$ {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=_ENV).returncode
    el = time.time() - t0
    if rc != 0:
        sys.exit(f"[FAIL] {desc} (exit {rc}) after {el:.0f}s")
    print(f"[OK] {desc} ({el:.0f}s)")


def derive_existing_ids(qas_jsonl: Path, n: int, seed: int) -> list[str]:
    """Reproduce the qa_ids the previous evaluate_ragas run sampled.

    The previous run did exactly: `qas = [q for q in qas if q.get("answers") and q["answers"][0]]`
    then `random.sample(qas, n)` after `random.seed(seed)`. We replicate it.
    """
    with open(qas_jsonl, encoding="utf-8") as f:
        qas = [json.loads(l) for l in f if l.strip()]
    qas = [q for q in qas if q.get("answers") and q["answers"][0]]
    random.seed(seed)
    sampled = random.sample(qas, n)
    return [str(q["id"]) for q in sampled]


def merge_results(existing: dict, extra: dict, n_existing: int, n_extra: int) -> dict:
    """Combine two RAGAS result JSON objects by sample-count-weighted means.

    Both files have shape `{"results": {method: {metric: mean, ...}, ...}, ...}`.
    The output preserves the same shape with merged means and `n_samples = sum`.
    """
    merged: dict[str, dict[str, float]] = {}
    methods = set(existing["results"]) | set(extra["results"])
    for m in sorted(methods):
        e_m = existing["results"].get(m, {})
        x_m = extra["results"].get(m, {})
        metrics = set(e_m) | set(x_m)
        merged[m] = {}
        for k in sorted(metrics):
            e_v = e_m.get(k)
            x_v = x_m.get(k)
            if e_v is None and x_v is None:
                continue
            if e_v is None:
                merged[m][k] = round(float(x_v), 4)
            elif x_v is None:
                merged[m][k] = round(float(e_v), 4)
            else:
                merged[m][k] = round(
                    (n_existing * float(e_v) + n_extra * float(x_v)) / (n_existing + n_extra),
                    4,
                )
    return {
        "results":   merged,
        "n_samples": n_existing + n_extra,
        "source":    {
            "existing": {"n": n_existing, "file": str(existing.get("qas_path", "?"))},
            "extra":    {"n": n_extra,    "file": str(extra.get("qas_path", "?"))},
        },
    }


def extend_condition(qas_filename: str, prev_json: Path, label: str) -> None:
    qas_path  = DATA / qas_filename
    extra_out = RESULTS / f"ragas_{label}_extra{N_EXTRA}.json"
    merged_out = RESULTS / f"ragas_{label}_{N_TARGET}.json"
    exclude_path = DERIVED / f"ragas_excluded_ids_{label}.json"

    DERIVED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Re-derive the 50 qa_ids that the prior run consumed.
    if exclude_path.exists():
        print(f"[SKIP] {exclude_path} already exists")
    else:
        ids = derive_existing_ids(qas_path, N_EXISTING, SEED)
        exclude_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK]   Wrote {len(ids)} excluded ids → {exclude_path}")

    # 2. Run RAGAS on 150 fresh samples disjoint from those 50.
    if extra_out.exists():
        print(f"[SKIP] {extra_out} already exists")
    else:
        run_step(
            [
                "uv", "run", "python", "scripts/evaluate_ragas.py",
                "--qas-path",     str(qas_path),
                "--index-dir",    str(INDEX_DIR),
                "--mlp-path",     str(CHECKPOINT),
                "--n-samples",    str(N_EXTRA),
                "--seed",         str(SEED),
                "--exclude-ids",  str(exclude_path),
                "--output",       str(extra_out),
            ],
            f"RAGAS extra {N_EXTRA} ({label}, disjoint from the original {N_EXISTING})",
        )

    # 3. Merge with previously saved 50-sample results.
    if not prev_json.exists():
        sys.exit(f"[FAIL] previous result file not found: {prev_json}")
    with open(prev_json,  encoding="utf-8") as f: existing = json.load(f)
    with open(extra_out, encoding="utf-8") as f: extra    = json.load(f)

    merged = merge_results(existing, extra, N_EXISTING, N_EXTRA)
    merged_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK]   Merged {N_EXISTING}+{N_EXTRA}={N_TARGET} → {merged_out}")
    for m, scores in merged["results"].items():
        print(f"  {m:16s}  " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))


def main() -> None:
    t0 = time.time()
    print(f"Target total samples: {N_TARGET}  ({N_EXISTING} existing + {N_EXTRA} new)")
    extend_condition("viaquad_dev.jsonl",       RESULTS / "ragas_clean.json", "clean")
    extend_condition("viaquad_dev_noisy.jsonl", RESULTS / "ragas_noisy.json", "noisy")
    total = (time.time() - t0) / 60
    print(f"\n{'=' * 78}\nExtension pipeline finished in {total:.1f} min.\n{'=' * 78}")


if __name__ == "__main__":
    main()
