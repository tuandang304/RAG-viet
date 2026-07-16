"""Benchmark LLM-generated noise on dangdocao_test using existing 3-way pipeline.

Evaluates retrieval performance for each noise type (missing_tone, typo_telex,
informal, code_switch) against the original clean test set and the old rule-based
noisy set (diacritics removal).

Reuses the MLP trained on ViQuAD (zero-shot cross-domain), the dangdocao index,
and the exact same evaluate_all.py pipeline.

Run from repo root:
    uv run python experiments/run_llm_noise_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
INDEX_DIR  = ROOT / "indexes" / "dangdocao"
DATA       = ROOT / "data" / "processed"
RESULTS    = ROOT / "results" / "llm_noise_benchmark"
CHECKPOINT = ROOT / "checkpoints" / "fusion_mlp_3way_full.keras"

_SUBPROC_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "PYTHONUNBUFFERED":     "1",
    "PYTHONIOENCODING":     "utf-8",
}

# All benchmark targets: (file stem relative to DATA, output name)
BENCHMARKS = [
    ("dangdocao_test.jsonl",                   "clean"),
    ("dangdocao_test_noisy.jsonl",             "rule_based_noisy"),
    ("dangdocao_test_llm_missing_tone.jsonl",  "llm_missing_tone"),
    ("dangdocao_test_llm_typo_telex.jsonl",    "llm_typo_telex"),
    ("dangdocao_test_llm_informal.jsonl",      "llm_informal"),
    ("dangdocao_test_llm_code_switch.jsonl",   "llm_code_switch"),
]


def run_eval(qas_file: str, out_name: str, force: bool) -> None:
    qas_path = DATA / qas_file
    out_path = RESULTS / f"{out_name}.json"

    if out_path.exists() and not force:
        print(f"[SKIP] {out_path.name} already exists")
        return

    if not qas_path.exists():
        print(f"[SKIP] {qas_path.name} not found")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "scripts/evaluate_all.py",
        "--qas-path",  str(qas_path),
        "--index-dir",  str(INDEX_DIR),
        "--mlp-path",   str(CHECKPOINT),
        "--output",     str(out_path),
    ]

    print(f"\n{'=' * 78}")
    print(f"Evaluating: {qas_file} -> {out_name}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'=' * 78}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=_SUBPROC_ENV)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[FAIL] {out_name} (exit {result.returncode}) after {elapsed:.0f}s")
    else:
        print(f"[OK] {out_name} ({elapsed:.0f}s)")


def print_comparison() -> None:
    """Print a comparison table of all benchmark results."""
    print(f"\n{'=' * 90}")
    print("  LLM Noise Benchmark — Comparison Summary")
    print(f"{'=' * 90}")

    metrics = ["NDCG@10", "MRR@10", "MAP@10", "Recall@10", "Recall@100", "Hit@1"]
    header = f"  {'Dataset':<25}" + "".join(f"  {m:>10}" for m in metrics)
    print(f"\n{header}")
    print("-" * 90)

    for _, out_name in BENCHMARKS:
        result_path = RESULTS / f"{out_name}.json"
        if not result_path.exists():
            continue
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        mlp = data["methods"]["mlp"]
        row = f"  {out_name:<25}" + "".join(f"  {mlp[m]:>10.4f}" for m in metrics)
        print(row)

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--only", choices=[b[1] for b in BENCHMARKS], default=None,
                        help="Run only one specific benchmark")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only print comparison table (no evaluation)")
    args = parser.parse_args()

    if args.compare_only:
        print_comparison()
        return

    if not CHECKPOINT.exists():
        sys.exit(f"[FAIL] MLP checkpoint not found at {CHECKPOINT}")
    if not INDEX_DIR.exists():
        sys.exit(f"[FAIL] Index directory not found at {INDEX_DIR}")

    print(f"Repo root:   {ROOT}")
    print(f"Index dir:   {INDEX_DIR}")
    print(f"Results:     {RESULTS}")
    print(f"MLP:         {CHECKPOINT}")

    t0 = time.time()

    for qas_file, out_name in BENCHMARKS:
        if args.only and out_name != args.only:
            continue
        run_eval(qas_file, out_name, force=args.force)

    total = (time.time() - t0) / 60
    print(f"\n{'=' * 78}")
    print(f"All benchmarks finished in {total:.1f} min")
    print(f"{'=' * 78}")

    print_comparison()


if __name__ == "__main__":
    main()
