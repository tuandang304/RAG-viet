"""Cross-domain zero-shot evaluation on DANGDOCAO (legal/administrative).

Re-uses the MLP trained on ViQuAD (Wikipedia) and evaluates it against a freshly
built DANGDOCAO index. No DANGDOCAO data touches MLP training — this is the
strict zero-shot cross-domain protocol described in §5.2 of the paper.

Steps (idempotent — skips work already on disk):
  1. Download DANGDOCAO dataset (HuggingFace → JSONL with group-by-passage
     train/dev/test split — see scripts/download_data.py for the leakage fix).
  2. Build FAISS + BM25 + BGE-M3 sparse index for the DANGDOCAO corpus.
  3. Augment dangdocao_test with diacritic-removed copies for noisy eval.
  4. Evaluate the ViQuAD MLP on dangdocao_test (clean + noisy).

Outputs:
  data/processed/dangdocao_*.jsonl
  indexes/dangdocao/{index.faiss, bm25.pkl, sparse.pkl, meta.json}
  results/3way_cross_dangdocao/eval_test.json
  results/3way_cross_dangdocao/eval_test_noisy.json

Run from the repo root:
    uv run python experiments/run_3way_cross_dangdocao.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
INDEX_DIR  = ROOT / "indexes" / "dangdocao"
DATA       = ROOT / "data" / "processed"
RESULTS    = ROOT / "results" / "3way_cross_dangdocao"
CHECKPOINT = ROOT / "checkpoints" / "fusion_mlp_3way_full.pt"  # trained on ViQuAD


_SUBPROC_ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS":      "1",
    "MKL_NUM_THREADS":      "1",
    "PYTHONUNBUFFERED":     "1",
}


def run_step(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"{desc}")
    print(f"$ {' '.join(cmd)}")
    print(f"{'=' * 78}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=_SUBPROC_ENV)
    elapsed = time.time() - t0
    if result.returncode != 0:
        sys.exit(f"\n[FAIL] {desc} (exit {result.returncode}) after {elapsed:.0f}s")
    print(f"[OK] {desc} ({elapsed:.0f}s)")


def step_download(force: bool) -> None:
    test_jsonl = DATA / "dangdocao_test.jsonl"
    if test_jsonl.exists() and not force:
        print(f"[SKIP] dangdocao splits already at {DATA}")
        return
    run_step(
        ["uv", "run", "python", "scripts/download_data.py", "--datasets", "dangdocao"],
        "Step 1/4 — Download DANGDOCAO (HuggingFace → JSONL, group-by-passage split)",
    )


def step_build_index(force: bool) -> None:
    sparse_pkl = INDEX_DIR / "sparse.pkl"
    if sparse_pkl.exists() and not force:
        size_mb = sparse_pkl.stat().st_size / 1e6
        print(f"[SKIP] DANGDOCAO sparse.pkl already at {sparse_pkl} ({size_mb:.1f} MB)")
        return
    run_step(
        [
            "uv", "run", "python", "scripts/build_index.py",
            "--data-path", str(DATA / "dangdocao_passages.jsonl"),
            "--index-dir", str(INDEX_DIR),
        ],
        "Step 2/4 — Build DANGDOCAO index (FAISS + BM25 + BGE-M3 sparse, ~37k passages)",
    )


def step_augment_test(force: bool) -> None:
    noisy_jsonl = DATA / "dangdocao_test_noisy.jsonl"
    if noisy_jsonl.exists() and not force:
        print(f"[SKIP] {noisy_jsonl} exists")
        return
    run_step(
        [
            "uv", "run", "python", "scripts/augment_data.py",
            "--input", str(DATA / "dangdocao_test.jsonl"),
            "--noise-ratio", "0.3",
            "--seed", "42",
        ],
        "Step 3/4 — Augment dangdocao_test with diacritic-removed copies",
    )


def step_eval(qas_rel: str, out_name: str, force: bool) -> None:
    out = RESULTS / f"{out_name}.json"
    if out.exists() and not force:
        print(f"[SKIP] {out} exists")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        sys.exit(
            f"[FAIL] MLP checkpoint not found at {CHECKPOINT}. "
            "Train it first with experiments/run_3way_viaquad.py (Step 2)."
        )
    run_step(
        [
            "uv", "run", "python", "scripts/evaluate_all.py",
            "--qas-path",  str(DATA / qas_rel),
            "--index-dir", str(INDEX_DIR),
            "--mlp-path",  str(CHECKPOINT),
            "--output",    str(out),
        ],
        f"Step 4/4 — Evaluate cross-domain ({out_name}): {qas_rel}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-index",    action="store_true")
    parser.add_argument("--force-augment",  action="store_true")
    parser.add_argument("--force-eval",     action="store_true")
    parser.add_argument("--skip-noisy",     action="store_true",
                        help="Skip the diacritic-removed test eval (saves FPT API calls)")
    parser.add_argument("--only", choices=["download", "index", "augment", "eval"],
                        default=None, help="Run only one stage")
    args = parser.parse_args()

    print(f"Repo root:   {ROOT}")
    print(f"Index dir:   {INDEX_DIR}")
    print(f"Results:     {RESULTS}")
    print(f"MLP source:  {CHECKPOINT} (trained on ViQuAD; DANGDOCAO is zero-shot)")

    t0 = time.time()

    if args.only in (None, "download"):
        step_download(force=args.force_download)
    if args.only in (None, "index"):
        step_build_index(force=args.force_index)
    if args.only in (None, "augment") and not args.skip_noisy:
        step_augment_test(force=args.force_augment)
    if args.only in (None, "eval"):
        step_eval("dangdocao_test.jsonl", "eval_test", force=args.force_eval)
        if not args.skip_noisy:
            step_eval("dangdocao_test_noisy.jsonl", "eval_test_noisy", force=args.force_eval)

    total = (time.time() - t0) / 60
    print(f"\n{'=' * 78}")
    print(f"Cross-domain pipeline finished in {total:.1f} min. Results: {RESULTS}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
