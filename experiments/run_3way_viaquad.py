"""Full 3-way ViQuAD pipeline orchestrator.

Steps (idempotent — skips work already on disk):
  1. Build BGE-M3 sparse index for the ViQuAD corpus (FAISS + BM25 reused if present).
  2. Train the fusion MLP on the augmented training set with 3-way soft labels.
  3. Evaluate on three conditions: dev clean, test clean, dev diacritic-noisy.

Outputs:
  indexes/viaquad/sparse.pkl                 — BGE-M3 inverted index
  checkpoints/fusion_mlp_3way_full.pt        — MLP checkpoint
  checkpoints/train_aug_embeddings.npy       — cached query embeddings
  results/3way_full/eval_dev.json
  results/3way_full/eval_test.json
  results/3way_full/eval_dev_noisy.json

Run from the repo root:
    uv run python experiments/run_3way_viaquad.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR  = ROOT / "indexes" / "viaquad"
CKPT_DIR   = ROOT / "checkpoints"
RESULTS    = ROOT / "results" / "3way_full"
DATA       = ROOT / "data" / "processed"

CHECKPOINT = CKPT_DIR / "fusion_mlp_3way_full.pt"
EMB_CACHE  = CKPT_DIR / "train_aug_embeddings.npy"


import os


# Environment for every subprocess. KMP_DUPLICATE_LIB_OK works around the
# FAISS-MKL ↔ PyTorch-OMP runtime collision (segfaults / Windows 0xC0000005)
# documented in CLAUDE.md. OMP_NUM_THREADS=1 keeps FAISS from spawning a
# thread pool that races with PyTorch's CUDA stream worker.
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


def step_build_sparse(force: bool) -> None:
    sparse_pkl = INDEX_DIR / "sparse.pkl"
    faiss_idx  = INDEX_DIR / "index.faiss"
    bm25_pkl   = INDEX_DIR / "bm25.pkl"
    if sparse_pkl.exists() and not force:
        print(f"[SKIP] sparse.pkl already at {sparse_pkl} ({sparse_pkl.stat().st_size / 1e6:.1f} MB)")
        return
    if not (faiss_idx.exists() and bm25_pkl.exists()):
        sys.exit(
            f"[FAIL] FAISS / BM25 index missing in {INDEX_DIR}. "
            "Run `uv run python scripts/build_index.py --no-sparse ...` first, "
            "or remove `--no-sparse` to build everything from scratch."
        )
    # build_index.py rebuilds all three signals. To preserve existing FAISS/BM25
    # we simply re-run the script; it overwrites those files with identical content
    # (deterministic given the same passages.jsonl).
    run_step(
        [
            "uv", "run", "python", "scripts/build_index.py",
            "--data-path", str(DATA / "viaquad_passages.jsonl"),
            "--index-dir", str(INDEX_DIR),
        ],
        "Step 1/3 — Build BGE-M3 sparse index (downloads ~570 MB on first run)",
    )


def step_train(force: bool) -> None:
    if CHECKPOINT.exists() and not force:
        print(f"[SKIP] MLP checkpoint at {CHECKPOINT}")
        return
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_step(
        [
            "uv", "run", "python", "scripts/train_mlp.py",
            "--qas-path",  str(DATA / "viaquad_train_aug.jsonl"),
            "--index-dir", str(INDEX_DIR),
            "--output",    str(CHECKPOINT),
            "--emb-cache", str(EMB_CACHE),
            "--max-samples", "5000",
            "--epochs",      "100",
            "--temperature", "0.3",
        ],
        "Step 2/3 — Train 3-way fusion MLP (5,000 samples · 100 epochs · T=0.3)",
    )


def step_eval(qas_rel: str, out_name: str, force: bool) -> None:
    out = RESULTS / f"{out_name}.json"
    if out.exists() and not force:
        print(f"[SKIP] {out} exists ({out.stat().st_size / 1024:.1f} KB)")
        # Print a one-line summary so the user sees the cached number.
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            mlp = data["methods"]["mlp"]
            print(f"        cached: NDCG@10={mlp['ndcg10']:.4f}  MRR@10={mlp['mrr10']:.4f}")
        except Exception:
            pass
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_step(
        [
            "uv", "run", "python", "scripts/evaluate_all.py",
            "--qas-path",  str(DATA / qas_rel),
            "--index-dir", str(INDEX_DIR),
            "--mlp-path",  str(CHECKPOINT),
            "--output",    str(out),
        ],
        f"Step 3/3 — Evaluate ({out_name}): {qas_rel}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force-sparse", action="store_true", help="Rebuild sparse index")
    parser.add_argument("--force-train",  action="store_true", help="Retrain MLP")
    parser.add_argument("--force-eval",   action="store_true", help="Re-evaluate (overwrite JSONs)")
    parser.add_argument("--only", choices=["sparse", "train", "eval"], default=None,
                        help="Run only one stage")
    args = parser.parse_args()

    print(f"Repo root: {ROOT}")
    print(f"Index dir: {INDEX_DIR}")
    print(f"Results:   {RESULTS}")

    pipeline_start = time.time()

    if args.only in (None, "sparse"):
        step_build_sparse(force=args.force_sparse)
    if args.only in (None, "train"):
        step_train(force=args.force_train)
    if args.only in (None, "eval"):
        for qas, name in [
            ("viaquad_dev.jsonl",       "eval_dev"),
            ("viaquad_test.jsonl",      "eval_test"),
            ("viaquad_dev_noisy.jsonl", "eval_dev_noisy"),
        ]:
            step_eval(qas, name, force=args.force_eval)

    total = time.time() - pipeline_start
    print(f"\n{'=' * 78}")
    print(f"Pipeline finished in {total / 60:.1f} min. Results: {RESULTS}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
