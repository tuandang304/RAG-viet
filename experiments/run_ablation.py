"""Soft-label ablation: train MLP with T ∈ {0.1, 1.0} + hard-label, eval each on dev.

Re-uses the existing FPT query-embedding cache `checkpoints/train_aug_embeddings.npy`,
so no extra FPT API calls are issued. BGE-M3 sparse encoding (Phase 2 of training and
sparse retrieval during eval) does run on the GPU — when running this in parallel
with the RAGAS orchestrator, the two processes share the RTX 3050 (≈ 2 GB each at
fp16) and stay under the 6 GB VRAM limit.

Outputs:
  checkpoints/fusion_mlp_T0.1.pt
  checkpoints/fusion_mlp_T1.0.pt
  checkpoints/fusion_mlp_hard.pt
  results/3way_full/eval_dev_T0.1.json
  results/3way_full/eval_dev_T1.0.json
  results/3way_full/eval_dev_hard.json
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
INDEX_DIR  = ROOT / "indexes" / "viaquad"
DATA       = ROOT / "data" / "processed"
CKPT_DIR   = ROOT / "checkpoints"
RESULTS    = ROOT / "results" / "3way_full"
EMB_CACHE  = CKPT_DIR / "train_aug_embeddings.npy"

VARIANTS = [
    # (label, train extra args, checkpoint filename, eval JSON name)
    ("T=0.1",      ["--temperature", "0.1"], "fusion_mlp_T0.1.pt", "eval_dev_T0.1.json"),
    ("T=1.0",      ["--temperature", "1.0"], "fusion_mlp_T1.0.pt", "eval_dev_T1.0.json"),
    ("HARD-label", ["--hard-label"],         "fusion_mlp_hard.pt", "eval_dev_hard.json"),
]


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


def train_variant(label: str, extra: list[str], ckpt_name: str) -> Path:
    ckpt = CKPT_DIR / ckpt_name
    if ckpt.exists():
        print(f"[SKIP] {ckpt} already exists")
        return ckpt
    run_step(
        [
            "uv", "run", "python", "scripts/train_mlp.py",
            "--qas-path",  str(DATA / "viaquad_train_aug.jsonl"),
            "--index-dir", str(INDEX_DIR),
            "--output",    str(ckpt),
            "--emb-cache", str(EMB_CACHE),
            "--max-samples", "5000",
            "--epochs", "100",
            *extra,
        ],
        f"Train MLP ({label})",
    )
    return ckpt


def eval_dev(ckpt: Path, out_name: str, label: str) -> None:
    out = RESULTS / out_name
    if out.exists():
        print(f"[SKIP] {out} already exists")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_step(
        [
            "uv", "run", "python", "scripts/evaluate_all.py",
            "--qas-path",  str(DATA / "viaquad_dev.jsonl"),
            "--index-dir", str(INDEX_DIR),
            "--mlp-path",  str(ckpt),
            "--output",    str(out),
        ],
        f"Eval dev for {label} → {out_name}",
    )


def main() -> None:
    t0 = time.time()
    print(f"Repo root: {ROOT}")
    print(f"Variants:  {[v[0] for v in VARIANTS]}")
    for label, extra, ckpt_name, out_name in VARIANTS:
        ckpt = train_variant(label, extra, ckpt_name)
        eval_dev(ckpt, out_name, label)
    total = (time.time() - t0) / 60
    print(f"\n{'=' * 78}\nAblation pipeline finished in {total:.1f} min.\n{'=' * 78}")


if __name__ == "__main__":
    main()
