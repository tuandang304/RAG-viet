"""End-to-end RAG quality (§5.7): RAGAS clean + diacritic-noisy.

Runs `scripts/evaluate_ragas.py` on both `viaquad_dev.jsonl` and
`viaquad_dev_noisy.jsonl` at n=50 each. Uses the three-way MLP checkpoint from
the headline run (`checkpoints/fusion_mlp_3way_full.pt`).

Outputs:
  results/3way_full/ragas_clean.json
  results/3way_full/ragas_noisy.json
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
RESULTS    = ROOT / "results" / "3way_full"
CHECKPOINT = ROOT / "checkpoints" / "fusion_mlp_3way_full.pt"
N_SAMPLES  = 50

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


def run_ragas(qas_rel: str, out_name: str, label: str) -> None:
    out = RESULTS / out_name
    if out.exists():
        print(f"[SKIP] {out} already exists")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_step(
        [
            "uv", "run", "python", "scripts/evaluate_ragas.py",
            "--qas-path",  str(DATA / qas_rel),
            "--index-dir", str(INDEX_DIR),
            "--mlp-path",  str(CHECKPOINT),
            "--n-samples", str(N_SAMPLES),
            "--output",    str(out),
        ],
        f"RAGAS {label}: {qas_rel}",
    )


def main() -> None:
    t0 = time.time()
    print(f"Repo root: {ROOT}")
    print(f"MLP:       {CHECKPOINT}")
    print(f"n-samples: {N_SAMPLES} per condition")
    run_ragas("viaquad_dev.jsonl",       "ragas_clean.json", "clean")
    run_ragas("viaquad_dev_noisy.jsonl", "ragas_noisy.json", "noisy")
    total = (time.time() - t0) / 60
    print(f"\n{'=' * 78}\nRAGAS pipeline finished in {total:.1f} min.\n{'=' * 78}")


if __name__ == "__main__":
    main()
