"""Run RAGAS end-to-end across the 4 paper conditions (full sample counts).

Sequentially calls scripts/evaluate_ragas.py for each (domain × noise) cell,
harnessing the incremental save + resume logic so a long-running job can be
killed and restarted without losing completed methods.

Defaults match the paper's "RAGAS end-to-end" section:
  - 4 conditions × full QA file (no subsampling)
  - methods: dynamic_mlp, fixed_equal_4, toneless_only, dense_only
  - LLM judge + generator: MiniMax-M3 (--llm-provider minimax)
  - Embeddings: FPT (unchanged)
  - RAGAS metrics: context_precision, context_recall, faithfulness

Outputs:
  results/ragas_full/vq_clean.json
  results/ragas_full/vq_noisy.json
  results/ragas_full/dd_clean.json
  results/ragas_full/dd_noisy.json

After all four complete, calls scripts/ragas_significance.py to refresh
results/ragas_full/significance.json.

Usage:
  uv run python scripts/run_ragas_full.py
  uv run python scripts/run_ragas_full.py --dry-run          # just print plan
  uv run python scripts/run_ragas_full.py --skip-significance
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RAGAS_DIR = ROOT / "results" / "ragas_full"
MLP = ROOT / "checkpoints" / "fusion_mlp_4way_aug.keras"

CONDS = [
    ("vq_clean", "data/processed/viaquad_dev.jsonl",
     "indexes/viaquad", "results/ragas_full/vq_clean.json"),
    ("vq_noisy", "data/processed/viaquad_dev_noisy.jsonl",
     "indexes/viaquad", "results/ragas_full/vq_noisy.json"),
    ("dd_clean", "data/processed/dangdocao_test.jsonl",
     "indexes/dangdocao", "results/ragas_full/dd_clean.json"),
    ("dd_noisy", "data/processed/dangdocao_test_noisy.jsonl",
     "indexes/dangdocao", "results/ragas_full/dd_noisy.json"),
]


def _all_done(out_path: Path, wanted: list[str]) -> bool:
    """True if out_path exists and lists every wanted method as completed."""
    if not out_path.exists():
        return False
    try:
        d = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    done = set(d.get("completed_methods", []))
    return all(m in done for m in wanted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="dynamic_mlp,fixed_equal_4,toneless_only,dense_only")
    parser.add_argument("--ragas-metrics",
                        default="context_precision,context_recall,faithfulness")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Samples per condition (0 = use all). Forwarded to "
                             "evaluate_ragas.py --n-samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip a condition whose output already lists all methods done.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every condition even if completed.")
    parser.add_argument("--skip-significance", action="store_true",
                        help="Don't run ragas_significance.py at the end.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan + sample counts, then exit.")
    args = parser.parse_args()

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    RAGAS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {RAGAS_DIR}")
    print(f"Methods   : {wanted}")
    print(f"MLP       : {MLP}")
    print(f"LLM judge : MiniMax-M3 (via --llm-provider minimax)")
    print(f"Embeddings: FPT Vietnamese_Embedding")
    print()

    if args.dry_run:
        for stem, qas_path, index_dir, out_rel in CONDS:
            n = sum(1 for _ in open(ROOT / qas_path, encoding="utf-8"))
            print(f"  [{stem:<10}] n={n:<5}  qa={qas_path}  idx={index_dir}  → {out_rel}")
        return

    if not MLP.exists():
        sys.exit(f"ERROR: MLP checkpoint not found: {MLP}")

    for stem, qas_path, index_dir, out_rel in CONDS:
        out_path = ROOT / out_rel
        if args.force:
            should_skip = False
        elif args.skip_existing and _all_done(out_path, wanted):
            print(f"[{stem}] already complete — skipping")
            continue
        else:
            should_skip = False

        print(f"\n{'=' * 70}")
        print(f"[{stem}] {qas_path}  ({index_dir})")
        print(f"  output: {out_path}")
        print(f"{'=' * 70}")

        cmd = [
            "uv", "run", "python", "scripts/evaluate_ragas.py",
            "--qas-path", qas_path,
            "--index-dir", index_dir,
            "--mlp-path",  str(MLP.relative_to(ROOT)),
            "--methods",   args.methods,
            "--ragas-metrics", args.ragas_metrics,
            "--seed",      str(args.seed),
            "--llm-provider", "minimax",
            "--output",    out_rel,
        ]
        if args.n_samples:
            cmd += ["--n-samples", str(args.n_samples)]
        # --resume-from points at the same output file: it reads partial state
        # (completed_methods) on entry and writes back atomically after each
        # method, so a Ctrl-C mid-method leaves the file consistent.
        cmd += ["--resume-from", out_rel]
        print("$", " ".join(cmd), flush=True)

        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(f"[{stem}] evaluate_ragas.py exited {rc} — aborting. "
                     "Re-run to resume from partial state.")

    if args.skip_significance:
        print("\n[skip] --skip-significance set; leaving existing significance.json alone")
        return

    print("\n=== Computing paired significance ===")
    sig_out = RAGAS_DIR / "significance.json"
    rc = subprocess.call([
        "uv", "run", "python", "scripts/ragas_significance.py",
        "--ragas-dir", str(RAGAS_DIR),
        "--output",    str(sig_out),
    ], cwd=str(ROOT))
    if rc != 0:
        sys.exit(f"ragas_significance.py exited {rc}")


if __name__ == "__main__":
    main()