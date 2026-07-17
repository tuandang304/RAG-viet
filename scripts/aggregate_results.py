"""Consolidate evaluation JSONs into the paper's result tables (Markdown).

Reads whatever result files exist and prints:
  1. Main baseline table   — P1d full test (results/p1_full_*.json)
  2. OOD LLM-noise table    — P2.7 (results/llm_noise_4way/*.json)
  3. Partial-noise curve    — P2.6 (results/curve/*.json)
  4. Restoration comparison — P1c (results/p1_restored_*.json vs *_noisy)
  5. RAGAS end-to-end       — (results/ragas_4way/*.json)

Missing files are skipped with a note, so it is safe to run mid-batch.

Usage:  uv run python scripts/aggregate_results.py
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path("results")
# Method display order + labels for the retrieval tables.
METHOD_ORDER = [
    ("mlp", "Router (4-way, ours)"),
    ("tuned_fixed", "Best fixed (dev-tuned)"),
    ("fixed_equal_4", "Fixed equal 4-way"),
    ("rrf", "RRF"),
    ("fixed_equal", "Fixed equal 3-way"),
    ("toneless", "Toneless only"),
    ("dense", "Dense only"),
    ("bm25", "BM25 only"),
    ("sparse", "Sparse only"),
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sig_marker(d: dict, method: str) -> str:
    """Return significance vs router: *** p<1e-3, ** p<0.01, * p<0.05, ns."""
    if method == "mlp":
        return ""
    sig = d.get("significance", {}).get(f"mlp_vs_{method}")
    if not sig:
        return ""
    p = sig["ttest_p"]
    if p != p:  # nan
        return "—"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def table_retrieval(files: list[tuple[str, Path]], title: str) -> None:
    loaded = [(name, _load(p)) for name, p in files]
    loaded = [(n, d) for n, d in loaded if d is not None]
    if not loaded:
        print(f"\n### {title}\n_(no result files yet)_")
        return
    print(f"\n### {title}\n")
    cols = " | ".join(f"{n} (n={d['n_queries']})" for n, d in loaded)
    print(f"| Method | {cols} |")
    print("|" + "---|" * (len(loaded) + 1))
    present = [m for m, _ in METHOD_ORDER if any(m in d["methods"] for _, d in loaded)]
    label = dict(METHOD_ORDER)
    for m in present:
        cells = []
        for _, d in loaded:
            v = d["methods"].get(m)
            if v is None:
                cells.append("—")
            else:
                mark = _sig_marker(d, m)
                cells.append(f"{v['NDCG@10']:.4f}{(' ' + mark) if mark else ''}")
        print(f"| {label[m]} | {' | '.join(cells)} |")
    print("\n_NDCG@10; significance of router vs baseline: *** p<0.001, ** p<0.01, * p<0.05, ns not sig, — undefined._")


def table_curve() -> None:
    curve_dir = RESULTS / "curve"
    if not curve_dir.exists():
        print("\n### Partial-noise curve (P2.6)\n_(no result files yet)_")
        return
    print("\n### Partial-noise curve (P2.6) — NDCG@10 vs % syllables stripped\n")
    for dom in ("viaquad", "dangdocao"):
        rows = []
        for p in (0, 25, 50, 75, 100):
            d = _load(curve_dir / f"{dom}_p{p}.json")
            if d:
                m = d["methods"]
                rows.append((p, m))
        if not rows:
            continue
        methods = ["mlp", "tuned_fixed", "fixed_equal", "toneless", "dense"]
        present = [mm for mm in methods if any(mm in m for _, m in rows)]
        label = dict(METHOD_ORDER)
        print(f"**{dom}**\n")
        print("| Noise % | " + " | ".join(label.get(mm, mm) for mm in present) + " |")
        print("|" + "---|" * (len(present) + 1))
        for p, m in rows:
            cells = [f"{m[mm]['NDCG@10']:.4f}" if mm in m else "—" for mm in present]
            print(f"| {p}% | {' | '.join(cells)} |")
        print()


def table_restoration() -> None:
    print("\n### Diacritic restoration vs toneless channel (P1c)\n")
    pairs = [
        ("ViQuAD", "p1_restored_viaquad.json", "p1_full_viaquad_noisy.json"),
        ("DANGDOCAO", "p1_restored_dangdocao.json", "p1_full_dangdocao_noisy.json"),
    ]
    any_row = False
    print("| Domain | Router on noisy | Router on restored | Δ |")
    print("|---|---|---|---|")
    for dom, restored_f, noisy_f in pairs:
        r = _load(RESULTS / restored_f)
        n = _load(RESULTS / noisy_f)
        if not r:
            continue
        any_row = True
        r_mlp = r["methods"]["mlp"]["NDCG@10"]
        n_mlp = n["methods"]["mlp"]["NDCG@10"] if n else None
        delta = f"{r_mlp - n_mlp:+.4f}" if n_mlp is not None else "—"
        noisy_cell = f"{n_mlp:.4f}" if n_mlp is not None else "—"
        print(f"| {dom} | {noisy_cell} | {r_mlp:.4f} | {delta} |")
    if not any_row:
        print("_(no restoration result files yet)_")


def table_ragas() -> None:
    ragas_dir = RESULTS / "ragas_4way"
    if not ragas_dir.exists():
        print("\n### RAGAS end-to-end\n_(no result files yet)_")
        return
    print("\n### RAGAS end-to-end answer quality\n")
    for stem in ("ragas_clean", "ragas_noisy"):
        d = _load(ragas_dir / f"{stem}.json")
        if not d:
            continue
        res = d["results"]
        metrics = list(next(iter(res.values())).keys())
        print(f"**{stem}** (n={d['n_samples']})\n")
        print("| Method | " + " | ".join(metrics) + " |")
        print("|" + "---|" * (len(metrics) + 1))
        for method, m in res.items():
            print(f"| {method} | " + " | ".join(f"{m[k]:.4f}" for k in metrics) + " |")
        print()


def main() -> None:
    print("# Consolidated evaluation report\n")
    table_retrieval(
        [
            ("ViQuAD clean", RESULTS / "p1_full_viaquad.json"),
            ("ViQuAD noisy", RESULTS / "p1_full_viaquad_noisy.json"),
            ("DANGDOCAO clean", RESULTS / "p1_full_dangdocao.json"),
            ("DANGDOCAO noisy", RESULTS / "p1_full_dangdocao_noisy.json"),
        ],
        "Main baseline table (P1d, full test sets)",
    )
    table_retrieval(
        [
            ("missing_tone", RESULTS / "llm_noise_4way/dangdocao_missing_tone.json"),
            ("typo_telex", RESULTS / "llm_noise_4way/dangdocao_typo_telex.json"),
            ("informal", RESULTS / "llm_noise_4way/dangdocao_informal.json"),
            ("code_switch", RESULTS / "llm_noise_4way/dangdocao_code_switch.json"),
        ],
        "OOD LLM-noise generalization (P2.7, DANGDOCAO full)",
    )
    table_curve()
    table_restoration()
    table_ragas()


if __name__ == "__main__":
    main()
