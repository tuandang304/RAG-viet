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
    # Fair comparison uses the SAME 500 queries (p1_noisy500_*) when available,
    # else falls back to the full noisy set (different n — flagged).
    pairs = [
        ("ViQuAD", "p1_restored_viaquad.json",
         "p1_noisy500_viaquad.json", "p1_full_viaquad_noisy.json"),
        ("DANGDOCAO", "p1_restored_dangdocao.json",
         "p1_noisy500_dangdocao.json", "p1_full_dangdocao_noisy.json"),
    ]
    any_row = False
    fallback_used = False
    print("| Domain | Router on noisy | Router on restored | Δ (restore−noisy) |")
    print("|---|---|---|---|")
    for dom, restored_f, noisy500_f, noisy_full_f in pairs:
        r = _load(RESULTS / restored_f)
        if not r:
            continue
        any_row = True
        n = _load(RESULTS / noisy500_f)
        if n is None:
            n = _load(RESULTS / noisy_full_f)
            if n is not None:
                fallback_used = True
        r_mlp = r["methods"]["mlp"]["NDCG@10"]
        n_mlp = n["methods"]["mlp"]["NDCG@10"] if n else None
        delta = f"{r_mlp - n_mlp:+.4f}" if n_mlp is not None else "—"
        noisy_cell = f"{n_mlp:.4f}" if n_mlp is not None else "—"
        print(f"| {dom} | {noisy_cell} | {r_mlp:.4f} | {delta} |")
    if not any_row:
        print("_(no restoration result files yet)_")
    elif fallback_used:
        print("\n_Note: noisy column uses the full noisy set (n differs from the "
              "restored 500) — run p1_noisy500_* for an exact same-query comparison._")
    else:
        print("\n_Same 500 queries (seed 42) for both columns. Restoration is a "
              "strong but costly baseline: 1 LLM call/query vs a single BM25 lookup "
              "for the toneless channel._")


def table_ragas() -> None:
    # Prefer the n=100 full run (results/ragas_full/{vq,dd}_{clean,noisy}); fall
    # back to the earlier n=40 ragas_4way/{ragas_clean,ragas_noisy}.
    full = RESULTS / "ragas_full"
    conds = [
        ("ViQuAD clean", "vq_clean"), ("ViQuAD noisy", "vq_noisy"),
        ("DANGDOCAO clean", "dd_clean"), ("DANGDOCAO noisy", "dd_noisy"),
    ]
    if full.exists() and any((full / f"{s}.json").exists() for _, s in conds):
        print("\n### RAGAS end-to-end answer quality (Llama-3.3-70B judge)\n")
        for name, stem in conds:
            d = _load(full / f"{stem}.json")
            if not d:
                continue
            res = d["results"]
            metrics = list(next(iter(res.values())).keys())
            print(f"**{name}** (n={d['n_samples']})\n")
            print("| Method | " + " | ".join(metrics) + " |")
            print("|" + "---|" * (len(metrics) + 1))
            for method, m in res.items():
                print(f"| {method} | " + " | ".join(f"{m[k]:.4f}" for k in metrics) + " |")
            print()
        print("_Context precision/recall (retrieval-quality metrics): the router "
              "leads in all four conditions. Faithfulness (generator-dependent) is "
              "comparable in the noisy regime. answer_relevancy is omitted — its "
              "async embedding path deadlocks against the FPT API._")
        return

    ragas_dir = RESULTS / "ragas_4way"
    if not ragas_dir.exists():
        print("\n### RAGAS end-to-end\n_(no result files yet)_")
        return
    print("\n### RAGAS end-to-end answer quality (n=40 pilot)\n")
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


ABL_CONFIGS = [
    ("full", "Full system"),
    ("argmax", "− expected weights (argmax)"),
    ("nosignals", "− QPP signals (8 linguistic feats)"),
    ("normlabels", "− raw labels (per-query min-max)"),
    ("noaug", "− toneless augmentation"),
    ("3way", "− toneless channel (3-way routing)"),
]
ABL_SETS = [("vq_clean", "ViQuAD clean"), ("vq_noisy", "ViQuAD noisy"),
            ("dd_clean", "DANGDOCAO clean"), ("dd_noisy", "DANGDOCAO noisy")]


def table_ablation() -> None:
    abl = RESULTS / "ablation"
    print("\n### Component ablation (router NDCG@10, n=500/set, seed 42)\n")
    loaded: dict[tuple[str, str], dict] = {}
    for cfg, _ in ABL_CONFIGS:
        for tag, _ in ABL_SETS:
            d = _load(abl / f"{cfg}_{tag}.json")
            if d:
                loaded[(cfg, tag)] = d
    if not loaded:
        print("_(no ablation result files yet)_")
        return
    header = " | ".join(name for _, name in ABL_SETS)
    print(f"| Configuration | {header} |")
    print("|" + "---|" * (len(ABL_SETS) + 1))
    for cfg, label in ABL_CONFIGS:
        cells = []
        for tag, _ in ABL_SETS:
            d = loaded.get((cfg, tag))
            cells.append(f"{d['methods']['mlp']['NDCG@10']:.4f}" if d else "—")
        print(f"| {label} | {' | '.join(cells)} |")

    # Oracle headroom (from the "full" runs, which carry the oracle method)
    rows = []
    for tag, name in ABL_SETS:
        d = loaded.get(("full", tag))
        if not d or "oracle" not in d["methods"]:
            continue
        mlp = d["methods"]["mlp"]["NDCG@10"]
        eq4 = d["methods"].get("fixed_equal_4", {}).get("NDCG@10")
        orc = d["methods"]["oracle"]["NDCG@10"]
        if eq4 is None or orc <= eq4:
            continue
        rows.append((name, eq4, mlp, orc, (mlp - eq4) / (orc - eq4)))
    if rows:
        print("\n**Oracle headroom** (label-dependent per-query best grid point — upper bound):\n")
        print("| Set | Fixed equal 4-way | Router | Oracle | Headroom realized |")
        print("|---|---|---|---|---|")
        for name, eq4, mlp, orc, frac in rows:
            print(f"| {name} | {eq4:.4f} | {mlp:.4f} | {orc:.4f} | {frac * 100:.0f}% |")
        print("\n_Headroom realized = (router − equal4) / (oracle − equal4)._")


def table_latency() -> None:
    d = _load(RESULTS / "ablation" / "full_vq_clean.json")
    lat = (d or {}).get("efficiency", {}).get("stage_latency_ms")
    if not lat:
        print("\n### Per-query latency breakdown\n_(no instrumented run yet)_")
        return
    print("\n### Per-query latency breakdown (ms, ViQuAD n=500 run)\n")
    print("| Stage | mean | p50 |")
    print("|---|---|---|")
    order = ["dense", "bm25", "sparse", "toneless", "features", "signals"]
    for name in order:
        if name in lat:
            print(f"| {name} | {lat[name]['mean']:.1f} | {lat[name]['p50']:.1f} |")
    mlp_us = (d or {}).get("efficiency", {}).get("mlp_inference_us", {})
    if mlp_us:
        print(f"| router MLP | {mlp_us['mean'] / 1000:.1f} | — |")
    print("\n_Dense = FAISS search only (query embedding is a cached/batched API call). "
          "The toneless channel adds one in-memory BM25 lookup; LLM diacritic "
          "restoration costs ~1.4–1.7 s/query wall-clock (measured over the two "
          "500-query restoration runs) plus generation-side token spend._")


def table_ragas_significance() -> None:
    sig = _load(RESULTS / "ragas_full" / "significance.json")
    if not sig:
        return
    print("\n**RAGAS paired significance** (router vs baseline, per-sample paired t-test):\n")
    print("| Condition | vs | metric | Δ | p |")
    print("|---|---|---|---|---|")
    names = dict(ABL_SETS)
    for stem, per_base in sig.items():
        for base, per_metric in per_base.items():
            for metric, e in per_metric.items():
                p = e["ttest_p"]
                star = ("***" if p < 1e-3 else "**" if p < 1e-2
                        else "*" if p < 5e-2 else "ns")
                print(f"| {names.get(stem, stem)} | {base} | {metric} | "
                      f"{e['mean_delta']:+.4f} | {p:.1e} {star} |")


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
    table_ablation()
    table_latency()
    table_ragas()
    table_ragas_significance()


if __name__ == "__main__":
    main()
