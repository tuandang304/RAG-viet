"""Parse the benchmark JSON results and print them as markdown tables exactly like the paper's format.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results/llm_noise_benchmark")
FILES = {
    "Clean": "clean.json",
    "Rule-based Noisy (Missing Diacritics 100%)": "rule_based_noisy.json",
    "LLM Noisy (Natural Missing Tones)": "llm_missing_tone.json",
    "LLM Noisy (Telex/VNI Typo)": "llm_typo_telex.json",
    "LLM Noisy (Informal speech/Teen code)": "llm_informal.json",
    "LLM Noisy (Code-Switching)": "llm_code_switch.json",
}

METHODS_MAP = {
    "bm25": "BM25 only",
    "dense": "Dense only",
    "sparse": "Sparse only (BGE-M3)",
    "dense_bm25": "Dense + BM25 (0.5/0.5)",
    "fixed_equal": "Fixed-equal three-way (1/3,1/3,1/3)",
    "mlp": "**Dynamic MLP (soft label, three-way)**",
}

METRICS = ["NDCG@10", "MRR@10", "MAP@10", "Recall@10", "Recall@100", "Hit@1"]

def main():
    print("# EVALUATION COMPARISONS (PAPER FORMAT)\n")
    for title, filename in FILES.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"### {title}\n*Result file {filename} not found.*\n")
            continue
            
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"### {title} (n = {data.get('n_queries', 4315)} queries)")
        print()
        print("| Method | NDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@100 | Hit@1 |")
        print("|---|---|---|---|---|---|---|")
        
        methods_data = data.get("methods", {})
        # Order them as in the paper
        for key in ["bm25", "dense", "sparse", "dense_bm25", "fixed_equal", "mlp"]:
            m_name = METHODS_MAP[key]
            m_val = methods_data.get(key, {})
            row = [m_name]
            for metric in METRICS:
                row.append(f"{m_val.get(metric, 0.0):.4f}")
            print("| " + " | ".join(row) + " |")
        print("\n")

if __name__ == "__main__":
    main()
