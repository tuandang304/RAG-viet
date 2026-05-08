"""End-to-end RAG evaluation using RAGAS metrics with Qwen3-32B as judge.

Metrics computed per retrieval method:
  - Context Precision   (LLM judges relevance of each retrieved chunk)
  - Context Recall      (LLM judges ground-truth coverage by retrieved chunks)
  - Faithfulness        (LLM judges if answer statements are grounded in context)
  - Answer Relevancy    (embedding similarity: question ↔ answer)

All LLM calls go to FPT AI Factory (OpenAI-compatible), using Qwen3-32B.
Evaluation runs on a small sample (default 50) to keep API costs manageable.

Usage:
    uv run python scripts/evaluate_ragas.py \\
        --qas-path data/processed/viaquad_dev.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --n-samples 50 \\
        --output results/ragas_results.json

    # Diacritic robustness test:
    uv run python scripts/evaluate_ragas.py \\
        --qas-path data/processed/viaquad_dev_noisy.jsonl \\
        --index-dir indexes/viaquad \\
        --mlp-path checkpoints/fusion_mlp_aug.pt \\
        --n-samples 50 \\
        --output results/ragas_noisy.json
"""

import argparse
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.vietnamese import extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.generator.llm import generate
from rag_vie.retrieval.bm25 import BM25Retriever
from rag_vie.retrieval.dense import DenseRetriever
from rag_vie.retrieval.hybrid import HybridRetriever


# ── RAGAS judge setup ──────────────────────────────────────────────────────

def make_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatOpenAI(
        model=settings.fpt_llm_model,
        base_url=settings.fpt_base_url,
        api_key=settings.fpt_api_key,
        temperature=0.0,
        max_tokens=1024,
    )
    return LangchainLLMWrapper(llm)


def make_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    emb = OpenAIEmbeddings(
        model=settings.fpt_embedding_model,
        base_url=settings.fpt_base_url,
        api_key=settings.fpt_api_key,
    )
    return LangchainEmbeddingsWrapper(emb)


# ── Retrieval helpers ──────────────────────────────────────────────────────

METHODS = {
    "dynamic_mlp":    None,           # weights from MLP, set after loading
    "fixed_0.5_0.5":  (0.5, 0.5),
    "dense_only":      (1.0, 0.0),
    "bm25_only":       (0.0, 1.0),
}


def retrieve_contexts(
    hybrid: HybridRetriever,
    passages_map: dict[str, str],
    query: str,
    weights: tuple[float, float],
    top_k: int,
) -> list[str]:
    hits = hybrid.retrieve(query, weights, settings.top_k_dense, settings.top_k_bm25, top_k)
    return [passages_map[pid] for pid, _, _ in hits if pid in passages_map]


# ── Build samples for one method ──────────────────────────────────────────

def build_samples(
    qas: list[dict],
    hybrid: HybridRetriever,
    passages_map: dict[str, str],
    mlp: FusionMLP,
    method: str,
    fixed_w: tuple[float, float] | None,
    top_k: int,
) -> list[SingleTurnSample]:
    samples = []
    for qa in tqdm(qas, desc=f"  Retrieving+generating [{method}]"):
        query = qa["question"]
        ground_truth = qa["answers"][0] if qa.get("answers") else ""
        if not ground_truth:
            continue

        if fixed_w is not None:
            weights = fixed_w
        else:
            weights = mlp.predict_weights(extract_features(query))

        contexts = retrieve_contexts(hybrid, passages_map, query, weights, top_k)
        if not contexts:
            continue

        answer = generate(query, contexts[:5])  # top-5 for generation

        samples.append(SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))
    return samples


# ── Evaluate one method ────────────────────────────────────────────────────

def evaluate_method(
    samples: list[SingleTurnSample],
    ragas_llm: LangchainLLMWrapper,
    ragas_emb: LangchainEmbeddingsWrapper,
) -> dict[str, float]:
    metrics = [
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]
    result = evaluate(samples, metrics=metrics)
    return {k: round(float(v), 4) for k, v in result.items()}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path",    required=True)
    parser.add_argument("--index-dir",   default="indexes/viaquad")
    parser.add_argument("--passages-path", default=None,
                        help="Passages JSONL (default: inferred from index-dir name)")
    parser.add_argument("--mlp-path",    required=True)
    parser.add_argument("--n-samples",   type=int, default=50)
    parser.add_argument("--top-k",       type=int, default=10)
    parser.add_argument("--methods",     default="dynamic_mlp,fixed_0.5_0.5,dense_only,bm25_only",
                        help="Comma-separated methods to evaluate")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load QAs
    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(l) for l in f if l.strip()]
    qas = [q for q in qas if q.get("answers") and q["answers"][0]]
    if args.n_samples < len(qas):
        qas = random.sample(qas, args.n_samples)
    print(f"Evaluating {len(qas)} samples with RAGAS")

    # Infer passages path
    if args.passages_path:
        passages_path = args.passages_path
    else:
        domain = Path(args.index_dir).name  # e.g. "viaquad"
        passages_path = f"data/processed/{domain}_passages.jsonl"
    print(f"Passages: {passages_path}")

    # Load passages lookup
    passages_map: dict[str, str] = {}
    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passages_map[obj["id"]] = obj["passage"]

    # Load retrievers
    dense  = DenseRetriever.load(args.index_dir)
    bm25   = BM25Retriever.load(str(Path(args.index_dir) / "bm25.pkl"))
    hybrid = HybridRetriever(dense, bm25)
    mlp    = FusionMLP.load(args.mlp_path)

    # RAGAS judge
    print("Initializing RAGAS judge (Qwen3-32B via FPT)…")
    ragas_llm = make_ragas_llm()
    ragas_emb = make_ragas_embeddings()

    # Evaluate each method
    selected_methods = [m.strip() for m in args.methods.split(",")]
    all_results: dict[str, dict] = {}

    for method in selected_methods:
        fixed_w = METHODS.get(method)
        use_mlp = (fixed_w is None and method == "dynamic_mlp")
        if method not in METHODS:
            print(f"Unknown method '{method}', skipping")
            continue

        print(f"\n[{method}]")
        samples = build_samples(
            qas, hybrid, passages_map, mlp,
            method=method,
            fixed_w=fixed_w if not use_mlp else None,
            top_k=args.top_k,
        )
        if not samples:
            print("  No valid samples, skipping")
            continue

        print(f"  Running RAGAS on {len(samples)} samples…")
        metrics = evaluate_method(samples, ragas_llm, ragas_emb)
        all_results[method] = metrics
        print(f"  {metrics}")

    # Print summary table
    if all_results:
        metric_names = list(next(iter(all_results.values())).keys())
        header = f"{'Method':<20}" + "".join(f"  {m[:16]:>16}" for m in metric_names)
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for method, m in all_results.items():
            row = f"{method:<20}" + "".join(f"  {m[k]:>16.4f}" for k in metric_names)
            print(row)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": all_results, "n_samples": len(qas),
                       "qas_path": args.qas_path}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
