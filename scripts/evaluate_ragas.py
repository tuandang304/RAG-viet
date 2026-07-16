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

# IMPORTANT: pyarrow must be imported BEFORE torch/faiss on Windows + CUDA
# (see scripts/evaluate_all.py for the access-violation incident this guards).
import pyarrow  # noqa: F401

import argparse
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
from tqdm import tqdm

from rag_vie.config import settings
from rag_vie.features.vietnamese import extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.generator.llm import generate
from rag_vie.retrieval.bm25 import BM25Retriever
# NOTE: DenseRetriever (which transitively imports faiss) is intentionally NOT
# imported at module load. main() imports it lazily AFTER BGE-M3 has been loaded.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_vie.retrieval.dense import DenseRetriever  # noqa: F401
    from rag_vie.retrieval.hybrid import HybridRetriever  # noqa: F401


# ── RAGAS judge setup ──────────────────────────────────────────────────────
# RAGAS 0.4.x dropped LangchainLLMWrapper / LangchainEmbeddingsWrapper for the
# "modern" InstructorBaseRagasLLM factory. We supply a pre-built OpenAI client
# pointing at FPT AI Factory (OpenAI-compatible API) — same credentials as the
# rest of the pipeline use via src/rag_vie/retrieval/embedder.py.

def _fpt_async_client() -> AsyncOpenAI:
    """Async OpenAI client pointed at FPT — required because ragas metrics
    call `agenerate()` internally on the supplied LLM/embedding clients.
    A sync `OpenAI(...)` client raises TypeError on every metric call."""
    return AsyncOpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)


def make_ragas_llm():
    return llm_factory(
        model=settings.fpt_llm_model,
        provider="openai",
        client=_fpt_async_client(),
        temperature=0.0,
        max_tokens=1024,
    )


def make_ragas_embeddings():
    return embedding_factory(
        provider="openai",
        model=settings.fpt_embedding_model,
        client=_fpt_async_client(),
    )


# ── Retrieval helpers ──────────────────────────────────────────────────────

# Three-way fusion weight triples matching evaluate_all.py METHODS.
METHODS: dict[str, tuple[float, float, float] | None] = {
    "dynamic_mlp":   None,                # weights from MLP, predicted per query
    "fixed_equal":   (1/3, 1/3, 1/3),     # uniform three-way
    "dense_only":    (1.0, 0.0, 0.0),
    "bm25_only":     (0.0, 1.0, 0.0),
    "sparse_only":   (0.0, 0.0, 1.0),
}


def retrieve_contexts(
    hybrid: "HybridRetriever",
    passages_map: dict[str, str],
    query: str,
    weights: tuple[float, ...],
    top_k: int,
) -> list[str]:
    hits = hybrid.retrieve(query, weights, settings.top_k_dense, settings.top_k_bm25, top_k)
    return [passages_map[pid] for pid, _, _ in hits if pid in passages_map]


# ── Build samples for one method ──────────────────────────────────────────

def build_samples(
    qas: list[dict],
    hybrid: "HybridRetriever",
    passages_map: dict[str, str],
    mlp: FusionMLP,
    method: str,
    fixed_w: tuple[float, ...] | None,
    top_k: int,
    bm25_vocab: set[str] | None = None,
) -> tuple[list[SingleTurnSample], list[str]]:
    """Return (samples, sample_qa_ids) — qa_ids preserved for incremental runs."""
    samples = []
    sample_ids: list[str] = []
    for qa in tqdm(qas, desc=f"  Retrieving+generating [{method}]"):
        query = qa["question"]
        ground_truth = qa["answers"][0] if qa.get("answers") else ""
        if not ground_truth:
            continue

        if fixed_w is not None:
            weights = fixed_w
        else:
            weights = mlp.predict_weights(extract_features(query, bm25_vocab=bm25_vocab))

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
        sample_ids.append(str(qa.get("id", "")))
    return samples, sample_ids


# ── Evaluate one method ────────────────────────────────────────────────────

def evaluate_method(
    samples: list[SingleTurnSample],
    ragas_llm,
    ragas_emb,
) -> tuple[dict[str, float], dict[str, list[float | None]]]:
    """Score every sample against all four RAGAS metrics, return per-metric mean.

    RAGAS 0.4.3 split metrics into modern `collections.*` classes that take
    keyword args via `.score()` rather than the old `evaluate(samples, metrics)`
    function (which now rejects collection-style instances). We call each metric
    per sample and aggregate manually — slower than the batched legacy path but
    works with the modern instructor-based LLM factory.
    """
    cp = ContextPrecisionWithReference(llm=ragas_llm)
    cr = ContextRecall(llm=ragas_llm)
    fa = Faithfulness(llm=ragas_llm)
    ar = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb)

    scores: dict[str, list[float]] = {
        "context_precision": [],
        "context_recall":    [],
        "faithfulness":      [],
        "answer_relevancy":  [],
    }

    # Each metric in ragas 0.4.x takes its own exact kwargs. Pass only what
    # the metric needs — passing extras raises TypeError.
    def _kw_for(name: str, s: SingleTurnSample) -> dict:
        if name == "context_precision":
            return dict(user_input=s.user_input, reference=s.reference,
                        retrieved_contexts=s.retrieved_contexts)
        if name == "context_recall":
            return dict(user_input=s.user_input, retrieved_contexts=s.retrieved_contexts,
                        reference=s.reference)
        if name == "faithfulness":
            return dict(user_input=s.user_input, response=s.response,
                        retrieved_contexts=s.retrieved_contexts)
        if name == "answer_relevancy":
            return dict(user_input=s.user_input, response=s.response)
        raise ValueError(name)

    # Per-sample raw scores, aligned to the samples list (None on failure).
    per_sample: dict[str, list[float | None]] = {
        "context_precision": [],
        "context_recall":    [],
        "faithfulness":      [],
        "answer_relevancy":  [],
    }

    for s in tqdm(samples, desc="  RAGAS scoring"):
        for name, metric in [
            ("context_precision", cp),
            ("context_recall",    cr),
            ("faithfulness",      fa),
            ("answer_relevancy",  ar),
        ]:
            val_to_record: float | None = None
            try:
                result = metric.score(**_kw_for(name, s))
                val = getattr(result, "value", None)
                if val is not None and not (isinstance(val, float) and val != val):
                    val_to_record = float(val)
                    scores[name].append(val_to_record)
            except Exception as e:
                print(f"    [warn] {name} failed on a sample: {type(e).__name__}: {e}", flush=True)
            per_sample[name].append(val_to_record)

    means = {
        name: round(sum(vals) / len(vals), 4) if vals else float("nan")
        for name, vals in scores.items()
    }
    return means, per_sample


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
    parser.add_argument("--methods",     default="dynamic_mlp,fixed_equal,dense_only,bm25_only,sparse_only",
                        help="Comma-separated methods to evaluate (3-way names)")
    parser.add_argument("--sparse-path", default=None,
                        help="Sparse index pkl (default: <index-dir>/sparse.pkl)")
    parser.add_argument("--no-sparse",   action="store_true",
                        help="Disable sparse retriever (run 2-way only)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--exclude-ids", default=None,
                        help="Path to a JSON list of QA ids to exclude from sampling. "
                             "Used by extend-to-N orchestrators to skip queries "
                             "already evaluated in an earlier run.")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load QAs
    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    qas = [q for q in qas if q.get("answers") and q["answers"][0]]

    # Apply exclusion BEFORE sampling so the resampled set is disjoint from prior runs.
    if args.exclude_ids:
        with open(args.exclude_ids, encoding="utf-8") as f:
            exclude = set(json.load(f))
        before = len(qas)
        qas = [q for q in qas if str(q.get("id", "")) not in exclude]
        print(f"Excluded {before - len(qas)} previously-evaluated qa_ids "
              f"(from {args.exclude_ids}); {len(qas)} remain.")

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

    # Load sparse retriever FIRST (PyTorch / CUDA must initialise before FAISS;
    # otherwise FAISS' MKL/OMP runtime collides and the process segfaults).
    sparse = None
    if not args.no_sparse:
        sparse_path = args.sparse_path or str(Path(args.index_dir) / "sparse.pkl")
        if Path(sparse_path).exists():
            from rag_vie.retrieval.sparse import SparseRetriever, _encode_sparse
            sparse = SparseRetriever.load(sparse_path)
            print(f"Sparse index loaded from {sparse_path}", flush=True)
            _ = _encode_sparse(["warmup"])
            print("BGE-M3 warmed up", flush=True)
        else:
            print(f"NOTE: sparse.pkl not found at {sparse_path} — running 2-way fallback.")

    # Lazy faiss / dense import — must come AFTER BGE-M3 init.
    from rag_vie.retrieval.dense import DenseRetriever
    from rag_vie.retrieval.hybrid import HybridRetriever

    dense  = DenseRetriever.load(args.index_dir)
    bm25   = BM25Retriever.load(str(Path(args.index_dir) / "bm25.pkl"))
    bm25_vocab = set(bm25._bm25.idf.keys())
    hybrid = HybridRetriever(dense, bm25, sparse=sparse)
    mlp    = FusionMLP.load(args.mlp_path)

    # RAGAS judge
    print("Initializing RAGAS judge (Qwen3-32B via FPT)…")
    ragas_llm = make_ragas_llm()
    ragas_emb = make_ragas_embeddings()

    # Evaluate each method
    selected_methods = [m.strip() for m in args.methods.split(",")]
    all_results: dict[str, dict] = {}

    per_sample_all: dict[str, dict] = {}

    for method in selected_methods:
        fixed_w = METHODS.get(method)
        use_mlp = (fixed_w is None and method == "dynamic_mlp")
        if method not in METHODS:
            print(f"Unknown method '{method}', skipping")
            continue

        print(f"\n[{method}]")
        samples, sample_ids = build_samples(
            qas, hybrid, passages_map, mlp,
            method=method,
            fixed_w=fixed_w if not use_mlp else None,
            top_k=args.top_k,
            bm25_vocab=bm25_vocab,
        )
        if not samples:
            print("  No valid samples, skipping")
            continue

        print(f"  Running RAGAS on {len(samples)} samples…")
        metrics, per_sample = evaluate_method(samples, ragas_llm, ragas_emb)
        all_results[method] = metrics
        per_sample_all[method] = {"qa_ids": sample_ids, "scores": per_sample}
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
            json.dump({
                "results":    all_results,
                "n_samples":  len(qas),
                "qas_path":   args.qas_path,
                "seed":       args.seed,
                "per_sample": per_sample_all,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
