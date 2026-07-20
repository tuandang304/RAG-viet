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
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

if hasattr(__import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI, OpenAI
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
from rag_vie.features.retrieval_signals import (
    SIGNAL_NAMES,
    SIGNAL_NAMES_4WAY,
    extract_retrieval_signals,
)
from rag_vie.features.vietnamese import FEATURE_NAMES, extract_features
from rag_vie.fusion.mlp import FusionMLP
from rag_vie.generator.llm import generate
from rag_vie.retrieval.bm25 import BM25Retriever
# NOTE: DenseRetriever (which transitively imports faiss) is intentionally NOT
# imported at module load. main() imports it lazily AFTER BGE-M3 has been loaded.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rag_vie.retrieval.dense import DenseRetriever  # noqa: F401
    from rag_vie.retrieval.hybrid import HybridRetriever  # noqa: F401


# ── LLM provider setup ─────────────────────────────────────────────────────
# Two providers supported: FPT AI Factory (default for production / main.py)
# and MiniMax M3 (used for RAGAS pilot runs since the FPT key is exhausted).
# Embeddings always go through FPT — MiniMax has no Vietnamese embedding.
#
# RAGAS 0.4.x dropped LangchainLLMWrapper / LangchainEmbeddingsWrapper for the
# "modern" InstructorBaseRagasLLM factory. We supply a pre-built OpenAI client
# — RAGAS metrics call `agenerate()` internally so a sync `OpenAI(...)` client
# raises TypeError on every metric call.

# Resolved at runtime by main() based on --llm-provider.
_LLM_BASE_URL: str = ""
_LLM_API_KEY: str = ""
_LLM_MODEL: str = ""
_LLM_DISABLE_THINKING: bool = True
_LLM_CLIENT_OVERRIDES: dict = {}


def _resolve_minimax_key() -> str:
    """Resolve MiniMax key at call time. Order: --minimax-api-key flag >
    settings.minimax_api_key > MINIMAX_API_KEY env > ANTHROPIC_AUTH_TOKEN env."""
    if _LLM_API_KEY:
        return _LLM_API_KEY
    if settings.minimax_api_key:
        return settings.minimax_api_key
    return os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")


def _fpt_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)


def _minimax_async_client() -> AsyncOpenAI:
    key = _resolve_minimax_key()
    if not key:
        raise RuntimeError(
            "MiniMax API key not found. Set MINIMAX_API_KEY (or ANTHROPIC_AUTH_TOKEN) "
            "in your environment, or pass --minimax-api-key, or add it to .env."
        )
    return AsyncOpenAI(api_key=key, base_url=_LLM_BASE_URL or settings.minimax_base_url)


def _minimax_sync_client() -> OpenAI:
    key = _resolve_minimax_key()
    if not key:
        raise RuntimeError(
            "MiniMax API key not found. Set MINIMAX_API_KEY (or ANTHROPIC_AUTH_TOKEN) "
            "in your environment, or pass --minimax-api-key, or add it to .env."
        )
    return OpenAI(api_key=key, base_url=_LLM_BASE_URL or settings.minimax_base_url)


def make_ragas_llm():
    """Build the RAGAS LLM judge for the selected provider."""
    if _LLM_BASE_URL:  # explicit MiniMax (or override)
        # llm_factory merges **kwargs into model_args, which is splatted into
        # `chat.completions.create(**provider_kwargs)` — so extra_body flows
        # through and the OpenAI client merges it into the request body sent
        # to MiniMax. Without this, M3 always emits a `<think>` block before
        # the structured judge answer, wasting ~10x tokens and confusing
        # instructor's Pydantic parser.
        extra = {"extra_body": {"thinking": {"type": "disabled"}}} if _LLM_DISABLE_THINKING else {}
        return llm_factory(
            model=_LLM_MODEL or settings.minimax_llm_model,
            provider="openai",
            client=_minimax_async_client(),
            temperature=0.0,
            max_tokens=1024,
            **extra,
        )
    # Default: FPT (back-compat with earlier RAGAS runs)
    return llm_factory(
        model=_LLM_MODEL or settings.fpt_llm_model,
        provider="openai",
        client=_fpt_async_client(),
        temperature=0.0,
        max_tokens=1024,
    )


def make_ragas_embeddings():
    """Embeddings always go through FPT (MiniMax has no Vietnamese embedding)."""
    return embedding_factory(
        provider="openai",
        model=settings.fpt_embedding_model,
        client=_fpt_async_client(),
    )


# ── MiniMax answer generator (used when --llm-provider minimax) ─────────────

_GEN_SYSTEM = (
    "Bạn là trợ lý AI hữu ích. Dựa vào các đoạn văn được cung cấp, "
    "hãy trả lời câu hỏi bằng tiếng Việt một cách chính xác và súc tích. "
    "Nếu không tìm thấy thông tin trong đoạn văn, hãy nói rõ điều đó."
)


def _generate_answer_minimax(query: str, passages: list[str], max_tokens: int = 512) -> str:
    """Generate an answer via MiniMax-M3 (OpenAI-compatible). Retries with
    exponential backoff on transient 5xx/429 (MiniMax proxies through CDN — same
    transient pattern as FPT in src/rag_vie/retrieval/embedder.py)."""
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user_message = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"
    client = _minimax_sync_client()
    kwargs: dict = {"model": _LLM_MODEL or settings.minimax_llm_model,
                    "messages": [{"role": "system", "content": _GEN_SYSTEM},
                                 {"role": "user", "content": user_message}],
                    "max_tokens": max_tokens, "temperature": 0.1}
    if _LLM_DISABLE_THINKING:
        # M3 is a reasoning model; `thinking` is a non-OpenAI-standard field that
        # the MiniMax server merges at the top of the request body. OpenAI's
        # client rejects unknown kwargs, so it has to go through `extra_body`.
        # Without this the response always begins with `<think>...</think>` and
        # wastes ~10x completion tokens.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kwargs)
            content = r.choices[0].message.content or ""
            # Belt-and-braces: strip a leading <think> block if the server
            # ignores the disable flag (catches regressions silently).
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            return content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt
            print(f"    [minimax gen] {type(exc).__name__} — retry {attempt + 1}/5 in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise last_exc if last_exc else RuntimeError("minimax gen unreachable")


# ── Retrieval helpers ──────────────────────────────────────────────────────

# Three-way fusion weight triples matching evaluate_all.py METHODS.
def build_methods(has_toneless: bool) -> dict[str, tuple[float, ...] | None]:
    methods: dict[str, tuple[float, ...] | None] = {
        "dynamic_mlp":   None,                     # weights from MLP, predicted per query
        "fixed_equal":   (1/3, 1/3, 1/3, 0.0),     # uniform three-way
        "dense_only":    (1.0, 0.0, 0.0, 0.0),
        "bm25_only":     (0.0, 1.0, 0.0, 0.0),
        "sparse_only":   (0.0, 0.0, 1.0, 0.0),
    }
    if has_toneless:
        methods["toneless_only"] = (0.0, 0.0, 0.0, 1.0)
        methods["fixed_equal_4"] = (0.25, 0.25, 0.25, 0.25)
    return methods


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
    # Mirror evaluate_all/pipeline: retrieve all channels first, then (for the
    # dynamic method) build the full feature vector the checkpoint expects.
    n_base = len(FEATURE_NAMES)
    use_signals_4way = mlp.input_dim == n_base + len(SIGNAL_NAMES_4WAY)
    use_signals = use_signals_4way or mlp.input_dim == n_base + len(SIGNAL_NAMES)

    samples = []
    sample_ids: list[str] = []
    for qa in tqdm(qas, desc=f"  Retrieving+generating [{method}]"):
        query = qa["question"]
        ground_truth = qa["answers"][0] if qa.get("answers") else ""
        if not ground_truth:
            continue

        # Retrieval + generation each hit the FPT API; a transient failure that
        # survives the embedder's own retries must skip this sample, not crash
        # the whole (multi-hour) run.
        try:
            hits = hybrid.search_all(query, settings.top_k_dense, settings.top_k_bm25)

            if fixed_w is not None:
                weights = fixed_w
            else:
                features = extract_features(query, bm25_vocab=bm25_vocab)
                if use_signals:
                    signals = extract_retrieval_signals(
                        {pid: s for pid, _, s in hits["dense"]},
                        {pid: s for pid, _, s in hits["bm25"]},
                        {pid: s for pid, _, s in hits["sparse"]},
                        toneless_scores=(
                            {pid: s for pid, _, s in hits["toneless"]} if use_signals_4way else None
                        ),
                    )
                    features = np.concatenate([features, signals])
                weights = mlp.predict_weights(features)

            fused = hybrid.fuse(hits, weights, top_k)
            contexts = [passages_map[pid] for pid, _, _ in fused if pid in passages_map]
            if not contexts:
                continue

            answer = (_generate_answer_minimax(query, contexts[:5])
                      if _LLM_BASE_URL else generate(query, contexts[:5]))  # top-5 for generation
        except Exception as e:
            print(f"    [warn] sample skipped ({type(e).__name__}): {e}", flush=True)
            continue

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
    metric_names: list[str],
) -> tuple[dict[str, float], dict[str, list[float | None]]]:
    """Score every sample against the selected RAGAS metrics, return per-metric mean.

    RAGAS 0.4.3 split metrics into modern `collections.*` classes that take
    keyword args via `.score()` rather than the old `evaluate(samples, metrics)`
    function (which now rejects collection-style instances). We call each metric
    per sample and aggregate manually — slower than the batched legacy path but
    works with the modern instructor-based LLM factory.

    NOTE: answer_relevancy routes through ragas' async embedding wrapper, which
    deadlocks against the FPT embedding client (every call hits the per-call
    timeout). It is off by default; the three LLM-judge metrics
    (context precision/recall, faithfulness) are the retrieval-relevant ones.
    """
    _factory = {
        "context_precision": lambda: ContextPrecisionWithReference(llm=ragas_llm),
        "context_recall":    lambda: ContextRecall(llm=ragas_llm),
        "faithfulness":      lambda: Faithfulness(llm=ragas_llm),
        "answer_relevancy":  lambda: AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    }
    metric_objs = [(n, _factory[n]()) for n in metric_names]
    scores: dict[str, list[float]] = {n: [] for n in metric_names}

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
    per_sample: dict[str, list[float | None]] = {n: [] for n in metric_names}

    # A single ragas metric.score() call can stall indefinitely (network hiccup
    # or the internal instructor loop retrying a malformed structured output).
    #
    # Concurrency model: ONE event loop per sample, run all metrics as
    # concurrent asyncio tasks via asyncio.gather on each metric's ascore().
    # Calling metric.score() (which wraps asyncio.run(ascore())) from multiple
    # threads DEADLOCKS because every thread tries to spin up its own event
    # loop while sharing the AsyncOpenAI client — the loop never closes cleanly.
    # Using ascore() directly in a single loop is the only safe concurrency path.
    import asyncio

    async def _score_all(s: SingleTurnSample) -> dict[str, float | None]:
        tasks = {name: asyncio.create_task(metric.ascore(**_kw_for(name, s)),
                                           name=f"{name}-sample")
                 for name, metric in metric_objs}
        results: dict[str, float | None] = {}
        for name, task in tasks.items():
            try:
                r = await asyncio.wait_for(task, timeout=180.0)
                val = getattr(r, "value", None)
                if val is not None and not (isinstance(val, float) and val != val):
                    results[name] = float(val)
                else:
                    results[name] = None
            except asyncio.TimeoutError:
                print(f"    [warn] {name} timed out on a sample — skipped", flush=True)
                results[name] = None
            except Exception as e:  # noqa: BLE001
                print(f"    [warn] {name} failed on a sample: {type(e).__name__}: {e}",
                      flush=True)
                results[name] = None
        return results

    for s in tqdm(samples, desc="  RAGAS scoring"):
        try:
            per_metric = asyncio.run(_score_all(s))
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] sample scoring crashed: {type(e).__name__}: {e}",
                  flush=True)
            per_metric = {n: None for n, _ in metric_objs}
        for name, _ in metric_objs:
            v = per_metric.get(name)
            if v is not None:
                scores[name].append(v)
            per_sample[name].append(v)

    means = {
        name: round(sum(vals) / len(vals), 4) if vals else float("nan")
        for name, vals in scores.items()
    }
    return means, per_sample


# ── Main ──────────────────────────────────────────────────────────────────

def _save_partial(path: Path, payload: dict) -> None:
    """Atomic JSON write so a crash mid-method doesn't corrupt the result file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path",    required=True)
    parser.add_argument("--index-dir",   default="indexes/viaquad")
    parser.add_argument("--passages-path", default=None,
                        help="Passages JSONL (default: inferred from index-dir name)")
    parser.add_argument("--mlp-path",    required=True)
    parser.add_argument("--n-samples",   type=int, default=0,
                        help="Number of QA samples to evaluate (0 = use ALL). "
                             "Defaults to 0 (full eval); was 50 in the original pilot.")
    parser.add_argument("--top-k",       type=int, default=10)
    parser.add_argument("--methods",     default="dynamic_mlp,fixed_equal_4,toneless_only,dense_only",
                        help="Comma-separated methods to evaluate")
    parser.add_argument("--sparse-path", default=None,
                        help="Sparse index pkl (default: <index-dir>/sparse.pkl)")
    parser.add_argument("--no-sparse",   action="store_true",
                        help="Disable sparse retriever (run 2-way only)")
    parser.add_argument("--no-toneless", action="store_true",
                        help="Skip the toneless BM25 channel even when bm25_toneless.pkl exists")
    parser.add_argument("--llm-provider", default="minimax",
                        choices=["fpt", "minimax"],
                        help="LLM provider for answer generation + RAGAS judge. "
                             "Embeddings always go through FPT.")
    parser.add_argument("--judge-model", default=None,
                        help="Override the LLM model name. Default: MiniMax-M3 "
                             "(when --llm-provider minimax) or settings.fpt_llm_model.")
    parser.add_argument("--minimax-base-url", default=None,
                        help="Override MiniMax base URL (default: settings.minimax_base_url)")
    parser.add_argument("--minimax-api-key",  default=None,
                        help="Override MiniMax API key (default: settings.minimax_api_key "
                             "→ ANTHROPIC_AUTH_TOKEN env)")
    parser.add_argument("--disable-thinking", action="store_true", default=True,
                        help="Send thinking={type:disabled} to suppress <think> blocks "
                             "(default ON for MiniMax M3 — reasoning model).")
    parser.add_argument("--no-disable-thinking", dest="disable_thinking",
                        action="store_false")
    parser.add_argument("--ragas-metrics",
                        default="context_precision,context_recall,faithfulness",
                        help="Comma-separated RAGAS metrics. answer_relevancy is excluded "
                             "by default (its async embedding wrapper deadlocks on FPT).")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--exclude-ids", default=None,
                        help="Path to a JSON list of QA ids to exclude from sampling. "
                             "Used by extend-to-N orchestrators to skip queries "
                             "already evaluated in an earlier run.")
    parser.add_argument("--resume-from", default=None,
                        help="Path to a partial result JSON; methods listed in its "
                             "completed_methods field are skipped. Combined with "
                             "--output, partial progress is saved after each method.")
    args = parser.parse_args()

    # ── Resolve LLM provider + model ────────────────────────────────────────
    global _LLM_BASE_URL, _LLM_API_KEY, _LLM_MODEL, _LLM_DISABLE_THINKING
    if args.llm_provider == "minimax":
        _LLM_BASE_URL = args.minimax_base_url or settings.minimax_base_url
        _LLM_API_KEY = args.minimax_api_key or settings.minimax_api_key
        _LLM_MODEL = args.judge_model or settings.minimax_llm_model
        _LLM_DISABLE_THINKING = args.disable_thinking
    else:  # fpt
        _LLM_BASE_URL = ""           # signals FPT path
        _LLM_API_KEY = ""
        _LLM_MODEL = args.judge_model or settings.fpt_llm_model
        settings.fpt_llm_model = _LLM_MODEL
        _LLM_DISABLE_THINKING = False  # FPT models are non-reasoning

    provider_label = (f"{args.llm_provider}/{_LLM_MODEL} @ {_LLM_BASE_URL}"
                      if _LLM_BASE_URL else f"{args.llm_provider}/{_LLM_MODEL} via FPT")
    print(f"LLM provider: {provider_label} | thinking={'off' if _LLM_DISABLE_THINKING else 'on'}")

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

    if args.n_samples and args.n_samples < len(qas):
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
    bm25_vocab = bm25.vocab

    toneless = None
    if not args.no_toneless:
        toneless_path = Path(args.index_dir) / "bm25_toneless.pkl"
        if toneless_path.exists():
            toneless = BM25Retriever.load(toneless_path)
            print(f"Toneless index loaded from {toneless_path}", flush=True)

    hybrid = HybridRetriever(dense, bm25, sparse=sparse, toneless=toneless)
    mlp    = FusionMLP.load(args.mlp_path)
    methods = build_methods(has_toneless=toneless is not None)

    # RAGAS judge
    ragas_metric_names = [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]
    print(f"Initializing RAGAS judge ({_LLM_MODEL} via {args.llm_provider}); "
          f"metrics={ragas_metric_names}")
    ragas_llm = make_ragas_llm()
    ragas_emb = make_ragas_embeddings()

    # ── Resume from partial file ────────────────────────────────────────────
    out_path = Path(args.output) if args.output else None
    completed: list[str] = []
    all_results: dict[str, dict] = {}
    per_sample_all: dict[str, dict] = {}
    base_payload: dict = {}
    if args.resume_from and Path(args.resume_from).exists():
        prev = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        completed = list(prev.get("completed_methods", []))
        all_results = dict(prev.get("results", {}))
        per_sample_all = dict(prev.get("per_sample", {}))
        base_payload = {k: v for k, v in prev.items()
                        if k not in {"results", "per_sample", "completed_methods"}}
        print(f"Resuming: {len(completed)} method(s) already done "
              f"({', '.join(completed)})")

    # Evaluate each method
    selected_methods = [m.strip() for m in args.methods.split(",")]
    for method in selected_methods:
        if method not in methods:
            print(f"Unknown method '{method}', skipping")
            continue
        if method in completed:
            print(f"\n[{method}] already done — skipping (resume)")
            continue

        fixed_w = methods.get(method)
        use_mlp = (fixed_w is None and method == "dynamic_mlp")

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
        metrics, per_sample = evaluate_method(samples, ragas_llm, ragas_emb, ragas_metric_names)
        all_results[method] = metrics
        per_sample_all[method] = {"qa_ids": sample_ids, "scores": per_sample}
        completed.append(method)
        print(f"  {metrics}")

        # ── Incremental save after each method ──────────────────────────────
        if out_path:
            payload = {**base_payload,
                       "results":    all_results,
                       "n_samples":  len(qas),
                       "qas_path":   args.qas_path,
                       "seed":       args.seed,
                       "per_sample": per_sample_all,
                       "completed_methods": completed,
                       "llm_provider": args.llm_provider,
                       "judge_model": _LLM_MODEL}
            _save_partial(out_path, payload)
            print(f"  [saved partial] {out_path}  ({len(completed)} method(s))",
                  flush=True)

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

    if out_path:
        print(f"\nFinal result → {out_path}  ({len(completed)}/{len(selected_methods)} methods)")


if __name__ == "__main__":
    main()
