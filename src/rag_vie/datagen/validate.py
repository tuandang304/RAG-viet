"""Validate noisy queries using FPT Vietnamese Embedding similarity.

Computes cosine similarity between original and noisy question embeddings.
Filters by threshold (default ≥ 0.85) and outputs statistics.

Usage:
    uv run python -m rag_vie.datagen.validate \
        --input data/generated/raw/dangdocao_test_missing_tone.jsonl

    uv run python -m rag_vie.datagen.validate --input-dir data/generated/raw/
"""

import argparse
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import settings


# ───────────────────────────────────────────────────────────────────────
# Embedding via FPT API (reuse same approach as src/rag_vie/retrieval/embedder.py)
# ───────────────────────────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.fpt_api_key,
            base_url=settings.fpt_base_url,
        )
    return _client


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts, returns (N, dim) float32 array."""
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.fpt_embedding_model, input=batch
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, dim) arrays → (N,)."""
    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


# ───────────────────────────────────────────────────────────────────────
# Validation logic
# ───────────────────────────────────────────────────────────────────────

def validate_file(input_path: Path, threshold: float | None = None) -> Path:
    """Validate one raw JSONL file. Returns path to validated output."""
    if threshold is None:
        threshold = settings.similarity_threshold

    # Read records
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        print(f"  ⚠ Empty file: {input_path}")
        return input_path

    print(f"\n  Validating {len(records)} records from {input_path.name}")

    # Extract texts
    originals = [r["original_question"] for r in records]
    noisys = [r["noisy_question"] for r in records]

    # Embed in batches
    print("  Embedding original questions...")
    orig_emb = embed_texts(originals)
    print("  Embedding noisy questions...")
    noisy_emb = embed_texts(noisys)

    # Compute similarity
    sims = cosine_similarity(orig_emb, noisy_emb)

    # Split pass/fail
    passed = []
    failed = []
    for rec, sim in zip(records, sims, strict=True):
        rec["similarity_score"] = round(float(sim), 4)
        if sim >= threshold:
            passed.append(rec)
        else:
            failed.append(rec)

    # Write validated output
    out_dir = settings.validated_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / input_path.name

    with open(out_path, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write rejected log
    log_dir = settings.logs_path
    log_dir.mkdir(parents=True, exist_ok=True)
    reject_path = log_dir / f"rejected_{input_path.name}"
    with open(reject_path, "w", encoding="utf-8") as f:
        for r in failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics
    noise_type = records[0].get("noise_type", "unknown") if records else "unknown"
    avg_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    max_sim = float(np.max(sims))

    print(f"  ── {noise_type} ──")
    print(f"  Total:    {len(records)}")
    print(f"  Passed:   {len(passed)} ({len(passed)/len(records)*100:.1f}%)")
    print(f"  Rejected: {len(failed)} ({len(failed)/len(records)*100:.1f}%)")
    print(f"  Similarity — avg: {avg_sim:.4f}, min: {min_sim:.4f}, max: {max_sim:.4f}")
    print(f"  Output:   {out_path}")
    if failed:
        print(f"  Rejects:  {reject_path}")

    return out_path


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate noisy queries using embedding similarity"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", type=str,
        help="Single raw JSONL file to validate"
    )
    group.add_argument(
        "--input-dir", type=str,
        help="Directory of raw JSONL files to validate"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=f"Similarity threshold (default: {settings.similarity_threshold})"
    )
    args = parser.parse_args()

    print("=== Validate Noisy Queries (FPT Embedding) ===")
    print(f"  Threshold: {args.threshold or settings.similarity_threshold}")

    if args.input:
        validate_file(Path(args.input), args.threshold)
    else:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.jsonl"))
        if not files:
            print(f"  ⚠ No JSONL files found in {input_dir}")
            return
        for f in files:
            validate_file(f, args.threshold)

    print("\n=== Validation Done ===")


if __name__ == "__main__":
    main()
