"""Distillation training for NeuralFeatureExtractor.

Uses heuristic extract_features() as a teacher to generate soft-label targets,
then trains the projection head of NeuralFeatureExtractor via MSE loss.

The PhoBERT encoder is fully frozen throughout; only the tiny projection head
(Linear(768→64)+ReLU+Linear(64→8)+Sigmoid, ~51 K params) is updated.

Usage:
    # Pre-train on ViQuAD training queries (distillation only):
    uv run python scripts/train_feature_extractor.py \\
        --qas-path data/processed/viaquad_train_aug.jsonl \\
        --index-dir indexes/viaquad \\
        --output checkpoints/feature_extractor.pt

    # Also include cross-domain queries:
    uv run python scripts/train_feature_extractor.py \\
        --qas-path data/processed/viaquad_train_aug.jsonl \\
        --extra-qas-paths data/processed/dangdocao_test.jsonl \\
        --index-dir indexes/viaquad \\
        --output checkpoints/feature_extractor.pt

    # Faster run without BM25 index (oov_ratio target will be 0.0):
    uv run python scripts/train_feature_extractor.py \\
        --qas-path data/processed/viaquad_train_aug.jsonl \\
        --output checkpoints/feature_extractor.pt --no-bm25
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from rag_vie.features.vietnamese import extract_features, FEATURE_NAMES
from rag_vie.features.neural import NeuralFeatureExtractor


def _load_queries(path: str) -> list[str]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" in obj:
                queries.append(obj["question"])
    return queries


def _build_soft_labels(
    queries: list[str],
    bm25_vocab: set[str] | None,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute heuristic features for all queries → float32 array (N, F)."""
    n_features = len(FEATURE_NAMES)
    labels = np.zeros((len(queries), n_features), dtype=np.float32)
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        for j, q in enumerate(batch):
            labels[i + j] = extract_features(q, bm25_vocab=bm25_vocab)
    return labels


def _encode_all(
    extractor: NeuralFeatureExtractor,
    queries: list[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode all queries through frozen encoder → (N, hidden_dim) tensor."""
    all_embeddings = []
    extractor.encoder.eval()
    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
        batch = queries[i : i + batch_size]
        inputs = extractor.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(extractor.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = extractor.encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)


def train(
    extractor: NeuralFeatureExtractor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> None:
    device = extractor.device
    embeddings = embeddings.to(device)
    targets = targets.to(device)

    optimizer = AdamW(extractor.projection.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    loss_fn = nn.MSELoss()

    n = len(embeddings)
    extractor.projection.train()

    for epoch in range(1, epochs + 1):
        indices = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            emb_b = embeddings[idx]
            tgt_b = targets[idx]

            preds = extractor.projection(emb_b)
            loss = loss_fn(preds, tgt_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distillation training for NeuralFeatureExtractor")
    parser.add_argument("--qas-path", required=True, help="Primary QA JSONL (questions used as training queries)")
    parser.add_argument("--extra-qas-paths", nargs="*", default=[], help="Additional QA JSONL files")
    parser.add_argument("--index-dir", default=None, help="Index directory (for BM25 vocab; enables oov_ratio targets)")
    parser.add_argument("--no-bm25", action="store_true", help="Skip BM25 vocab loading (oov_ratio targets = 0.0)")
    parser.add_argument("--output", default="checkpoints/feature_extractor.pt")
    parser.add_argument("--model-name", default="vinai/phobert-base")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load queries
    queries = _load_queries(args.qas_path)
    for extra in args.extra_qas_paths:
        queries.extend(_load_queries(extra))
    queries = list(dict.fromkeys(queries))  # deduplicate, preserve order
    print(f"Total unique queries: {len(queries)}")

    # Optional: load BM25 vocab to get oov_ratio soft-labels
    bm25_vocab: set[str] | None = None
    if not args.no_bm25 and args.index_dir:
        bm25_pkl = Path(args.index_dir) / "bm25.pkl"
        if bm25_pkl.exists():
            import pickle
            with open(bm25_pkl, "rb") as f:
                bm25_obj = pickle.load(f)
            bm25_vocab = set(bm25_obj["bm25"].idf.keys())
            print(f"BM25 vocab size: {len(bm25_vocab):,}")
        else:
            print(f"Warning: {bm25_pkl} not found — oov_ratio targets will be 0.0")

    # Compute heuristic soft-labels
    print("Computing heuristic soft-labels...")
    labels_np = _build_soft_labels(queries, bm25_vocab)
    print(f"Soft-label stats: mean={labels_np.mean(0).round(3)}, std={labels_np.std(0).round(3)}")

    # Build neural extractor
    print(f"Loading encoder: {args.model_name}")
    extractor = NeuralFeatureExtractor(model_name=args.model_name)

    # Pre-compute frozen encoder embeddings (reused across epochs)
    print("Pre-computing encoder embeddings (frozen)...")
    embeddings = _encode_all(extractor, queries)

    targets = torch.tensor(labels_np)

    # Train projection head
    print(f"\nTraining projection head for {args.epochs} epochs...")
    train(
        extractor=extractor,
        embeddings=embeddings,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    extractor.save(args.output)
    print(f"\nSaved projection weights → {args.output}")

    # Quick sanity check
    extractor.projection.eval()
    sample_features = extractor.extract(queries[0])
    heuristic_features = extract_features(queries[0], bm25_vocab=bm25_vocab)
    print(f"\nSanity check on: {queries[0]!r}")
    print(f"  Neural:    {sample_features.round(3)}")
    print(f"  Heuristic: {heuristic_features.round(3)}")
    mse = float(np.mean((sample_features - heuristic_features) ** 2))
    print(f"  MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
