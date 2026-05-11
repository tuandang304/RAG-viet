"""BGE-M3 sparse (lexical) retriever — local model, inverted index.

Uses FlagEmbedding BGEM3FlagModel to compute SPLADE-style lexical weights.
Model is downloaded once (~570 MB) and cached by HuggingFace hub. Runs on CUDA
when available (FlagEmbedding's default device-detection picks GPU automatically
and `use_fp16=True` cuts VRAM in half), otherwise falls back to CPU.
"""

import os
import pickle
from pathlib import Path

_MODEL_NAME = "BAAI/bge-m3"
_model = None   # lazy singleton — loaded on first call

# Batch size for BGE-M3 encoding. Override with BGE_M3_BATCH_SIZE if needed
# (default 64 fits comfortably in 6 GB VRAM with fp16 weights + activations).
_DEFAULT_BATCH = int(os.environ.get("BGE_M3_BATCH_SIZE", 64))


def _device_info() -> tuple[str, bool]:
    """Return (device_name, use_fp16). fp16 only makes sense on CUDA."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0), True
    return "cpu", False


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        device, fp16 = _device_info()
        print(f"  [sparse] Loading BGE-M3 on {device} (fp16={fp16})", flush=True)
        # FlagEmbedding auto-detects CUDA when present; use_fp16 only applies on GPU.
        _model = BGEM3FlagModel(_MODEL_NAME, use_fp16=fp16)
    return _model


def _encode_sparse(texts: list[str], batch_size: int | None = None) -> list[dict[str, float]]:
    """Return list of lexical weight dicts: [{token_string: weight, ...}].

    Encodes the whole list in one `model.encode` call when possible — FlagEmbedding
    handles internal mini-batching, and a single call amortizes Python overhead.
    """
    model = _get_model()
    out = model.encode(
        texts,
        batch_size=batch_size or _DEFAULT_BATCH,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return out["lexical_weights"]


class SparseRetriever:
    """BGE-M3 lexical sparse retriever — inverted index over token-level weights."""

    def __init__(self) -> None:
        self._inverted_index: dict[str, list[tuple[str, float]]] = {}
        self._passage_map: dict[str, str] = {}

    def build(self, passages: list[str], ids: list[str], batch_size: int | None = None) -> None:
        from tqdm import tqdm

        self._inverted_index = {}
        self._passage_map = dict(zip(ids, passages))

        bs = batch_size or _DEFAULT_BATCH
        # Encode in larger chunks of `bs * 8` and let FlagEmbedding sub-batch internally.
        chunk = bs * 8
        for i in tqdm(range(0, len(passages), chunk), desc="  Sparse index"):
            batch = passages[i : i + chunk]
            batch_ids = ids[i : i + chunk]
            vecs = _encode_sparse(batch, batch_size=bs)
            for doc_id, vec in zip(batch_ids, vecs):
                for token, weight in vec.items():
                    if weight > 0:
                        self._inverted_index.setdefault(token, []).append(
                            (doc_id, float(weight))
                        )

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        query_vec = _encode_sparse([query])[0]
        scores: dict[str, float] = {}
        for token, q_w in query_vec.items():
            for doc_id, d_w in self._inverted_index.get(token, []):
                scores[doc_id] = scores.get(doc_id, 0.0) + q_w * d_w

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(pid, self._passage_map[pid], s) for pid, s in top_k]

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[str, str, float]]]:
        """Batch version of `search` — runs BGE-M3 once on the whole query list.

        Much faster than calling `search` in a Python loop because a single encode
        amortizes the GPU dispatch + Python/C boundary overhead across N queries.
        """
        vecs = _encode_sparse(queries)
        out: list[list[tuple[str, str, float]]] = []
        for q_vec in vecs:
            scores: dict[str, float] = {}
            for token, q_w in q_vec.items():
                for doc_id, d_w in self._inverted_index.get(token, []):
                    scores[doc_id] = scores.get(doc_id, 0.0) + q_w * d_w
            top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            out.append([(pid, self._passage_map[pid], s) for pid, s in top_k])
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"index": self._inverted_index, "passages": self._passage_map}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> "SparseRetriever":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._inverted_index = data["index"]
        obj._passage_map = data["passages"]
        return obj
