"""Loads and caches RAGPipeline instances for the testing API.

Index/checkpoint artifacts (FAISS index, BM25/sparse pickles, Keras MLP)
typically live on the machine that ran scripts/build_index.py + train_mlp.py.
When that data isn't present locally (e.g. it lives on another machine),
every lookup here fails with a clear PipelineLoadError instead of crashing —
the API and web UI stay usable to browse config even with no data mounted.
"""

from pathlib import Path

from ..config import settings
from ..fusion.mlp import FusionMLP
from ..pipeline import RAGPipeline
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import DenseRetriever
from ..retrieval.sparse import SparseRetriever

CHANNEL_LABELS = ("dense", "bm25", "sparse", "toneless")


class PipelineLoadError(Exception):
    """Required index/checkpoint artifacts are missing or unreadable."""


class PipelineManager:
    """Lazily builds RAGPipeline instances and caches them by (index_dir, mlp_path, use_generator).

    Loading FAISS + BM25 + BGE-M3 + Keras artifacts takes real time, so a
    pipeline built once is reused across requests for the life of the process.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, bool], RAGPipeline] = {}

    def list_index_dirs(self) -> list[str]:
        base = Path(settings.index_dir)
        if not base.exists():
            return []
        return sorted(
            d.name for d in base.iterdir() if d.is_dir() and (d / "index.faiss").exists()
        )

    def list_mlp_checkpoints(self) -> list[str]:
        base = Path(settings.checkpoint_dir)
        if not base.exists():
            return []
        return sorted(f.name for f in base.glob("*.keras"))

    def default_mlp_path(self) -> str:
        return f"{settings.checkpoint_dir}/fusion_mlp_multidomain.keras"

    def get_pipeline(self, index_dir: str, mlp_path: str, use_generator: bool) -> RAGPipeline:
        key = (index_dir, mlp_path, use_generator)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._load(index_dir, mlp_path, use_generator)
            self._cache[key] = cached
        return cached

    def _load(self, index_dir: str, mlp_path: str, use_generator: bool) -> RAGPipeline:
        index_path = Path(index_dir)
        if not (index_path / "index.faiss").exists():
            raise PipelineLoadError(
                f"Không tìm thấy index tại '{index_dir}' (thiếu index.faiss). "
                "Data có thể đang ở máy khác — trỏ INDEX_DIR hoặc copy thư mục index về đây."
            )
        bm25_path = index_path / "bm25.pkl"
        if not bm25_path.exists():
            raise PipelineLoadError(f"Không tìm thấy '{bm25_path}'.")
        if not Path(mlp_path).exists():
            raise PipelineLoadError(f"Không tìm thấy checkpoint MLP tại '{mlp_path}'.")

        sparse_path = index_path / "sparse.pkl"
        toneless_path = index_path / "bm25_toneless.pkl"

        dense = DenseRetriever.load(index_path)
        bm25 = BM25Retriever.load(bm25_path)
        mlp = FusionMLP.load(mlp_path)
        sparse = SparseRetriever.load(sparse_path) if sparse_path.exists() else None
        toneless = BM25Retriever.load(toneless_path) if toneless_path.exists() else None

        return RAGPipeline(
            dense, bm25, mlp, sparse,
            use_generator=use_generator,
            toneless=toneless,
        )


manager = PipelineManager()
