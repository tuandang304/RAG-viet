"""FastAPI backend for the RAG_vie testing web UI.

Run with:
    uv run uvicorn rag_vie.api.app:app --reload

Serves the JSON API under /api/* and the static single-page UI at /.
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import settings
from .schemas import (
    CompareResponse,
    HealthResponse,
    IndexesResponse,
    MethodResult,
    QueryRequest,
    QueryResponse,
    RetrievedPassage,
)
from .service import CHANNEL_LABELS, PipelineLoadError, manager

_STATIC_DIR = Path(__file__).parent / "static"

_METHOD_LABELS = {
    "mlp": "Dynamic MLP (ours)",
    "fixed_equal": "Fixed equal (1/3, 1/3, 1/3)",
    "dense_bm25": "Dense + BM25 (0.5, 0.5)",
    "dense": "Dense only",
    "bm25": "BM25 only",
    "sparse": "Sparse only (BGE-M3)",
}

app = FastAPI(title="RAG_vie — Testing UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/indexes", response_model=IndexesResponse)
def list_indexes() -> IndexesResponse:
    return IndexesResponse(
        index_dirs=manager.list_index_dirs(),
        mlp_checkpoints=manager.list_mlp_checkpoints(),
        default_index_dir=settings.index_dir,
        default_mlp_path=manager.default_mlp_path(),
    )


def _resolve_pipeline(req: QueryRequest):
    index_dir = req.index_dir or settings.index_dir
    mlp_path = req.mlp_path or manager.default_mlp_path()
    try:
        pipeline = manager.get_pipeline(index_dir, mlp_path, req.use_generator)
    except PipelineLoadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return pipeline, index_dir, mlp_path


@app.post("/api/query", response_model=QueryResponse)
def run_query(req: QueryRequest) -> QueryResponse:
    pipeline, index_dir, mlp_path = _resolve_pipeline(req)

    start = time.perf_counter()
    try:
        result = pipeline.run(req.query)
    except Exception as exc:  # FPT API errors, malformed checkpoints, etc.
        raise HTTPException(status_code=502, detail=f"Lỗi khi chạy pipeline: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        query=result.query,
        weights=dict(zip(CHANNEL_LABELS, result.weights, strict=False)),
        retrieved=[
            RetrievedPassage(id=pid, passage=passage, score=score)
            for pid, passage, score in result.retrieved
        ],
        answer=result.answer,
        latency_ms=round(latency_ms, 1),
        index_dir=index_dir,
        mlp_path=mlp_path,
    )


@app.post("/api/compare", response_model=CompareResponse)
def compare_methods(req: QueryRequest) -> CompareResponse:
    """Run one query through the MLP router and the paper's fixed-weight
    baselines, sharing a single retrieval pass — only the MLP result pays for
    an LLM call."""
    pipeline, index_dir, mlp_path = _resolve_pipeline(req)

    start = time.perf_counter()
    try:
        results = pipeline.compare(req.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Lỗi khi chạy pipeline: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    methods = [
        MethodResult(
            name=name,
            label=_METHOD_LABELS.get(name, name),
            weights=dict(zip(CHANNEL_LABELS, result.weights, strict=False)),
            retrieved=[
                RetrievedPassage(id=pid, passage=passage, score=score)
                for pid, passage, score in result.retrieved
            ],
            answer=result.answer,
        )
        for name, result in results.items()
    ]

    return CompareResponse(
        query=req.query,
        methods=methods,
        latency_ms=round(latency_ms, 1),
        index_dir=index_dir,
        mlp_path=mlp_path,
    )


# Mounted last so it only catches requests that didn't match an /api/* route above.
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
