from openai import OpenAI
import numpy as np
from ..config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)
    return _client


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of passages, returns (N, dim) float32 array."""
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=settings.fpt_embedding_model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query, returns (1, dim) float32 array."""
    return embed_texts([query])
