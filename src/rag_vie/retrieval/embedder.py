import time

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from ..config import settings

_client: OpenAI | None = None

# Transient failures worth retrying: connection drops, request timeouts, rate
# limits, and 5xx/524 origin timeouts (FPT sits behind Cloudflare, which returns
# a retryable 524 when the origin is slow). A single such hiccup must not crash
# a multi-hour evaluation.
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0   # seconds: 2, 4, 8, 16, 32


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)
    return _client


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429
    return False


def _embed_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=settings.fpt_embedding_model, input=batch
            )
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 — re-raised below if not retryable
            if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                raise
            last_exc = exc
            wait = _BACKOFF_BASE * (2 ** attempt)
            print(
                f"  [embedder] {type(exc).__name__} — retry {attempt + 1}/{_MAX_RETRIES} "
                f"in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    raise last_exc if last_exc else RuntimeError("unreachable")


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of passages, returns (N, dim) float32 array.

    Each batch is retried with exponential backoff on transient FPT/Cloudflare
    errors (5xx/524, timeouts, rate limits) — see module constants.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        all_embeddings.extend(_embed_batch(client, texts[i : i + batch_size]))
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query, returns (1, dim) float32 array."""
    return embed_texts([query])
