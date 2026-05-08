import pickle
from pathlib import Path

import httpx

from ..config import settings

# Timeout riêng cho sparse vì corpus lớn có thể chậm hơn dense
_TIMEOUT = httpx.Timeout(120.0)


def _call_sparse_api(texts: list[str]) -> list[dict[int, float]]:
    """Gọi FPT AI Factory để lấy BGE-M3 sparse (lexical) weights.

    FPT trả về danh sách sparse vector, mỗi vector là list of {index, value}.
    Endpoint: POST {base_url}/embed_sparse
    Body:     {"model": "...", "input": ["text1", ...]}
    Response: {"data": [{"sparse_embedding": [{"index": 123, "value": 0.5}, ...]}, ...]}

    Nếu FPT dùng format khác, chỉnh hàm _parse_sparse_response() bên dưới.
    """
    url = settings.fpt_base_url.rstrip("/") + "/embed_sparse"
    headers = {
        "Authorization": f"Bearer {settings.fpt_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": settings.fpt_sparse_model, "input": texts}

    response = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    response.raise_for_status()
    return _parse_sparse_response(response.json())


def _parse_sparse_response(body: dict) -> list[dict[int, float]]:
    """Chuyển response JSON của FPT sang list[dict[token_id, weight]].

    Hỗ trợ hai format phổ biến:
    - Format A (TEI-style):  data[i]["sparse_embedding"] = [{"index": int, "value": float}, ...]
    - Format B (dict-style): data[i]["sparse_embedding"] = {"123": 0.5, "456": 0.3, ...}
    """
    results: list[dict[int, float]] = []
    for item in body["data"]:
        raw = item["sparse_embedding"]
        if isinstance(raw, list):
            # Format A
            vec = {int(entry["index"]): float(entry["value"]) for entry in raw}
        else:
            # Format B
            vec = {int(k): float(v) for k, v in raw.items()}
        results.append(vec)
    return results


class SparseRetriever:
    """BGE-M3 sparse lexical retriever — tín hiệu từ FPT AI Factory API.

    Dùng learned sparse weights (SPLADE-style) từ BGE-M3, khác với BM25 ở chỗ
    token weights được học từ mô hình thay vì tính theo TF-IDF.
    """

    def __init__(self) -> None:
        self._inverted_index: dict[int, list[tuple[str, float]]] = {}
        self._passage_map: dict[str, str] = {}

    def build(self, passages: list[str], ids: list[str], batch_size: int = 32) -> None:
        self._inverted_index = {}
        self._passage_map = dict(zip(ids, passages))

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            sparse_vecs = _call_sparse_api(batch_passages)
            for doc_id, sparse_vec in zip(batch_ids, sparse_vecs):
                for token_id, weight in sparse_vec.items():
                    self._inverted_index.setdefault(token_id, []).append((doc_id, float(weight)))

    def search(self, query: str, k: int) -> list[tuple[str, str, float]]:
        """Returns list of (id, passage, score)."""
        query_vecs = _call_sparse_api([query])
        query_vec = query_vecs[0]

        scores: dict[str, float] = {}
        for token_id, q_weight in query_vec.items():
            for doc_id, d_weight in self._inverted_index.get(token_id, []):
                scores[doc_id] = scores.get(doc_id, 0.0) + q_weight * d_weight

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(pid, self._passage_map[pid], score) for pid, score in top_k]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"index": self._inverted_index, "passages": self._passage_map}, f)

    @classmethod
    def load(cls, path: str | Path) -> "SparseRetriever":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj._inverted_index = data["index"]
        obj._passage_map = data["passages"]
        return obj
