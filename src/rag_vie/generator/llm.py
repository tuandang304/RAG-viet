import re

from openai import OpenAI
from ..config import settings

_client: OpenAI | None = None

# Qwen3 is a reasoning model and may wrap its chain-of-thought in <think>…</think>.
# Strip it so the returned answer (and any RAGAS scoring) sees only the final response.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()

_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI hữu ích. Dựa vào các đoạn văn được cung cấp, "
    "hãy trả lời câu hỏi bằng tiếng Việt một cách chính xác và súc tích. "
    "Nếu không tìm thấy thông tin trong đoạn văn, hãy nói rõ điều đó."
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)
    return _client


def generate(query: str, passages: list[str], max_tokens: int = 512) -> str:
    """Generate an answer from retrieved passages using Qwen3:32B on FPT AI Factory."""
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user_message = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"

    response = _get_client().chat.completions.create(
        model=settings.fpt_llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return _strip_think(response.choices[0].message.content or "")
