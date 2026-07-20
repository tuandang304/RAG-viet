"""Smoke test MiniMax Vietnamese RAG prompt with thinking disabled."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI
client = OpenAI(api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
                base_url="https://api.minimax.io/v1")

sys_prompt = "Bạn là trổ trợ lý AI. Trả lời ngắn gọn bằng tiếng Việt."
ctx = "[1] Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc."
user_msg = f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: Thủ đô của Việt Nam là gì?"
r = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "system", "content": sys_prompt},
              {"role": "user", "content": user_msg}],
    max_tokens=512,
    temperature=0.1,
    extra_body={"thinking": {"type": "disabled"}},
)
c = r.choices[0].message.content
print("content:", repr(c))
print("has <think>:", "<think>" in c)
print("usage:", r.usage)
print("finish:", r.choices[0].finish_reason)