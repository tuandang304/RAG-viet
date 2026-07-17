"""Diacritic-restoration baseline: LLM restores tone marks BEFORE retrieval.

The obvious alternative to the toneless retrieval channel is "fix the query
first": restore diacritics with an LLM, then run the standard hybrid pipeline.
This script produces the restored-query JSONL so evaluate_all.py can measure
that baseline on exactly the same queries as the noisy n=500 runs.

Usage:
    uv run python scripts/restore_diacritics.py \\
        --qas-path data/processed/viaquad_dev_noisy.jsonl \\
        --max-samples 500 --seed 42 \\
        --output data/processed/viaquad_dev_noisy500_restored.jsonl

    # then:
    uv run python scripts/evaluate_all.py \\
        --qas-path data/processed/viaquad_dev_noisy500_restored.jsonl ...

--max-samples/--seed replicate evaluate_all's subsampling, so the restored
file contains the SAME 500 queries the earlier noisy evaluations used.
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from rag_vie.config import settings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SYSTEM_PROMPT = (
    "Bạn là công cụ khôi phục dấu tiếng Việt. Cho một câu đã mất dấu, khôi "
    "phục đầy đủ dấu. Giữ nguyên số lượng từ và trật tự từ, không thêm bớt "
    "hay diễn giải. Chỉ in ra đúng câu đã khôi phục dấu, KHÔNG lời dẫn, "
    "KHÔNG markdown, KHÔNG dấu ngoặc kép.\n"
    "Ví dụ:\n"
    "Vào: thu do cua viet nam la gi\n"
    "Ra: Thủ đô của Việt Nam là gì\n"
    "Vào: toi cuop tai san bi phat tu bao nhieu nam\n"
    "Ra: Tội cướp tài sản bị phạt tù bao nhiêu năm"
)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.S)
# Preambles the model sometimes prepends despite instructions.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:đã\s+)?(?:khôi\s+phục\s+dấu|câu\s+(?:đã\s+)?khôi\s+phục|kết\s+quả|ra)\s*:?\s*",
    flags=re.I,
)


def _clean(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]                     # answer is the last content line
    line = line.replace("**", "").replace("*", "")   # strip markdown emphasis
    line = _PREAMBLE_RE.sub("", line)                # strip "Đã khôi phục dấu:" etc.
    line = line.strip().strip('"').strip("'").strip("`").strip()
    return line


def restore_one(client: OpenAI, question: str) -> str:
    # Qwen3.6 on FPT is a reasoning model whose thinking tokens count against
    # max_tokens (enable_thinking=False is ignored server-side) — too small a
    # budget yields finish_reason="length" with EMPTY content. Long legal
    # queries can burn >2k reasoning tokens, so escalate the budget once.
    n_words = len(question.split())
    for max_tokens in (4096, 12288):
        response = client.chat.completions.create(
            model=settings.fpt_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Vào: {question}\nRa:"},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        cleaned = _clean(response.choices[0].message.content or "")
        if not cleaned:
            continue
        # Guard against the model rambling/expanding: word count must stay
        # close to the input. Otherwise the restoration is untrustworthy —
        # keep the noisy query so the baseline isn't unfairly polluted.
        if abs(len(cleaned.split()) - n_words) <= max(2, round(0.3 * n_words)):
            return cleaned
        return question
    return question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas-path", required=True, help="Noisy QA JSONL")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Subsample N queries (same semantics/seed as evaluate_all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.qas_path, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(qas):,} queries from {args.qas_path}")

    if args.max_samples is not None and args.max_samples < len(qas):
        qas = random.Random(args.seed).sample(qas, args.max_samples)
        print(f"Subsampled to {len(qas):,} queries (seed={args.seed})")

    client = OpenAI(api_key=settings.fpt_api_key, base_url=settings.fpt_base_url)

    def process(qa: dict) -> dict:
        out = dict(qa)
        out["original_question"] = qa["question"]
        try:
            out["question"] = restore_one(client, qa["question"])
        except Exception as exc:  # keep the noisy query on API failure
            print(f"  restore failed ({exc}) — keeping original", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        restored = list(tqdm(pool.map(process, qas), total=len(qas), desc="Restoring"))

    changed = sum(1 for r in restored if r["question"] != r["original_question"])
    print(f"Restored {changed}/{len(restored)} queries (rest unchanged)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in restored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
