"""Generate noisy Vietnamese queries via Ollama local LLM.

Note: Set PYTHONIOENCODING=utf-8 if running on Windows to avoid encoding errors.

Calls the Ollama REST API (chat endpoint) for each query × noise type.
Supports checkpoint/resume, retries, and progress tracking.

Usage:
    uv run python -m rag_vie.datagen.generate_noise \
        --input data/processed/dangdocao_test.jsonl \
        --noise-type missing_tone \
        --limit 20

    uv run python -m rag_vie.datagen.generate_noise \
        --input data/processed/dangdocao_test.jsonl \
        --noise-type all \
        --resume
"""

import argparse
import json
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from .config import settings
from .prompts import NOISE_TYPES, ALL_NOISE_TYPE_IDS, NoisePrompt


# ───────────────────────────────────────────────────────────────────────
# Ollama API call
# ───────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: NoisePrompt, query: str) -> str:
    """Send a single query to Ollama chat API and return the rewritten text."""
    url = f"{settings.ollama_base_url}/api/chat"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.format_user(query)},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": settings.ollama_temperature,
            "num_ctx": settings.ollama_num_ctx,
        },
    }

    last_err = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            # Strip potential markdown quotes or extra whitespace
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            return content
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_err = e
            wait = 2 ** attempt
            tqdm.write(f"  ⚠ Attempt {attempt}/{settings.max_retries} failed: {e}. "
                       f"Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Ollama API failed after {settings.max_retries} retries: {last_err}"
    )


# ───────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ───────────────────────────────────────────────────────────────────────

def _checkpoint_path(output_file: Path) -> Path:
    return output_file.with_suffix(".checkpoint.json")


def _load_checkpoint(output_file: Path) -> set[str]:
    """Return set of IDs already processed (from existing output file)."""
    done_ids: set[str] = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec.get("original_id", ""))
    return done_ids


# ───────────────────────────────────────────────────────────────────────
# Main generation loop
# ───────────────────────────────────────────────────────────────────────

def generate_for_type(
    input_path: Path,
    noise_type: str,
    limit: int = 0,
    resume: bool = False,
) -> Path:
    """Generate noisy queries for one noise type. Returns path to output file."""
    prompt = NOISE_TYPES[noise_type]
    dataset_name = input_path.stem  # e.g. "dangdocao_test"

    # Prepare output
    out_dir = settings.raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"{dataset_name}_{noise_type}.jsonl"

    # Load input
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if limit > 0:
        records = records[:limit]

    # Resume support
    done_ids: set[str] = set()
    if resume:
        done_ids = _load_checkpoint(output_file)
        if done_ids:
            tqdm.write(f"  ↻ Resuming: {len(done_ids)} already done, "
                       f"{len(records) - len(done_ids)} remaining")

    # Open file in append mode for resume
    mode = "a" if resume and done_ids else "w"

    generated = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    with open(output_file, mode, encoding="utf-8") as fout:
        pbar = tqdm(records, desc=f"  {noise_type}", unit="q")
        for rec in pbar:
            original_id = rec.get("id", "")
            if original_id in done_ids:
                skipped += 1
                continue

            question = rec["question"]

            try:
                noisy = _call_ollama(prompt, question)
            except RuntimeError as e:
                tqdm.write(f"  ✗ SKIP {original_id}: {e}")
                errors += 1
                continue

            out_record = {
                "id": f"{original_id}_{noise_type}",
                "original_id": original_id,
                "noise_type": noise_type,
                "original_question": question,
                "noisy_question": noisy,
                "model": settings.ollama_model,
                "prompt_version": "v1",
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            generated += 1

            # Update progress bar
            elapsed = time.time() - start_time
            rate = generated / elapsed if elapsed > 0 else 0
            pbar.set_postfix(gen=generated, err=errors, rate=f"{rate:.1f}q/s")

            # Flush periodically
            if generated % settings.checkpoint_every == 0:
                fout.flush()

    elapsed = time.time() - start_time
    tqdm.write(
        f"\n  ✓ {noise_type}: {generated} generated, {skipped} skipped, "
        f"{errors} errors in {elapsed:.0f}s → {output_file}"
    )
    return output_file


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate noisy Vietnamese queries via Ollama"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input QA JSONL file (e.g. data/processed/dangdocao_test.jsonl)"
    )
    parser.add_argument(
        "--noise-type", default="all",
        choices=ALL_NOISE_TYPE_IDS + ["all"],
        help="Which noise type to generate (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit number of input questions (0 = no limit)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint (skip already-processed IDs)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    types = ALL_NOISE_TYPE_IDS if args.noise_type == "all" else [args.noise_type]

    print(f"=== Generate Noise via Ollama ({settings.ollama_model}) ===")
    print(f"  Input: {input_path}")
    print(f"  Noise types: {types}")
    print(f"  Limit: {'none' if args.limit == 0 else args.limit}")
    print(f"  Resume: {args.resume}")
    print()

    for nt in types:
        generate_for_type(input_path, nt, limit=args.limit, resume=args.resume)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
