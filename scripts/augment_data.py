"""Augment QA JSONL files with diacritic-removed query variants.

Generates two output files:
  1. <input>_aug.jsonl  — original queries + 30% noisy copies (diacritics removed).
                          Used for MLP training so the model sees low-diacritic_ratio inputs.
  2. <input>_noisy.jsonl — 100% of queries with diacritics removed.
                           Used as a held-out robustness evaluation set.

Only the `question` field is modified; `relevant_ids`, `answers`, and `id` are preserved
(noisy copies get a `_noisy` suffix on their id to avoid duplicate key issues).

Usage:
    uv run python scripts/augment_data.py \\
        --input data/processed/viaquad_train.jsonl \\
        --noise-ratio 0.3 \\
        --seed 42

    uv run python scripts/augment_data.py \\
        --input data/processed/viaquad_dev.jsonl \\
        --noise-ratio 0.3 \\
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from rag_vie.utils.text import remove_diacritics


def augment(
    qas: list[dict],
    noise_ratio: float,
    rng: random.Random,
) -> list[dict]:
    """Return original list + noisy copies of `noise_ratio` fraction."""
    n_noisy = round(len(qas) * noise_ratio)
    chosen = rng.sample(range(len(qas)), n_noisy)
    noisy_copies = []
    for idx in chosen:
        qa = qas[idx].copy()
        qa["question"] = remove_diacritics(qa["question"])
        qa["id"] = qa.get("id", str(idx)) + "_noisy"
        noisy_copies.append(qa)
    return qas + noisy_copies


def make_noisy(qas: list[dict]) -> list[dict]:
    """Return a fully diacritic-removed version of every QA."""
    result = []
    for qa in qas:
        out = qa.copy()
        out["question"] = remove_diacritics(qa["question"])
        result.append(out)
    return result


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written {len(records):,} records → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source QA JSONL file")
    parser.add_argument("--noise-ratio", type=float, default=0.3,
                        help="Fraction of queries to add as noisy copies (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-noisy-split", action="store_true",
                        help="Skip generating the 100%%-noisy evaluation file")
    args = parser.parse_args()

    src = Path(args.input)
    with open(src, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(qas):,} QA pairs from {src}")

    rng = random.Random(args.seed)

    # 1. Augmented training file
    stem = src.stem  # e.g. "viaquad_train"
    aug_path = src.parent / f"{stem}_aug.jsonl"
    augmented = augment(qas, args.noise_ratio, rng)
    n_added = len(augmented) - len(qas)
    print(f"Augmented: {len(qas):,} original + {n_added:,} noisy copies "
          f"({args.noise_ratio*100:.0f}%) = {len(augmented):,} total")
    write_jsonl(aug_path, augmented)

    # 2. Fully-noisy evaluation file (for robustness test)
    if not args.no_noisy_split:
        noisy_path = src.parent / f"{stem}_noisy.jsonl"
        noisy = make_noisy(qas)
        print(f"Noisy eval: {len(noisy):,} queries (100% diacritics removed)")
        write_jsonl(noisy_path, noisy)

    # Quick sanity check
    sample_orig = qas[0]["question"]
    sample_noisy = remove_diacritics(sample_orig)
    print(f"\nSample:  {sample_orig}")
    print(f"  →      {sample_noisy}")


if __name__ == "__main__":
    main()
