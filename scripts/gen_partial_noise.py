"""Generate partial-diacritic-noise variants for the noise-robustness curve.

For a fixed base query set, produce one file per noise level p ∈ levels:
each syllable independently has its diacritics stripped with probability p.
This sweeps the query's diacritic_ratio (≈ 1 − p), the exact feature the
router gates on, so evaluating every level on the SAME base queries traces
NDCG@10 vs noise level for the router and each fixed baseline.

Usage:
    uv run python scripts/gen_partial_noise.py \\
        --input data/processed/viaquad_dev.jsonl \\
        --n 500 --seed 42 \\
        --out-prefix data/processed/curve/viaquad
    # → viaquad_p0.jsonl, viaquad_p25.jsonl, ... viaquad_p100.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from rag_vie.utils.text import remove_diacritics

LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def partial_strip(question: str, p: float, rng: random.Random) -> str:
    if p <= 0:
        return question
    return " ".join(
        remove_diacritics(tok) if rng.random() < p else tok
        for tok in question.split()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", required=True,
                        help="e.g. data/processed/curve/viaquad → <prefix>_p{level}.jsonl")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        qas = [json.loads(line) for line in f if line.strip()]
    qas = [q for q in qas if q.get("relevant_ids")]
    if args.n < len(qas):
        qas = random.Random(args.seed).sample(qas, args.n)
    print(f"Base: {len(qas)} queries from {args.input}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        # Fresh RNG per level so removal decisions are independent and
        # reproducible; the base query order is identical across levels.
        rng = random.Random(args.seed + int(level * 100))
        tag = f"p{int(level * 100)}"
        path = out_prefix.parent / f"{out_prefix.name}_{tag}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for q in qas:
                out = dict(q)
                out["question"] = partial_strip(q["question"], level, rng)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  {tag}: → {path}")


if __name__ == "__main__":
    main()
