"""Orchestrator — run noise generation + validation pipeline.

Usage:
    # Test trước 20 câu (~2 phút)
    uv run python -m rag_vie.datagen.run_all --limit 20

    # Chạy 1 dataset, 1 loại nhiễu
    uv run python -m rag_vie.datagen.run_all \
        --input data/processed/dangdocao_test.jsonl \
        --noise-types missing_tone

    # Chạy full pipeline cho tất cả dataset
    uv run python -m rag_vie.datagen.run_all --noise-types all

    # Chỉ validate (không sinh mới)
    uv run python -m rag_vie.datagen.run_all --validate-only
"""

import argparse
from pathlib import Path

from .generate_noise import generate_for_type
from .validate import validate_file
from .prompts import ALL_NOISE_TYPE_IDS


# Default input files — 3 datasets hiện có
DEFAULT_INPUTS = [
    "data/processed/dangdocao_test.jsonl",
    "data/processed/dangdocao_train.jsonl",
    "data/processed/viaquad_train.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run noise generation + validation pipeline"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", default=None,
        help="Input QA JSONL file(s). Default: all 3 datasets"
    )
    parser.add_argument(
        "--noise-types", nargs="+", default=["all"],
        choices=ALL_NOISE_TYPE_IDS + ["all"],
        help="Noise types to generate (default: all)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit input questions per dataset (0 = no limit)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip validation step (generate only)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run validation on existing raw output"
    )
    args = parser.parse_args()

    # Resolve noise types
    if "all" in args.noise_types:
        noise_types = ALL_NOISE_TYPE_IDS
    else:
        noise_types = args.noise_types

    # Resolve input files
    input_files = [Path(p) for p in (args.input or DEFAULT_INPUTS)]

    print("=" * 50)
    print("  GenData - LLM Noise Generation Pipeline")
    print("=" * 50)
    print(f"  Inputs:      {[p.name for p in input_files]}")
    print(f"  Noise types: {noise_types}")
    print(f"  Limit:       {'none' if args.limit == 0 else args.limit}")
    print(f"  Resume:      {args.resume}")
    print()

    generated_files: list[Path] = []

    if not args.validate_only:
        # ── Step 1: Generate ────────────────────────────────────────
        print("--- Step 1: Generate Noisy Queries ---")
        for input_path in input_files:
            if not input_path.exists():
                print(f"  ⚠ Skipping (not found): {input_path}")
                continue
            for nt in noise_types:
                out = generate_for_type(
                    input_path, nt,
                    limit=args.limit,
                    resume=args.resume,
                )
                generated_files.append(out)

    if not args.skip_validate:
        # ── Step 2: Validate ────────────────────────────────────────
        print("\n--- Step 2: Validate with Embedding Similarity ---")

        if args.validate_only:
            # Find all raw files
            from .config import settings
            raw_dir = settings.raw_dir
            if raw_dir.exists():
                generated_files = sorted(raw_dir.glob("*.jsonl"))
            else:
                print(f"  ⚠ No raw output directory found: {raw_dir}")
                return

        for raw_file in generated_files:
            if raw_file.exists():
                validate_file(raw_file)

    print("\n" + "=" * 50)
    print("  Pipeline Done")
    print("=" * 50)


if __name__ == "__main__":
    main()
