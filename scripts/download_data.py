"""Download and preprocess two Vietnamese QA datasets to JSONL format.

Datasets:
  UIT-ViQuAD 2.0              — tải tự động (taidng/UIT-ViQuAD2.0)
  DANGDOCAO/GeneratingQuestions — tải tự động (pháp lý / hành chính)

Output per dataset:
  data/processed/{dataset}_passages.jsonl  — corpus (unique contexts to index)
  data/processed/{dataset}_train.jsonl     — training QA pairs
  data/processed/{dataset}_dev.jsonl       — dev QA pairs
  data/processed/{dataset}_test.jsonl      — test QA pairs

Format passages.jsonl:  {"id": str, "passage": str}
Format qas.jsonl:       {"id": str, "question": str, "relevant_ids": [str], "answers": [str]}

Usage:
  uv run python scripts/download_data.py
  uv run python scripts/download_data.py --datasets viaquad
  uv run python scripts/download_data.py --datasets dangdocao
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def _ctx_id(prefix: str, text: str) -> str:
    """Stable passage ID dựa trên MD5 của nội dung."""
    return f"{prefix}_{hashlib.md5(text.encode()).hexdigest()[:10]}"


# ---------------------------------------------------------------------------
# UIT-ViQuAD 2.0
# ---------------------------------------------------------------------------

def process_viaquad(out_dir: Path) -> None:
    print("\n=== UIT-ViQuAD 2.0 (taidng/UIT-ViQuAD2.0) ===")
    ds = load_dataset("taidng/UIT-ViQuAD2.0")

    passages: dict[str, str] = {}   # id → text (deduplicated)
    split_qas: dict[str, list[dict]] = {}

    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for hf_split, out_split in split_map.items():
        if hf_split not in ds:
            continue
        qas = []
        for row in tqdm(ds[hf_split], desc=f"  {hf_split}"):
            ctx = row["context"].strip()
            pid = _ctx_id("viaquad", ctx)
            passages[pid] = ctx

            # Bỏ qua câu hỏi không trả lời được (is_impossible) ở dev/test
            # Giữ lại cho train để MLP học cả trường hợp này
            ans_field = row.get("answers") or {}
            ans_texts = (ans_field.get("text") or []) if isinstance(ans_field, dict) else []
            qas.append({
                "id": str(row["id"]),
                "question": row["question"].strip(),
                "relevant_ids": [pid],
                "answers": ans_texts,
            })
        split_qas[out_split] = qas

    passage_list = [{"id": pid, "passage": text} for pid, text in passages.items()]
    _save_jsonl(passage_list, out_dir / "viaquad_passages.jsonl")
    for split, qas in split_qas.items():
        _save_jsonl(qas, out_dir / f"viaquad_{split}.jsonl")
    print(f"  Tổng passages: {len(passage_list):,}")


# ---------------------------------------------------------------------------
# DANGDOCAO/GeneratingQuestions
# ---------------------------------------------------------------------------

def process_dangdocao(out_dir: Path, dev_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> None:
    """Tải và xử lý DANGDOCAO/GeneratingQuestions.

    Dataset chỉ có split 'train' → tự chia train/dev/test theo tỉ lệ 80/10/10.
    Mỗi row có cấu trúc SQuAD-style lồng trong field 'data'.
    """
    import random
    random.seed(seed)

    print("\n=== DANGDOCAO/GeneratingQuestions ===")
    ds = load_dataset("DANGDOCAO/GeneratingQuestions")

    # Flatten: mỗi row → nhiều QA pairs (thực tế 1 QA/row, nhưng xử lý đúng cấu trúc)
    all_records: list[dict] = []
    passages: dict[str, str] = {}

    for row in tqdm(ds["train"], desc="  Parsing"):
        data = row["data"]
        title = data.get("title", "")
        for para in data["paragraphs"]:
            ctx = para["context"].strip()
            pid = _ctx_id("dangdocao", ctx)
            passages[pid] = ctx
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                ans_texts = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                all_records.append({
                    "id": str(qa["id"]),
                    "question": qa["question"].strip(),
                    "relevant_ids": [pid],
                    "answers": ans_texts,
                    "_domain": title,   # kept for analysis, stripped before saving
                })

    # Shuffle và chia split
    random.shuffle(all_records)
    n = len(all_records)
    n_test = int(n * test_ratio)
    n_dev  = int(n * dev_ratio)
    splits = {
        "test":  all_records[:n_test],
        "dev":   all_records[n_test : n_test + n_dev],
        "train": all_records[n_test + n_dev :],
    }

    passage_list = [{"id": pid, "passage": text} for pid, text in passages.items()]
    _save_jsonl(passage_list, out_dir / "dangdocao_passages.jsonl")

    for split, records in splits.items():
        clean = [{k: v for k, v in r.items() if k != "_domain"} for r in records]
        _save_jsonl(clean, out_dir / f"dangdocao_{split}.jsonl")

    print(f"  Tổng passages: {len(passage_list):,}")
    print(f"  Train/Dev/Test: {len(splits['train']):,} / {len(splits['dev']):,} / {len(splits['test']):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download & preprocess Vietnamese QA datasets")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["viaquad", "dangdocao"],
        default=["viaquad", "dangdocao"],
    )
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "viaquad" in args.datasets:
        process_viaquad(out_dir)

    if "dangdocao" in args.datasets:
        process_dangdocao(out_dir)

    print("\nDone. Files saved to:", out_dir)


if __name__ == "__main__":
    main()
