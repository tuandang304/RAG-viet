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

    Split is **group-by-passage** (group-aware): every QA whose `relevant_ids`
    contains a given passage is assigned to the same split as that passage.
    This prevents passage-level leakage between train/dev/test — without it,
    the same passage can have one QA in train and another in test, and the
    MLP indirectly learns retrieval patterns for that passage at train time.
    """
    import random
    rng = random.Random(seed)

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

    # Group-by-passage split — assign every passage to one split, then map QAs.
    all_pids = sorted(passages.keys())   # sorted for determinism before shuffle
    rng.shuffle(all_pids)
    n_pid = len(all_pids)
    n_test_pid = int(n_pid * test_ratio)
    n_dev_pid  = int(n_pid * dev_ratio)
    pid_to_split = {}
    for pid in all_pids[:n_test_pid]:
        pid_to_split[pid] = "test"
    for pid in all_pids[n_test_pid : n_test_pid + n_dev_pid]:
        pid_to_split[pid] = "dev"
    for pid in all_pids[n_test_pid + n_dev_pid :]:
        pid_to_split[pid] = "train"

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for rec in all_records:
        # relevant_ids holds exactly one pid (constructed above); take the first.
        pid = rec["relevant_ids"][0]
        splits[pid_to_split[pid]].append(rec)

    # Shuffle within each split so the QA order is independent of the underlying
    # passage iteration order from HuggingFace (which is grouped by title/document).
    for split in splits.values():
        rng.shuffle(split)

    passage_list = [{"id": pid, "passage": text} for pid, text in passages.items()]
    _save_jsonl(passage_list, out_dir / "dangdocao_passages.jsonl")

    for split, records in splits.items():
        clean = [{k: v for k, v in r.items() if k != "_domain"} for r in records]
        _save_jsonl(clean, out_dir / f"dangdocao_{split}.jsonl")

    # Sanity: assert no passage leakage across splits.
    pids_per_split = {s: {r["relevant_ids"][0] for r in records} for s, records in splits.items()}
    leak_train_test = pids_per_split["train"] & pids_per_split["test"]
    leak_train_dev  = pids_per_split["train"] & pids_per_split["dev"]
    leak_dev_test   = pids_per_split["dev"]   & pids_per_split["test"]
    assert not (leak_train_test or leak_train_dev or leak_dev_test), \
        "Passage leakage detected after group-aware split — bug in pid_to_split"

    print(f"  Tổng passages: {len(passage_list):,}")
    print(f"  Passages per split (train/dev/test): "
          f"{len(pids_per_split['train']):,} / {len(pids_per_split['dev']):,} / {len(pids_per_split['test']):,}")
    print(f"  QAs per split    (train/dev/test): "
          f"{len(splits['train']):,} / {len(splits['dev']):,} / {len(splits['test']):,}")


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
