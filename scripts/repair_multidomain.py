import json
import hashlib
from pathlib import Path
from datasets import load_dataset

def _ctx_id(prefix: str, text: str) -> str:
    """Stable passage ID based on MD5 hash of text."""
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"

def main():
    train_path = Path("data/processed/multidomain_train.jsonl")
    passages_path = Path("data/processed/multidomain_passages.jsonl")
    
    print("==============================================")
    # 1. Load and repair training queries
    print(f"Loading queries from {train_path}...")
    queries = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
                
    print(f"Loaded {len(queries)} training queries.")
    
    repaired_count = 0
    for q in queries:
        # Check if query contains raw IDs that should be prefixed
        # We look at the first relevant_id. If it does not start with 'tvpl_' and the source indicates tvpl:
        is_tvpl = "tvpl" in q.get("id", "") or "tvpl" in q.get("source", "")
        new_rel_ids = []
        for rid in q["relevant_ids"]:
            if is_tvpl and not rid.startswith("tvpl_"):
                new_rel_ids.append(f"tvpl_{rid}")
                repaired_count += 1
            else:
                new_rel_ids.append(rid)
        q["relevant_ids"] = new_rel_ids
        
    print(f"Repaired {repaired_count} passage references in queries.")
    
    # Save repaired queries
    with open(train_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Repaired queries saved back to {train_path}.")
    
    print("\n==============================================")
    # 2. Re-extract unique passage IDs
    referenced_pids = set()
    for q in queries:
        for pid in q["relevant_ids"]:
            referenced_pids.add(pid)
            
    print(f"Total unique referenced passages: {len(referenced_pids)}")
    
    # 3. Load TVPL and ViCoQA corpora to fetch text
    print("\nLoading TVPL corpus from HuggingFace...")
    tvpl_c_ds = load_dataset("GreenNode/TVPL-Retrieval-VN", "corpus")["test"]
    tvpl_corpus = {row["id"]: row["text"].strip() for row in tvpl_c_ds}
    print(f"Loaded {len(tvpl_corpus)} TVPL corpus passages.")
    
    print("\nLoading ViCoQA train split from HuggingFace...")
    vicoqa_train = load_dataset("HAT-FU/vicoqa_v1")["train"]
    vicoqa_corpus = {}
    for row in vicoqa_train:
        story_text = row["story"].strip()
        pid = _ctx_id("vicoqa", story_text)
        vicoqa_corpus[pid] = story_text
    print(f"Loaded {len(vicoqa_corpus)} ViCoQA corpus passages.")
    
    # 4. Construct the repaired merged passages list
    print("\nConstructing merged passages...")
    merged_passages = []
    missing_pids = []
    for pid in sorted(referenced_pids):
        if pid.startswith("tvpl_"):
            raw_id = pid.replace("tvpl_", "")
            if raw_id in tvpl_corpus:
                merged_passages.append({"id": pid, "passage": tvpl_corpus[raw_id]})
            else:
                missing_pids.append(pid)
        elif pid.startswith("vicoqa_"):
            if pid in vicoqa_corpus:
                merged_passages.append({"id": pid, "passage": vicoqa_corpus[pid]})
            else:
                missing_pids.append(pid)
        else:
            missing_pids.append(pid)
            
    print(f"Successfully fetched {len(merged_passages)} passages.")
    if missing_pids:
        print(f"WARNING: {len(missing_pids)} passages were not found in either corpus!")
        print("Missing sample IDs:", missing_pids[:10])
        
    # 5. Save the merged passages
    with open(passages_path, "w", encoding="utf-8") as f:
        for r in merged_passages:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Repaired passages saved to {passages_path}.")
    print("==============================================")

if __name__ == "__main__":
    main()
