"""Download, sample, and construct the multi-domain training dataset:
- BM25 channel: 2,000 clean queries from GreenNode/TVPL-Retrieval-VN
- Dense channel: 2,000 clean queries from HAT-FU/vicoqa_v1
- Sparse channel: 2,000 noisy queries (1,000 TVPL + 1,000 ViCoQA) generated via local Ollama
"""

import ast
import json
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Import noise generator components from rag_vie.datagen
from rag_vie.datagen.generate_noise import _call_ollama
from rag_vie.datagen.prompts import NOISE_TYPES
from rag_vie.datagen.validate import embed_texts, cosine_similarity

# Constants
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _ctx_id(prefix: str, text: str) -> str:
    """Stable passage ID based on MD5 hash of text."""
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"

def main():
    print("====================================================")
    print("STEP 1: LOADING DATASETS FROM HUGGINGFACE")
    print("====================================================")
    
    # 1. Load TVPL
    print("Loading TVPL (GreenNode/TVPL-Retrieval-VN) queries, corpus, and qrels...")
    tvpl_q_ds = load_dataset("GreenNode/TVPL-Retrieval-VN", "queries")["test"]
    tvpl_c_ds = load_dataset("GreenNode/TVPL-Retrieval-VN", "corpus")["test"]
    tvpl_r_ds = load_dataset("GreenNode/TVPL-Retrieval-VN")["test"]
    
    # Map TVPL corpus ID to text
    tvpl_corpus = {row["id"]: row["text"].strip() for row in tvpl_c_ds}
    print(f"Loaded {len(tvpl_corpus):,} TVPL corpus passages.")
    
    # Map TVPL query ID to gold corpus IDs
    tvpl_qrels = defaultdict(list)
    for row in tvpl_r_ds:
        tvpl_qrels[row["query-id"]].append(row["corpus-id"])
    print(f"Loaded qrels for {len(tvpl_qrels):,} TVPL queries.")
    
    # Filter TVPL queries that have valid qrels in corpus
    tvpl_queries_pool = []
    for row in tvpl_q_ds:
        qid = row["id"]
        q_text = row["text"].strip()
        gold_ids = tvpl_qrels.get(qid, [])
        # Ensure gold IDs exist in corpus and prefix them with 'tvpl_'
        valid_gold_ids = [f"tvpl_{gid}" for gid in gold_ids if gid in tvpl_corpus]
        if q_text and valid_gold_ids:
            tvpl_queries_pool.append({
                "id": f"tvpl_{qid}",
                "question": q_text,
                "relevant_ids": valid_gold_ids,
                "source": "tvpl"
            })
    print(f"TVPL queries pool size (with valid qrels): {len(tvpl_queries_pool):,}")
    
    # 2. Load ViCoQA
    print("\nLoading ViCoQA (HAT-FU/vicoqa_v1) train split...")
    vicoqa_train = load_dataset("HAT-FU/vicoqa_v1")["train"]
    
    # Flatten ViCoQA conversations
    # Each story is a passage. We generate a stable ID for it.
    # Questions is a string representing a list of questions.
    vicoqa_queries_pool = []
    vicoqa_corpus = {}
    
    for row in tqdm(vicoqa_train, desc="Parsing ViCoQA"):
        story_text = row["story"].strip()
        pid = _ctx_id("vicoqa", story_text)
        vicoqa_corpus[pid] = story_text
        
        try:
            questions_list = ast.literal_eval(row["questions"])
        except Exception:
            continue
            
        for idx, q_text in enumerate(questions_list):
            q_text = q_text.strip()
            if q_text:
                vicoqa_queries_pool.append({
                    "id": f"vicoqa_{pid}_{idx}",
                    "question": q_text,
                    "relevant_ids": [pid],
                    "source": "vicoqa"
                })
                
    print(f"Loaded {len(vicoqa_corpus):,} ViCoQA corpus passages.")
    print(f"ViCoQA queries pool size: {len(vicoqa_queries_pool):,}")
    
    print("\n====================================================")
    print("STEP 2: SAMPLING QUERIES FOR CHANNELS")
    print("====================================================")
    
    # Shuffle pools
    random.shuffle(tvpl_queries_pool)
    random.shuffle(vicoqa_queries_pool)
    
    # Sample clean TVPL (BM25)
    clean_tvpl = tvpl_queries_pool[:2000]
    # Sample TVPL for noise
    noise_tvpl_source = tvpl_queries_pool[2000:3000]
    
    # Sample clean ViCoQA (Dense)
    clean_vicoqa = vicoqa_queries_pool[:2000]
    # Sample ViCoQA for noise
    noise_vicoqa_source = vicoqa_queries_pool[2000:3000]
    
    print(f"Clean TVPL (BM25): {len(clean_tvpl)}")
    print(f"TVPL for noise: {len(noise_tvpl_source)}")
    print(f"Clean ViCoQA (Dense): {len(clean_vicoqa)}")
    print(f"ViCoQA for noise: {len(noise_vicoqa_source)}")
    
    print("\n====================================================")
    print("STEP 3: GENERATING NOISY QUERIES VIA OLLAMA")
    print("====================================================")
    
    # We want 2,000 noisy queries. We have 1,000 TVPL and 1,000 ViCoQA source queries.
    # We split them equally into 4 noise types: missing_tone, typo_telex, informal, code_switch.
    # So 500 queries per noise type (250 TVPL + 250 ViCoQA).
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    noise_types_list = ["missing_tone", "typo_telex", "informal", "code_switch"]
    noisy_queries = []
    
    # Prepare noise distribution batches
    noise_batches = {nt: [] for nt in noise_types_list}
    for i, q in enumerate(noise_tvpl_source):
        nt = noise_types_list[i % 4]
        noise_batches[nt].append(q)
    for i, q in enumerate(noise_vicoqa_source):
        nt = noise_types_list[i % 4]
        noise_batches[nt].append(q)
        
    for nt, batch in noise_batches.items():
        print(f"\nGenerating '{nt}' noise for {len(batch)} queries using Ollama (parallel)...")
        prompt_config = NOISE_TYPES[nt]
        
        def process_item(q, nt=nt, prompt_config=prompt_config):
            question = q["question"]
            try:
                noisy_text = _call_ollama(prompt_config, question)
                return {
                    "id": f"{q['id']}_{nt}",
                    "question": noisy_text,
                    "relevant_ids": q["relevant_ids"],
                    "source": f"{q['source']}_noisy_{nt}",
                    "original_question": question
                }
            except Exception:
                # Fallback to original
                return {
                    "id": f"{q['id']}_{nt}",
                    "question": question,
                    "relevant_ids": q["relevant_ids"],
                    "source": f"{q['source']}_noisy_{nt}_fallback",
                    "original_question": question
                }
                
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_item, q) for q in batch]
            for future in tqdm(as_completed(futures), total=len(batch), desc=f"Ollama {nt}"):
                noisy_queries.append(future.result())
                
    print(f"\nGenerated {len(noisy_queries)} noisy queries.")
    
    print("\n====================================================")
    print("STEP 4: SEMANTIC SIMILARITY VALIDATION VIA FPT")
    print("====================================================")
    
    # Validate noisy queries using FPT Embedding API
    # Cosine similarity must be >= 0.80 (relaxed slightly for training set to capture real semantic noise)
    # If similarity is < 0.80, we fall back to the original clean question.
    threshold = 0.80
    
    print("Embedding clean version of noisy queries...")
    clean_texts = [q["original_question"] for q in noisy_queries]
    noisy_texts = [q["question"] for q in noisy_queries]
    
    try:
        clean_embeddings = embed_texts(clean_texts)
        noisy_embeddings = embed_texts(noisy_texts)
        similarities = cosine_similarity(clean_embeddings, noisy_embeddings)
        
        passed_count = 0
        fallback_count = 0
        
        for idx, sim in enumerate(similarities):
            noisy_queries[idx]["similarity_score"] = float(sim)
            if sim < threshold:
                # Revert to original question if it drifted too much or failed completely
                noisy_queries[idx]["question"] = noisy_queries[idx]["original_question"]
                noisy_queries[idx]["source"] = noisy_queries[idx]["source"] + "_fallback_drift"
                fallback_count += 1
            else:
                passed_count += 1
                
        print("Validation finished:")
        print(f"  Passed: {passed_count} ({passed_count/len(noisy_queries)*100:.1f}%)")
        print(f"  Fell back: {fallback_count} ({fallback_count/len(noisy_queries)*100:.1f}%)")
        print(f"  Average similarity: {np.mean(similarities):.4f}")
    except Exception as e:
        print(f"Warning: FPT Embedding validation failed: {e}. Keeping all noisy queries directly.")
        
    print("\n====================================================")
    print("STEP 5: CONSTRUCTING CORPUS (PASSAGES) & MERGING DATA")
    print("====================================================")
    
    # Construct Merged train set
    # Clean TVPL (2000), Clean ViCoQA (2000), Noisy (2000)
    merged_train = []
    
    # We clean up temporary fields from noisy queries
    clean_noisy_queries = []
    for q in noisy_queries:
        clean_q = {
            "id": q["id"],
            "question": q["question"],
            "relevant_ids": q["relevant_ids"],
            "answers": []
        }
        clean_noisy_queries.append(clean_q)
        
    # Standard format clean queries
    clean_tvpl_fmt = [{"id": q["id"], "question": q["question"], "relevant_ids": q["relevant_ids"], "answers": []} for q in clean_tvpl]
    clean_vicoqa_fmt = [{"id": q["id"], "question": q["question"], "relevant_ids": q["relevant_ids"], "answers": []} for q in clean_vicoqa]
    
    merged_train.extend(clean_tvpl_fmt)
    merged_train.extend(clean_vicoqa_fmt)
    merged_train.extend(clean_noisy_queries)
    
    # Extract unique passages referenced in the merged train queries
    referenced_pids = set()
    for q in merged_train:
        for pid in q["relevant_ids"]:
            referenced_pids.add(pid)
            
    print(f"Total queries in train set: {len(merged_train):,}")
    print(f"Number of unique passages referenced: {len(referenced_pids):,}")
    
    # Construct corpus list
    merged_passages = []
    for pid in referenced_pids:
        if pid.startswith("tvpl_"):
            raw_id = pid.replace("tvpl_", "")
            merged_passages.append({"id": pid, "passage": tvpl_corpus[raw_id]})
        elif pid.startswith("vicoqa_"):
            merged_passages.append({"id": pid, "passage": vicoqa_corpus[pid]})
            
    # Save files
    passages_path = DATA_DIR / "multidomain_passages.jsonl"
    train_path = DATA_DIR / "multidomain_train.jsonl"
    
    with open(passages_path, "w", encoding="utf-8") as f:
        for r in merged_passages:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(train_path, "w", encoding="utf-8") as f:
        for r in merged_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nSaved {len(merged_passages):,} passages → {passages_path}")
    print(f"Saved {len(merged_train):,} training queries → {train_path}")
    print("\n====================================================")
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    main()
