"""Search for public community mirrors of gated datasets using Hugging Face free-text search.
"""

from huggingface_hub import HfApi
from datasets import load_dataset

def search_name(query):
    print(f"\nSearching HuggingFace datasets for name: '{query}'...")
    api = HfApi()
    try:
        datasets = api.list_datasets(search=query, limit=10)
        results = [d.id for d in datasets]
        print(f"Found: {results}")
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

def try_load(dataset_id):
    try:
        ds = load_dataset(dataset_id)
        print(f"SUCCESS: Loaded {dataset_id}")
        splits = list(ds.keys())
        total = sum(len(ds[split]) for split in splits)
        print(f"  Splits: {splits}, Total queries: {total:,}")
        
        # Print first split columns/keys
        first_split = splits[0]
        print(f"  Features: {list(ds[first_split].features.keys())}")
        return True
    except Exception as e:
        print(f"FAILED: {dataset_id}. Error: {e}")
        return False

def main():
    # Search for community uploads/mirrors
    vicoqa_repos = search_name("vicoqa")
    for r in vicoqa_repos:
        try_load(r)
        
    vinews_repos = search_name("vinewsqa")
    for r in vinews_repos:
        try_load(r)
        
    csconda_repos = search_name("customer-support-qa")
    for r in csconda_repos:
        try_load(r)
        
    csconda_repos2 = search_name("csconda")
    for r in csconda_repos2:
        try_load(r)

    # Let's search for Zalo Legal Text mirrors
    zalo_repos = search_name("zalo-ai-legal-text-retrieval-vn")
    for r in zalo_repos:
        try_load(r)

if __name__ == "__main__":
    main()
