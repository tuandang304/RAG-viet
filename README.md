# Dynamic Hybrid RAG for Vietnamese

Adaptive retrieval-augmented generation for Vietnamese that replaces fixed fusion weights with a lightweight per-query MLP. The MLP takes eight Vietnamese-aware linguistic features and regresses NDCG@10 for 66 points on the weight simplex (Grid NDCG Predictor); the argmax point gives `(w_dense, w_bm25, w_sparse)`, enabling dynamic three-way fusion across dense (FPT), BM25, and BGE-M3 sparse signals.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.13 |
| [uv](https://docs.astral.sh/uv/) | latest |
| FPT AI Factory account | — |

> **Note:** All Python commands must be run through `uv run`. Do not use `python` directly.

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd RAG_vie

# 2. Install all dependencies (creates .venv automatically)
uv sync
```

---

## Environment Setup

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set the following variables:

```env
FPT_API_KEY=your_api_key_here          # Get from https://factory.fpt.ai
FPT_BASE_URL=https://mkp-api.fptcloud.com/v1
FPT_EMBEDDING_MODEL=vietnamese-embedding
FPT_LLM_MODEL=Qwen3-32B

# Retrieval config (defaults work fine)
TOP_K_DENSE=100
TOP_K_BM25=100
TOP_K_FINAL=10
EMBEDDING_DIM=1024
```

> `.env` is gitignored — never commit it.

---

## Full Setup Workflow

Run steps in order on a fresh machine.

### Step 1 — Download Datasets

Downloads UIT-ViQuAD 2.0 and DANGDOCAO from HuggingFace and converts them to JSONL:

```bash
uv run python scripts/download_data.py
```

Outputs to `data/processed/`:

| File | Description |
|------|-------------|
| `viaquad_passages.jsonl` | 5,317 Wikipedia passages |
| `viaquad_train/dev/test.jsonl` | 28,454 / 3,814 / 7,301 QA pairs |
| `dangdocao_passages.jsonl` | 37,239 legal/administrative passages |
| `dangdocao_train/dev/test.jsonl` | ~35K / ~4.4K / ~4.4K QA pairs |

Download a single dataset only:

```bash
uv run python scripts/download_data.py --datasets viaquad
uv run python scripts/download_data.py --datasets dangdocao
```

---

### Step 2 — Augment Training Data

Generates diacritic-removed query variants for MLP training (noise robustness):

```bash
uv run python scripts/augment_data.py \
    --input data/processed/viaquad_train.jsonl \
    --noise-ratio 0.3 --seed 42

uv run python scripts/augment_data.py \
    --input data/processed/viaquad_dev.jsonl \
    --noise-ratio 0.3 --seed 42
```

Each run produces two files alongside the input:

| File | Description |
|---|---|
| `*_aug.jsonl` | Original + 30% diacritic-removed copies — used for MLP training |
| `*_noisy.jsonl` | 100% diacritic-removed — used for robustness evaluation |

---

### Step 3 — Build Indexes

Builds FAISS dense, BM25, and BGE-M3 sparse indexes. **Requires an active `.env`.**

```bash
# ViQuAD index
uv run python scripts/build_index.py \
    --data-path data/processed/viaquad_passages.jsonl \
    --index-dir indexes/viaquad

# DANGDOCAO index
uv run python scripts/build_index.py \
    --data-path data/processed/dangdocao_passages.jsonl \
    --index-dir indexes/dangdocao

# Skip sparse index (no BGE-M3 download required)
uv run python scripts/build_index.py \
    --data-path data/processed/viaquad_passages.jsonl \
    --index-dir indexes/viaquad \
    --no-sparse
```

Each index directory will contain:

| File | Description |
|---|---|
| `index.faiss` | FAISS `IndexFlatIP` with L2-normalized 1024-dim embeddings |
| `bm25.pkl` | BM25Okapi model with underthesea word tokenization |
| `sparse.pkl` | BGE-M3 inverted index (lexical weights) — ~570 MB model downloaded on first run |
| `meta.json` | Passage IDs and metadata |

> **API cost note:** Building indexes calls the FPT embedding API for every passage. ViQuAD (~5K passages) and DANGDOCAO (~37K passages) will incur API costs accordingly.
>
> **BGE-M3 note:** The sparse index uses `BAAI/bge-m3` via FlagEmbedding (local inference, no API cost). The model (~570 MB) is downloaded automatically on first run.

---

### Step 4 — Train the Fusion MLP

Trains the dynamic fusion MLP on ViQuAD training queries using soft-label supervision:

```bash
uv run python scripts/train_mlp.py \
    --qas-path data/processed/viaquad_train_aug.jsonl \
    --index-dir indexes/viaquad \
    --output checkpoints/fusion_mlp_3way.keras \
    --emb-cache checkpoints/train_aug_embeddings.npy
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--max-samples` | 5000 | Number of training queries (randomly sampled) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--temperature` | 0.3 | Softmax temperature for soft-label construction (lower = sharper) |
| `--emb-cache` | None | Path to cache query embeddings — avoids re-calling the API on retrain |
| `--init-from` | None | Fine-tune from an existing checkpoint instead of training from scratch |

Fine-tuning from an existing checkpoint:

```bash
uv run python scripts/train_mlp.py \
    --init-from checkpoints/fusion_mlp_3way.keras \
    --emb-cache checkpoints/train_aug_embeddings.npy \
    --lr 1e-4 --epochs 50
```

> **macOS / ARM64:** Training is automatically spawned in a subprocess to avoid an OMP deadlock between FAISS and PyTorch. This is handled internally — no action needed.

---

### Step 5 — Evaluate

Run the unified evaluation script to compute all four metric groups:

```bash
# In-domain: ViQuAD dev set
uv run python scripts/evaluate_all.py \
    --qas-path data/processed/viaquad_dev.jsonl \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --output results/eval_all_dev.json

# In-domain: ViQuAD test set (held-out)
uv run python scripts/evaluate_all.py \
    --qas-path data/processed/viaquad_test.jsonl \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --output results/eval_all_test.json

# Diacritic robustness (dev queries with all tone marks removed)
uv run python scripts/evaluate_all.py \
    --qas-path data/processed/viaquad_dev_noisy.jsonl \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --output results/eval_all_dev_noisy.json

# Cross-domain zero-shot: MLP trained on ViQuAD, tested on DANGDOCAO
uv run python scripts/evaluate_all.py \
    --qas-path data/processed/dangdocao_test.jsonl \
    --index-dir indexes/dangdocao \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --output results/eval_all_cross.json
```

The script evaluates six methods simultaneously — `mlp`, `fixed_equal` (1/3,1/3,1/3), `dense_bm25` (0.5,0.5,0), `dense`, `bm25`, `sparse` — and reports:

| Metric group | What you get |
|---|---|
| **Retrieval** | NDCG@10, MRR@10, MAP@10, Recall@10, Recall@100, Hit@1 per method |
| **Significance** | Paired t-test p, Wilcoxon p, 95% bootstrap CI for MLP vs each baseline |
| **Efficiency** | MLP param count, index sizes (MB), latency p50/p95, throughput (q/s) |
| **Weight analysis** | Entropy H, weight distributions, Pearson correlations (diacritic↔w_dense, compound↔w_bm25, english↔w_sparse) |
| **Stratified** | NDCG@10 and mean weights per query-feature stratum (11 strata: diac_low/mid/high, comp, eng, length, clause) |

Each query calls the FPT embedding API exactly once; all four methods share the same dense/BM25 hits.

---

### Step 6 — End-to-end QA Evaluation (optional)

Evaluates full RAG quality using RAGAS metrics with Qwen3-32B as judge. **Makes LLM API calls for each sample — use a small `--n-samples`.**

```bash
# Clean queries
uv run python scripts/evaluate_ragas.py \
    --qas-path data/processed/viaquad_dev.jsonl \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --n-samples 50 \
    --output results/ragas_clean.json

# Diacritic-removed queries
uv run python scripts/evaluate_ragas.py \
    --qas-path data/processed/viaquad_dev_noisy.jsonl \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --n-samples 50 \
    --output results/ragas_noisy.json
```

RAGAS metrics (judged by Qwen3-32B): Context Precision, Context Recall, Faithfulness, Answer Relevancy.

---

## Run a Single Query

```bash
uv run python main.py \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --query "Thủ đô của Việt Nam là gì?"

# Skip LLM generation, return retrieved passages only
uv run python main.py \
    --index-dir indexes/viaquad \
    --mlp-path checkpoints/fusion_mlp_3way.keras \
    --query "Thủ đô của Việt Nam là gì?" \
    --no-generate
```

---

## Project Structure

```
RAG-viet/
├── src/rag_vie/                # Installable package — all library code lives here
│   ├── config.py               # pydantic-settings — reads from .env
│   ├── pipeline.py             # RAGPipeline: query → answer (end-to-end) + CLI entry
│   ├── retrieval/
│   │   ├── embedder.py         # FPT embedding API, batch size 32
│   │   ├── dense.py            # DenseRetriever — FAISS IndexFlatIP, L2-norm
│   │   ├── bm25.py             # BM25Retriever — BM25Okapi + underthesea
│   │   ├── sparse.py           # SparseRetriever — BGE-M3 local, inverted index
│   │   └── hybrid.py           # HybridRetriever — 3-way min-max normalize + weighted sum
│   ├── features/
│   │   ├── vietnamese.py       # 8 Vietnamese-aware query features (heuristic)
│   │   └── neural.py           # NeuralFeatureExtractor — frozen PhoBERT + projection head
│   ├── fusion/
│   │   └── mlp.py              # FusionMLP (Keras) — Grid NDCG Predictor, 66-point simplex argmax
│   ├── generator/
│   │   └── llm.py              # Qwen3-32B via FPT chat/completions
│   ├── datagen/                # LLM-based noise generation (Ollama + FPT validation)
│   │   ├── prompts.py          # 4 noise types: missing_tone, typo_telex, informal, code_switch
│   │   ├── generate_noise.py   # python -m rag_vie.datagen.generate_noise
│   │   ├── validate.py         # Semantic-similarity validation of generated noise
│   │   └── run_all.py          # python -m rag_vie.datagen.run_all
│   └── utils/
│       ├── metrics.py          # Shared ranking metrics + min-max normalization (train & eval)
│       ├── fusion.py           # fuse_scores — weighted 4-way score fusion
│       └── text.py             # Text normalization helpers
│
├── scripts/                    # Pipeline steps (run in order on a fresh machine)
│   ├── download_data.py        # Step 1 — HuggingFace → JSONL
│   ├── augment_data.py         # Step 2 — diacritic-removal augmentation
│   ├── build_index.py          # Step 3 — FAISS + BM25 + sparse index builder
│   ├── prepare_multidomain_dataset.py  # Step 3b — multi-domain MLP training set (TVPL + ViCoQA + noise)
│   ├── train_mlp.py            # Step 4 — grid soft-label MLP training
│   ├── train_feature_extractor.py      # Optional — distill heuristic features into PhoBERT head
│   ├── evaluate_all.py         # Step 5 — unified evaluation (4 metric groups)
│   └── evaluate_ragas.py       # Step 6 — end-to-end RAGAS evaluation
│
├── experiments/                # Paper experiment runners (wrap scripts/ with fixed configs)
├── tests/                      # pytest unit tests — uv run pytest
│
├── data/
│   ├── processed/              # JSONL datasets (gitignored, generated by download_data.py)
│   └── generated/              # LLM-generated noisy benchmarks (gitignored)
├── indexes/                    # FAISS + BM25 + sparse indexes (gitignored)
├── checkpoints/                # MLP .keras checkpoints and embedding caches (gitignored)
├── results/                    # Evaluation JSON outputs (gitignored)
│
├── docs/
│   ├── research_paper.md       # Paper draft (target: Q3 2026 journal)
│   ├── proposal/               # Research proposal (md + tex)
│   └── planning/               # Working plans and notes
├── main.py                     # Single-query CLI entry point (= uv run rag-vie)
├── .env.example                # Environment variable template
└── pyproject.toml              # Deps + pytest/ruff config
```

---

## Data Formats

**Passages JSONL** — input to `build_index.py`:
```json
{"id": "viaquad_abc123", "passage": "Hà Nội là thủ đô của Việt Nam..."}
```

**QA JSONL** — input to `train_mlp.py` and `evaluate_all.py`:
```json
{
  "id": "q001",
  "question": "Thủ đô của Việt Nam là gì?",
  "relevant_ids": ["viaquad_abc123"],
  "answers": ["Hà Nội"]
}
```

---

## Datasets

| Dataset | HuggingFace ID | Domain | Size |
|---|---|---|---|
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` | Wikipedia (Vietnamese) | 5,317 passages · 39.5K QA pairs |
| DANGDOCAO | `DANGDOCAO/GeneratingQuestions` | Legal / Administrative (736 sub-domains) | 37,239 passages · 43.9K QA pairs |

Both datasets download automatically via HuggingFace `datasets` — no manual registration required.

---

## Development

### Run tests

```bash
uv run pytest                  # full suite (loads TensorFlow — ~2 min)
uv run pytest -m "not slow"    # fast unit tests only (~1 min)
```

Tests live in `tests/` and cover the pure logic (feature extraction, fusion math, BM25, simplex grids) without calling the FPT API — they run fine without a real `.env`.

### Lint

```bash
uv run ruff check src tests scripts experiments          # check
uv run ruff check src tests scripts experiments --fix    # auto-fix
```

Both are configured in `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]`). Please keep both green before committing.

### Adding packages

```bash
uv add <package>          # production dependency
uv add --dev <package>    # dev / notebook only
uv sync                   # sync after editing pyproject.toml manually
```

---

## Citation

```bibtex
@article{ragvie2026,
  title   = {Dynamic Hybrid Retrieval-Augmented Generation for Vietnamese:
             Adaptive Fusion of Dense and Sparse Signals via a Lightweight MLP},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026}
}
```
