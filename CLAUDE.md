# CLAUDE.md — RAG_vie

Nghiên cứu Dynamic Hybrid RAG cho tiếng Việt. Mục tiêu: publish Q3 journal.  
Stack: Python 3.13 · uv · FPT AI Factory (embedding + LLM) · FAISS · BM25 · BGE-M3 (PyTorch) · Keras/TensorFlow MLP.

---

## Chạy lệnh

Luôn dùng `uv run` — không dùng `python` trực tiếp.

```bash
uv run python scripts/build_index.py --data-path data/processed/viaquad_passages.jsonl --index-dir indexes/viaquad
uv run python scripts/evaluate_all.py --qas-path data/processed/viaquad_dev.jsonl --index-dir indexes/viaquad --mlp-path checkpoints/fusion_mlp_aug.keras
uv run python main.py --index-dir indexes/viaquad --mlp-path checkpoints/fusion_mlp_aug.keras --query "Thủ đô của Việt Nam là gì?"
uv run jupyter lab
```

Cài thêm package:
```bash
uv add <package>          # production
uv add --dev <package>    # dev/notebook only
uv sync                   # đồng bộ sau khi sửa pyproject.toml
```

---

## Cấu hình (.env)

Copy `.env.example` → `.env` và điền:

| Biến | Mô tả |
|---|---|
| `FPT_API_KEY` | API key từ FPT AI Factory dashboard |
| `FPT_BASE_URL` | Base URL của FPT AI Factory (OpenAI-compatible) |
| `FPT_EMBEDDING_MODEL` | Tên model embedding dense trên FPT |
| `FPT_LLM_MODEL` | Tên LLM trên FPT (mặc định: `Qwen3-32B`) |

`.env` đã được `.gitignore` — không commit lên git.

---

## Kiến trúc

```
Query
  │
  ├─ embed_query → FPT API → FAISS search → top-100 dense hits   (s_dense)
  ├─ underthesea tokenize → BM25 search → top-100 hits           (s_bm25)
  └─ BGE-M3 sparse encode → inverted index → top-100 hits        (s_sparse)
          │  min-max normalize mỗi nguồn
          ▼
  features = extract_features(query)            # 13 query features (ngôn ngữ)
           + extract_signal_features(scores)    # 7 signal features (đồng thuận giữa các nguồn)
          │
          └─ MLP(20 → 64 → 32 → 3) → softmax → (a, b, c) weights
                  └─ fused score = a·s_dense + b·s_bm25 + c·s_sparse
                          └─ top-10 → Qwen3:32B generator → answer
```

> **Thứ tự quan trọng:** retrieval chạy **trước** để tính được signal features (độ chồng lấn /
> độ nhọn / đồng thuận top-1 giữa 3 nguồn), rồi MLP mới ra trọng số. Xem `pipeline.py`.

**Ba tín hiệu retrieval:**
| Signal | Nguồn | Đặc điểm |
|---|---|---|
| `s_dense` | FPT Vietnamese_Embedding → FAISS (1024-dim, L2-norm, Inner Product) | Semantic similarity |
| `s_bm25` | underthesea word segmentation + BM25Okapi | Classic term frequency |
| `s_sparse` | BGE-M3 (local, BAAI/bge-m3) → inverted index (lexical weights) | Learned sparse retrieval |

**API calls:**
- Dense embedding: `POST {FPT_BASE_URL}/embeddings`
- LLM generation: `POST {FPT_BASE_URL}/chat/completions`
- Sparse: local inference only (BGE-M3 via FlagEmbedding, ~570 MB download on first run)

**Fusion weights** `(a, b, c) = softmax(MLP(features))` — MLP nhỏ (input 20 = 13 query + 7 signal features, ~3,715 params, Keras/TensorFlow), dự đoán động theo từng query.  
Khi MLP chưa train, weights ≈ `(0.33, 0.33, 0.33)`.

---

## Cấu trúc source

```
src/rag_vie/
├── config.py                  # pydantic-settings, đọc từ .env
├── retrieval/
│   ├── embedder.py            # gọi FPT embedding API (batch=32)
│   ├── bm25.py                # BM25Okapi + underthesea tokenizer
│   ├── dense.py               # DenseRetriever (FAISS IndexFlatIP, L2-norm)
│   ├── sparse.py              # SparseRetriever (BGE-M3 lexical weights, inverted index)
│   └── hybrid.py              # HybridRetriever: 3-way normalize + weighted sum
├── features/
│   ├── vietnamese.py          # 13 query features: diacritic, compound, english, tech, clause, question_word, length, oov, avg_idf, max_idf, digit, proper_noun, acronym
│   ├── signal.py              # 7 signal features từ score đã chuẩn hoá: overlap top-k (3 cặp), sharpness (3 nguồn), top1_agreement
│   └── combined.py            # combine() + ALL_FEATURE_NAMES (query ++ signal) — input của MLP
├── fusion/
│   └── mlp.py                 # FusionMLP (Keras): Dense(64)→LayerNorm→GELU→Dropout→Dense(32)→LayerNorm→GELU→Dropout→Dense(3,softmax)
├── generator/
│   └── llm.py                 # generate() gọi Qwen3:32B qua FPT
├── utils/
│   ├── metrics.py            # ndcg/mrr/map/recall/hit_at_1 + min_max_normalize (dùng chung train + eval)
│   ├── fusion.py             # fuse_scores: a·dense + b·bm25 + c·sparse
│   └── text.py               # remove_diacritics (mô phỏng gõ thiếu dấu)
└── pipeline.py                # RAGPipeline.run(query) → RAGResult
```

---

## Data format

Tất cả dataset chuẩn hoá về JSONL, mỗi dòng là một JSON object.

**Passages file** (dùng để build index):
```json
{"id": "viaquad_0001", "passage": "Hà Nội là thủ đô của Việt Nam..."}
```

**QA file** (dùng để evaluate):
```json
{"question": "Thủ đô của Việt Nam là gì?", "relevant_ids": ["viaquad_0001", "viaquad_0002"]}
```

**Datasets:**
| Dataset | HF ID | Domain | Split dùng |
|---|---|---|---|
| UIT-ViQuAD 2.0 | `taidng/UIT-ViQuAD2.0` | Wikipedia | Train MLP + in-domain test |
| DANGDOCAO | `DANGDOCAO/GeneratingQuestions` | Pháp lý / Hành chính (736 sub-domain) | Zero-shot cross-domain test |

Tải dataset: `uv run python scripts/download_data.py`

---

## Scripts

**Build index** (chạy một lần sau khi có passages JSONL):
```bash
uv run python scripts/build_index.py \
  --data-path data/processed/viaquad_passages.jsonl \
  --index-dir indexes/viaquad
# Tự tạo: indexes/viaquad/index.faiss, bm25.pkl, sparse.pkl
# Lần đầu sẽ download BAAI/bge-m3 (~570MB)
```

**Train MLP** (soft-label; `--loss kl` để nới rộng phân phối trọng số):
```bash
uv run python scripts/train_mlp.py \
  --qas-path data/processed/viaquad_train_aug.jsonl \
  --index-dir indexes/viaquad \
  --output checkpoints/fusion_mlp_aug.keras \
  --emb-cache checkpoints/train_aug_embeddings.npy
  # tùy chọn: --loss kl | --temperature 0.1 | --hard-label (ablation §5.6)
```

**Evaluate** (so sánh các baseline — tất cả trong một lệnh):
```bash
# Dynamic MLP (3-way) + tất cả baselines
uv run python scripts/evaluate_all.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --mlp-path checkpoints/fusion_mlp_aug.keras \
  --output results/viaquad_dev.json

# Cross-domain zero-shot (train ViQuAD → test DANGDOCAO)
uv run python scripts/evaluate_all.py \
  --qas-path data/processed/dangdocao_test.jsonl \
  --index-dir indexes/dangdocao \
  --mlp-path checkpoints/fusion_mlp_aug.keras \
  --output results/dangdocao_test.json

# Không có sparse index (2-way fallback)
uv run python scripts/evaluate_all.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --no-sparse
```

Methods được evaluate: `mlp`, `fixed_equal` (1/3,1/3,1/3), `dense_bm25` (0.5,0.5,0), `dense`, `bm25`, `sparse`, `rrf` (Reciprocal Rank Fusion, parameter-free), `oracle` (per-query best simplex point — trần trên / headroom). `evaluate_all.py` batch pre-embed toàn bộ query trước vòng lặp (không gọi API từng query).

---

## Metrics

`evaluate_all.py` trả về 4 nhóm metrics:
1. **Retrieval quality**: NDCG@10, MRR@10, MAP@10, Recall@10, Recall@100, Hit@1
2. **Statistical significance**: Paired t-test + Wilcoxon, 95% bootstrap CI (2000 resamples)
3. **Efficiency**: số param MLP, latency suy luận MLP (μs), latency retrieval+fusion p50/p95 + throughput (q/s, không tính embedding API), kích thước index (MB).
4. **Weight interpretability**: Entropy H, Pearson correlations (diacritic↔w_dense, compound↔w_bm25, english↔w_sparse), stratified analysis (11 strata)

Kết quả lưu vào `results/` dạng JSON khi truyền `--output`.

Baseline cần vượt (theo thứ tự khó tăng dần):
1. BM25 only
2. Dense only
3. Sparse only (BGE-M3)
4. RRF (parameter-free hybrid)
5. Fixed hybrid equal (1/3, 1/3, 1/3)
6. Dynamic MLP ← đây là contribution chính
7. `oracle` (per-query best simplex) = trần trên, dùng để đo headroom còn lại của MLP

---

## Lưu ý khi code

- Settings chỉ được load khi import `from rag_vie.config import settings` — không instantiate ở `__init__.py` vì sẽ fail nếu không có `.env`.
- FAISS index dùng **Inner Product** sau khi L2-normalize → tương đương cosine similarity.
- BM25 score không có upper bound → bắt buộc min-max normalize trước khi fuse với dense score.
- underthesea `word_tokenize(text, format="text")` trả về string với từ ghép nối bằng `_` (ví dụ: `học_sinh`).
- MLP `output_dim=3` cho three-way fusion `(a, b, c)` → `(w_dense, w_bm25, w_sparse)`. `input_dim` = `len(ALL_FEATURE_NAMES)` = 20 (13 query + 7 signal); train_mlp tự suy từ `X.shape[1]` nên đổi feature set không cần sửa MLP.
- **Signal features cần điểm retrieval** → trong train/eval phải tính `extract_signal_features` từ score đã `min_max_normalize` rồi `combine()` với query features; thứ tự concat cố định (query trước, signal sau) để train/eval/inference khớp nhau. Đừng đảo thứ tự.
- **FusionMLP là Keras/TensorFlow** (không phải PyTorch). `FusionMLP.save()` ghi định dạng Keras → đặt đuôi `.keras`. `torch` vẫn cần cho BGE-M3.
- Test thuần (không cần `.env`/API): `uv run pytest tests/` — bao metrics, fusion, signal, features. `tests/test_features.py` tự skip nếu underthesea/torch không load được.
- Train MLP chạy phase thu thập dữ liệu (FAISS) rồi **spawn subprocess riêng** để fit Keras → tránh xung đột OMP giữa FAISS-MKL và TensorFlow/PyTorch.
- Soft-label grid: `_N=20` trong `train_mlp.py` → simplex 3-way **231 điểm** (step 0.05); 2-way (c=0) là 21 điểm.
- `retrieval/sparse.py` dùng BGE-M3 local (BAAI/bge-m3 qua FlagEmbedding) — **không** phải FPT API. Lần đầu chạy sẽ download ~570 MB.
- OMP deadlock (FAISS + PyTorch/TF) trên **macOS và Windows** (Windows biểu hiện là exit `0xC0000005`): luôn set `KMP_DUPLICATE_LIB_OK=TRUE`, import `pyarrow` sớm, và khởi tạo PyTorch (BGE-M3) **trước** FAISS.
- Metric + min-max normalize + fuse dùng chung từ `rag_vie.utils.metrics` và `rag_vie.utils.fusion` — **không** copy lại trong scripts (đảm bảo train và eval tính giống hệt nhau).
- Scores từ dense, BM25, và sparse đều được min-max normalize trước khi fuse — quan trọng vì BM25/sparse không có upper bound.
- Train MLP mới: `uv run python scripts/train_mlp.py --output checkpoints/fusion_mlp_aug.keras --emb-cache checkpoints/train_aug_embeddings.npy`
