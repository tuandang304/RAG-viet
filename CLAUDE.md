# CLAUDE.md — RAG_vie

Nghiên cứu Dynamic Hybrid RAG cho tiếng Việt. Mục tiêu: publish Q3 journal.  
Stack: Python 3.13 · uv · FPT AI Factory (embedding + LLM) · FAISS · BM25 · PyTorch MLP.

---

## Chạy lệnh

Luôn dùng `uv run` — không dùng `python` trực tiếp.

```bash
uv run python scripts/build_index.py --data-path data/processed/passages.jsonl
uv run python scripts/evaluate.py --qas-path data/processed/viaquad_dev.jsonl
uv run python main.py --query "Thủ đô của Việt Nam là gì?"
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
| `FPT_SPARSE_MODEL` | Tên model cho sparse endpoint (thường giống `FPT_EMBEDDING_MODEL`) |
| `FPT_LLM_MODEL` | Tên LLM trên FPT (mặc định: `Qwen3-32B`) |

`.env` đã được `.gitignore` — không commit lên git.

---

## Kiến trúc

```
Query
  │
  ├─ extract_features(query) → 7 Vietnamese-aware features
  │       └─ MLP → softmax → (a, b, c) weights
  │
  ├─ embed_query(query) → FPT embedding API
  │       └─ FAISS search → top-100 dense hits           (s_dense)
  │
  ├─ underthesea tokenize → BM25 search → top-100 hits   (s_bm25)
  │
  └─ BGE-M3 sparse encode → inverted index → top-100 hits (s_sparse)
          │
          └─ min-max normalize mỗi nguồn
                  └─ fused score = a·s_dense + b·s_bm25 + c·s_sparse
                          └─ top-10 → Qwen3:32B generator → answer
```

**Ba tín hiệu retrieval:**
| Signal | Nguồn | Đặc điểm |
|---|---|---|
| `s_dense` | FPT Vietnamese_Embedding → FAISS | Semantic similarity |
| `s_bm25` | underthesea + BM25Okapi | Classic term frequency |
| `s_sparse` | BGE-M3 lexical weights (SPLADE-style) | Learned sparse, khác BM25 |

**Cả ba tín hiệu đều dùng FPT AI Factory API** — không có model local.  
- Dense: `POST {FPT_BASE_URL}/embeddings`  
- Sparse: `POST {FPT_BASE_URL}/embed_sparse`  
- LLM: `POST {FPT_BASE_URL}/chat/completions`  

**Fusion weights** `(a, b, c) = softmax(MLP(features))` — MLP nhỏ (~2,700 params), dự đoán động theo từng query.  
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
│   └── vietnamese.py          # 7 features: diacritic, compound, english, tech, clause, question_word, length
├── fusion/
│   └── mlp.py                 # FusionMLP: Linear(7→64→32→3) + softmax → (a, b, c)
├── generator/
│   └── llm.py                 # generate() gọi Qwen3:32B qua FPT
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
| Dataset | Domain | Split dùng |
|---|---|---|
| UIT-ViQuAD 2.0 | Wikipedia | Train MLP + in-domain test |
| VIMQA | Wikipedia (multi-hop) | Zero-shot test |
| ViNewsQA | Tin tức VnExpress | Zero-shot test (cross-domain) |

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

**Evaluate** (so sánh các baseline):
```bash
# Dynamic MLP (3-way)
uv run python scripts/evaluate.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --mlp-path checkpoints/fusion_mlp.pt

# Fixed weight đều (1/3 mỗi tín hiệu)
uv run python scripts/evaluate.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --fixed-weights 0.33,0.33,0.34

# Dense only
uv run python scripts/evaluate.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --fixed-weights 1.0,0.0,0.0

# BM25 only
uv run python scripts/evaluate.py \
  --qas-path data/processed/viaquad_dev.jsonl \
  --index-dir indexes/viaquad \
  --fixed-weights 0.0,1.0,0.0
```

---

## Metrics

`evaluate.py` trả về **NDCG@10**, **MRR@10**, **Recall@100**.  
Kết quả lưu vào `results/` dạng JSON khi truyền `--output`.

Baseline cần vượt (theo thứ tự khó tăng dần):
1. BM25 only (`--fixed-weights 0.0,1.0`)
2. Dense only (`--fixed-weights 1.0,0.0`)
3. Fixed hybrid `0.5,0.5`
4. Best fixed-weight (tune trên dev set)
5. Dynamic MLP ← đây là contribution chính

---

## Lưu ý khi code

- Settings chỉ được load khi import `from rag_vie.config import settings` — không instantiate ở `__init__.py` vì sẽ fail nếu không có `.env`.
- FAISS index dùng **Inner Product** sau khi L2-normalize → tương đương cosine similarity.
- BM25 score không có upper bound → bắt buộc min-max normalize trước khi fuse với dense score.
- underthesea `word_tokenize(text, format="text")` trả về string với từ ghép nối bằng `_` (ví dụ: `học_sinh`).
- MLP `output_dim=3` cho three-way fusion `(a, b, c)` → `(w_dense, w_bm25, w_sparse)`.
- `SparseRetriever` gọi FPT endpoint `{FPT_BASE_URL}/embed_sparse`. Nếu FPT dùng format response khác, chỉnh hàm `_parse_sparse_response()` trong `retrieval/sparse.py`.
- Sparse inverted index lưu dạng `dict[token_id, list[(doc_id, weight)]]` — với corpus ~100K doc chiếm khoảng 2-4GB RAM. Nếu quá lớn, cắt bớt top-k token per doc khi build.
