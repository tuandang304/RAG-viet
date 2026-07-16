# Kế hoạch chi tiết: Thu thập, Thống kê và Xây dựng Dataset Huấn luyện MLP

Dưới đây là kết quả thống kê chi tiết sau khi tải thành công các bộ dữ liệu và đề xuất phương án sinh nhiễu để tạo tập huấn luyện tối ưu nhất cho MLP Router.

---

## 1. Kết quả thống kê dữ liệu thực tế (HuggingFace)

Do bộ dữ liệu gốc `uitnlp/vicoqa` và `ura-hcmut/Vietnamese-Customer-Support-QA` bị khóa (gated) trên HugingFace Hub, chúng tôi đã tìm kiếm các bản mirror cộng đồng mở công khai và thu được kết quả như sau:

| Bộ dữ liệu | Nguồn tải công khai | Số lượng Văn bản (Passages) | Số lượng Câu hỏi (Queries) | Đặc điểm |
|---|---|---|---|---|
| **BM25 Dataset** (Zalo Legal) | `nrl-ai/vn-rag-bench` (file `vn_legal_zalo_full.json`) | **61,068** | **788** | Lấy từ benchmark Zalo AI Challenge 2021, chỉ có 788 câu hỏi test được gán nhãn sẵn. |
| **BM25 Dataset** (TVPL Legal) | `GreenNode/TVPL-Retrieval-VN` (chuyên biệt pháp lý) | **10,576** | **9,985** | **Giải pháp mở rộng:** Bộ dữ liệu pháp lý cực kỳ đồ sộ với 9,985 câu hỏi và đầy đủ qrels (nhãn liên kết). |
| **Dense Dataset** (ViCoQA) | `HAT-FU/vicoqa_v1` (public mirror) | **4,848** (stories) | **60,064** (56,008 train / 4,056 val) | Hỏi đáp hội thoại dạng y tế/sức khỏe, paraphrase cao. |
| **Sparse Dataset** (CSConDa) | *Không khả dụng public* | - | - | Gated, không có mirror public hoạt động ổn định. |

---

## 2. Giải pháp mở rộng kênh BM25 (Tại sao Zalo Legal chỉ có 800 câu?)

Lý do tập Zalo Legal từ `vn-rag-bench` chỉ có 788 câu là vì đây là file **Evaluation Fixture** được trích xuất từ tập test của cuộc thi Zalo AI Challenge 2021 nhằm mục đích đánh giá (benchmark), không phải để huấn luyện.

**Đề xuất tối ưu:** 
Chúng ta sẽ chuyển sang sử dụng bộ dữ liệu **`GreenNode/TVPL-Retrieval-VN` (Thư Viện Pháp Luật)** cho kênh BM25:
*   Bộ dữ liệu này có **9,985 câu hỏi thực tế** và **10,576 văn bản luật**.
*   Dữ liệu đã được định dạng sạch sẽ dưới 3 cấu hình: `queries` (id + text), `corpus` (id + text), và `default` (qrels mapping `query-id` và `corpus-id`).
*   Giúp tăng số lượng dữ liệu huấn luyện cho kênh BM25 lên tương đương với Dense (ViCoQA) và hỗ trợ sinh nhiễu Sparse chất lượng cao hơn.

---

## 3. Phương án xây dựng Dataset huấn luyện MLP cân bằng (~6,000 queries)

Chúng ta sẽ tăng quy mô tập huấn luyện MLP lên **~6,000 câu hỏi** (cân bằng 1:1:1 giữa 3 kênh):

```
                         [Tập Huấn Luyện MLP Hỗn Hợp]
                                 (~6,000 queries)
                                        |
         +------------------------------+------------------------------+
         |                              |                              |
   Domain BM25 (Clean)            Domain DENSE (Clean)           Domain SPARSE (Noisy)
   (~2,000 queries TVPL)          (~2,000 queries ViCoQA)        (~2,000 queries Noisy)
                                                                       |
                                                +----------------------+----------------------+
                                                |                      |                      |
                                            50% Từ TVPL            50% Từ ViCoQA              |
                                          (~1,000 queries)       (~1,000 queries)             |
                                                |                      |                      |
                                                +-----------+----------+                      |
                                                            |                                 |
                                             LLM Noise Pipeline (Qwen3-14B)                   |
                                                            |                                 |
                                        +-------------------+-------------------+             |
                                        |                   |                   |             |
                                  Missing Tone         Telex Typo           Informal     Code-Switch
                                   (500 q)              (500 q)             (500 q)       (500 q)
```

### Chi tiết các thành phần:
1. **Thành phần BM25-dominant (~2,000 câu):** 
   - Trích xuất ngẫu nhiên 2,000 câu hỏi sạch từ `GreenNode/TVPL-Retrieval-VN`.
2. **Thành phần Dense-dominant (~2,000 câu):**
   - Sample ngẫu nhiên 2,000 câu hỏi sạch từ `HAT-FU/vicoqa_v1`.
3. **Thành phần Sparse-dominant (~2,000 câu bị nhiễu):**
   - Trích xuất ngẫu nhiên 1,000 câu từ TVPL và 1,000 câu từ ViCoQA (không trùng với tập clean ở trên).
   - Chạy qua pipeline **LLM Noise Generator** để sinh nhiễu ngẫu nhiên với tỉ lệ đều nhau (500 câu cho mỗi loại nhiễu: `missing_tone`, `typo_telex`, `informal`, `code_switch`).

---

## 4. Kế hoạch hành động tiếp theo

1. **Bước 1: Viết script trích xuất và định dạng chuẩn hóa (`scripts/prepare_raw_data.py`)**
   - Tải và trích xuất các câu hỏi sạch từ TVPL và ViCoQA, chuyển về cùng một format chuẩn: `{"id": str, "question": str, "relevant_ids": list[str], "source": str}`.
   - Lưu trữ các tập clean này vào `data/processed/`.

2. **Bước 2: Chạy sinh nhiễu LLM cho tập Sparse**
   - Áp dụng pipeline Ollama (Qwen3-14B) để chuyển đổi 2,000 câu hỏi clean thành các câu hỏi nhiễu tương ứng.
   - Thẩm định độ tương đồng ngữ nghĩa bằng FPT Embedding (ngưỡng tương đồng $\ge 0.85$ như cũ).

3. **Bước 3: Gộp dữ liệu & Chạy Grid Search tìm Nhãn tối ưu**
   - Gộp cả 3 tập trên lại thành `data/processed/mlp_train_multidomain.jsonl` (~6,000 câu hỏi).
   - Tiến hành chạy truy hồi trên 3 kênh (Dense, BM25, Sparse) cho 66 điểm simplex trên toàn bộ tập train này để tìm nhãn NDCG thực tế cho từng câu hỏi.

4. **Bước 4: Huấn luyện MLP Router với Expected NDCG Loss**
   - Huấn luyện MLP với hàm Loss mới để dự đoán phân phối xác suất trên 66 điểm simplex và tối đa hóa Expected NDCG.
