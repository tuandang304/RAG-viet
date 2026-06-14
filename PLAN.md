# PLAN.md — Kế hoạch cải tiến idea nghiên cứu (Dynamic Hybrid RAG cho tiếng Việt)

Tài liệu này đề xuất hướng nâng cấp **ý tưởng khoa học** của paper (không chỉ refactor code).
Mục tiêu: biến đóng góp "dynamic fusion" từ một cải thiện *biên* (+0.0028 NDCG@10 so với
fixed-equal) thành một đóng góp *chắc chắn, có cơ chế rõ ràng, và bán được cho reviewer Q3*.

> Nguồn chuẩn cho trạng thái hiện tại: `docs/research_paper.md` (§5 = kết quả, §6 = limitations),
> `CLAUDE.md`, và code trong `src/rag_vie/`.

---

## 0. Tiền đề bắt buộc — đồng bộ lại thực nghiệm

Code đã lệch khỏi cấu hình sinh ra số liệu trong paper (8 features vs 7, Keras MLP
`Dense→LayerNorm→GELU` vs `Linear+ReLU`, grid 231 điểm vs 66). **Trước mọi cải tiến,
phải chạy lại toàn bộ §5 với code hiện tại** để có baseline "thật" làm mốc so sánh.

- [ ] Re-run `scripts/train_mlp.py` → `scripts/evaluate_all.py` cho 4 điều kiện (dev, test, dev_noisy, dangdocao).
- [ ] Cập nhật bảng §5 + gỡ banner cảnh báo trong `research_paper.md`.
- [ ] Chốt lại con số "MLP vs fixed-equal" mới — đây là **mốc** mọi cải tiến bên dưới phải vượt.

---

## 1. Chẩn đoán điểm yếu (vì sao gain đang biên)

Đọc từ chính kết quả của paper:

| Bằng chứng | Hệ quả |
|---|---|
| Entropy trọng số `H̄ ≈ 1.098` ≈ `ln3 = 1.099`, `σ_w ≈ 0.01` | MLP gần như luôn xuất `(0.33, 0.33, 0.33)` — "dynamic" chỉ dịch ±0.03 quanh tâm simplex. |
| Recall@100 ≈ 0.99 cho mọi method 3-way | Bài toán thực chất là **re-rank trong tập gần như đã đủ**, không phải mở rộng candidate → trần cải thiện thấp. |
| Hard-label sụp đổ (NDCG 0.77 < dense-only) | Supervision hiện tại buộc phải "mềm", nhưng mềm quá → model nhạt. Có một *cliff* T* chưa định vị. |
| `compound_ratio ↔ w_bm25`: r ≈ 0 (n.s.) | Ít nhất 1/8 feature **vô dụng** trên clean text → feature space yếu. |
| RAGAS clean: MLP ≈ fixed-equal, thua dense-only 3/4 metric | Gain retrieval **không** truyền xuống end-to-end ở điều kiện sạch. |
| Noisy (mất dấu): MLP > mọi baseline, dẫn 3/4 RAGAS | **Robustness mới là chỗ đóng góp mạnh nhất**, không phải clean accuracy. |

**Kết luận chiến lược:** (a) cần fusion *biểu cảm hơn* và *sắc hơn*; (b) cần *re-position*
contribution quanh **robustness dưới nhiễu tiếng Việt** thay vì cố ăn +0.003 trên clean.

---

## 2. Các hướng cải tiến (ưu tiên P0 → P2)

### P0-A. Cho MLP "nhìn thấy" tín hiệu retrieval, không chỉ feature ngôn ngữ
**Hypothesis:** MLP hiện chỉ nhận `φ(query)` (8 đặc trưng ngôn ngữ) nên không biết *các
retriever có đồng thuận hay mâu thuẫn trên chính query này không*. Bổ sung **signal-aware
features** rẻ tính từ kết quả top-k của mỗi nguồn sẽ cho model bằng chứng trực tiếp để
phân bổ trọng số.

**Việc cần làm:**
- Thêm các đặc trưng tính từ `dense_hits / bm25_hits / sparse_hits` (đã có sẵn trong `hybrid.py`):
  - Độ chồng lấn top-10 giữa từng cặp nguồn (Jaccard).
  - Độ "nhọn" của phân phối điểm mỗi nguồn (max − mean, hoặc entropy của score đã chuẩn hoá).
  - Rank của doc top-1 chung giữa các nguồn.
- Mở rộng input MLP từ `R^8` → `R^(8+k)`; sửa `features/vietnamese.py` (hoặc tạo
  `features/signal.py`) và `fusion/mlp.py` (`input_dim`).
- **Lưu ý chi phí:** các feature này cần chạy retrieval *trước* khi tính trọng số → đổi thứ
  tự trong `pipeline.py`/`train_mlp.py` (retrieve → featurize → weight → fuse). Vẫn rẻ vì
  retrieval dù sao cũng phải chạy.

**Payoff kỳ vọng:** cao — đây là lý do cơ chế khiến model dám rời tâm simplex. **Rủi ro:** trung bình (cần cẩn thận leakage khi tính feature từ chính tập đang rank).

### P0-B. Đổi loss & target để nới rộng phân phối trọng số
**Hypothesis:** MSE-to-soft-label kéo mọi dự đoán về trung bình → entropy bão hoà. Loss
xếp hạng trực tiếp sẽ thưởng cho việc *dịch trọng số đúng hướng* thay vì *khớp một con số mềm*.

**Việc cần làm (thử song song, chọn cái thắng):**
1. **KL-divergence** giữa softmax(MLP) và phân phối mục tiêu (thay MSE) — paper §6 đã gợi ý.
2. **Listwise ranking loss** (ví dụ xấp xỉ LambdaRank/ApproxNDCG): tối ưu trực tiếp NDCG@10
   của thứ hạng *sau fusion* theo trọng số MLP, bỏ qua bước soft-label trung gian. Đây là
   thay đổi *idea* lớn nhất: học fusion **end-to-end theo metric** thay vì học bắt chước grid.
3. Định vị **temperature cliff** `T*`: quét `T ∈ {0.2, 0.15, 0.1, 0.05, 0.02}` (mở rộng ablation §5.6),
   vẽ NDCG & entropy theo T để tìm điểm sụp.

**Payoff:** cao (đánh thẳng vào limitation chính). **Rủi ro:** trung bình–cao (ranking loss khó train ổn định).

### P1-C. Sửa/Thay feature space yếu
**Hypothesis:** `compound_ratio` vô dụng (r≈0); một số feature khác có thể trùng thông tin.
**Việc cần làm:**
- Phân tích đóng góp 8 feature mới (gồm `oov_ratio` vừa thêm) bằng ablation bỏ-từng-feature +
  permutation importance trên dev.
- Thay `compound_ratio` bằng đặc trưng giàu thông tin hơn: tỉ lệ entity in-hoa, tỉ lệ số,
  hoặc embedding câu hỏi nén xuống ~8 chiều bằng PCA (giữ MLP vẫn "nhẹ").
- Báo cáo bảng "feature → Δ NDCG" để tăng tính giải thích (reviewer thích).

**Payoff:** trung bình. **Rủi ro:** thấp.

### P1-D. Baseline mạnh hơn + trần oracle
**Hypothesis:** Để chứng minh "dynamic" đáng giá, cần so với *best fixed-weight tune trên dev*
(không chỉ fixed-equal) và cho thấy **khoảng cách tới oracle per-query** còn rộng.
**Việc cần làm (đa số chỉ thêm method vào `evaluate_all.py::METHODS`):**
- `best_fixed`: quét grid trên dev, chọn `(a,b,c)` tốt nhất, áp lên test. (CLAUDE.md §Metrics đã liệt kê baseline #5 này nhưng code chưa có.)
- `rrf`: Reciprocal Rank Fusion — baseline hybrid kinh điển, hiện đang thiếu.
- `oracle`: với mỗi query chọn điểm grid tốt nhất (per-query argmax NDCG) → trần trên của fusion.
  Khoảng cách `oracle − mlp` chính là "headroom" để định khung mức độ còn cải thiện được.

**Payoff:** cao cho *câu chuyện* của paper (định khung đóng góp). **Rủi ro:** thấp.

### P2-E. Thêm tín hiệu thứ tư: BGE-M3 ColBERT multi-vector
Paper §6 đã đề xuất. BGE-M3 đã cho sẵn multi-vector → mở rộng fusion sang 4-way `(a,b,c,d)`,
simplex 3 chiều. Chi phí: index + bộ nhớ ColBERT lớn. Chỉ làm khi P0–P1 đã cho gain rõ.

### P2-F. Re-position contribution quanh ROBUSTNESS
Dựa trên §5.1/§5.7: gain lớn và ổn định nằm ở **query mất dấu**. Đề xuất:
- Đưa robustness lên thành **đóng góp chính** trong abstract/§1, clean accuracy là phụ.
- Bổ sung **noise thật**: thu thập/giả lập query người dùng gõ thiếu dấu *một phần* (không chỉ
  100% strip), nhiều mức nhiễu (0/25/50/75/100%) → vẽ đường cong robustness MLP vs baseline.
- Đây là điểm bán hàng khác biệt so với hybrid RAG tiếng Anh.

---

## 3. Khoảng trống đo lường cần lấp (đã phát hiện trong code)

- [ ] `evaluate_all.py` **chưa** đo latency end-to-end p50/p95 & throughput dù CLAUDE/README từng
      claim. Bổ sung đo per-query wall-clock (retrieve + fuse) và q/s. (Đã sửa doc cho khớp;
      nếu muốn giữ claim thì phải code thêm.)
- [ ] Embedding query trong `evaluate_all.py` đang gọi API **tuần tự từng query** → vừa chậm vừa
      làm nhiễu số efficiency. Pre-embed theo batch + cache (như `train_mlp.py` đã làm) trước vòng lặp.
- [ ] Qwen3-32B là reasoning model → cân nhắc strip `<think>...</think>` trong `generator/llm.py`
      trước khi chấm RAGAS (tránh nhiễu Faithfulness/Answer-Relevancy).
- [ ] Thêm `tests/test_metrics.py` (known-answer cho ndcg/mrr/map/recall) — bảo vệ trực tiếp
      các con số trong paper. Giờ đã có `src/rag_vie/utils/metrics.py` làm single source of truth.

---

## 4. Thứ tự thực thi đề xuất

1. **Tuần 1** — P0 prerequisite: re-run §5 với code hiện tại, chốt baseline mới + lấp đo lường (§3).
2. **Tuần 2** — P1-D (baseline mạnh + oracle headroom) + P1-C (feature ablation). Rẻ, cho ngay
   "câu chuyện" rõ ràng về chỗ còn cải thiện.
3. **Tuần 3–4** — P0-A (signal-aware features) + P0-B (KL / ranking loss + tìm T*). Đây là hai
   đòn bẩy cơ chế mạnh nhất; đo lại toàn bộ.
4. **Tuần 5** — P2-F robustness curve (đóng góp bán hàng) + viết lại §1/§6 theo positioning mới.
5. **Tuần 6+** — P2-E (ColBERT 4-way) nếu còn thời gian và P0 đã thắng.

---

## 5. Tiêu chí thành công

- **Định lượng:** MLP vượt `best_fixed` (không chỉ fixed-equal) với p < 0.01 trên test;
  entropy trọng số giảm rõ (model dám rời tâm simplex) mà NDCG *không* sụp như hard-label.
- **Cơ chế:** ít nhất một tương quan feature↔weight mới, mạnh và có thể giải thích (ngoài
  `english↔w_sparse` hiện có).
- **Robustness:** đường cong NDCG theo mức nhiễu cho thấy khoảng cách MLP–baseline *nới rộng*
  khi nhiễu tăng.
- **Headroom:** báo cáo `oracle − mlp` để định khung trung thực mức đóng góp.
