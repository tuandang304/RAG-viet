# Phần mở rộng: Dynamic Hybrid RAG cho tiếng Việt

**Phiên bản:** 1.1
**Ngày:** 08/05/2026

## Tóm tắt (Executive Summary)

Đề xuất này mở rộng và cụ thể hóa bài toán **Dynamic Hybrid RAG** (kết hợp các phương pháp tìm kiếm động) dành riêng cho ngữ cảnh **tiếng Việt**. Khác với tiếng Anh, tiếng Việt đối mặt với các thách thức đặc thù như: việc tách từ (word segmentation) ảnh hưởng mạnh đến thuật toán BM25, lỗi gõ thiếu dấu (diacritic noise), và sự pha trộn ngôn ngữ (code-switching). 

Để giải quyết vấn đề này, đề xuất tập trung vào **kiến trúc Three-way fusion** tận dụng mô hình BGE-M3. Thay vì cố định trọng số như các phương pháp truyền thống, một mạng nơ-ron nhỏ (MLP) sẽ dựa vào các **Vietnamese-aware features** (đặc trưng nhận biết tiếng Việt) trích xuất từ câu hỏi để học cách phân bổ linh hoạt trọng số `(a, b, c)` cho ba tín hiệu tìm kiếm: Dense vector, Sparse lexical và Multi-vector (ColBERT). 

Đóng góp cốt lõi của hướng nghiên cứu bao gồm:
1. **Kiến trúc dung hợp động (Dynamic Fusion)** tối ưu hóa cho tiếng Việt.
2. **Bộ đặc trưng ngôn ngữ** phân tích tỉ lệ từ ghép, có/không dấu, từ khóa chuyên ngành v.v.
3. **Thiết lập thực nghiệm Cross-domain toàn diện** trên ba bộ dữ liệu (UIT-ViQuAD 2.0, VIMQA, ViNewsQA) nhằm kiểm chứng tính tổng quát và khả năng chống chịu nhiễu (diacritic robustness).


---

## 1. Định vị lại đóng góp (replaces §1)

Đóng góp của paper được phát biểu lại theo bốn ý:

1. **Một MLP fusion module nhẹ** học `(a, b)` động theo từng query, thay cho weighted sum cố định trong hybrid retrieval.
2. **Một bộ feature extractor nhận biết đặc trưng tiếng Việt** (diacritic, từ ghép, code-switching Việt-Anh, mức độ chuyên ngành), bổ sung cho các retrieval-signal feature chuẩn.
3. **Một thiết lập đánh giá cross-domain trên ba dataset tiếng Việt** (general Wikipedia / multi-hop / tin tức), kiểm tra cả khả năng generalize và độ vững khi domain shift.
4. **Phân tích khi dynamic fusion thực sự thắng** baseline best fixed-weight — tức trả lời được câu hỏi: query loại nào, domain nào thì cách này có lợi rõ rệt, loại nào thì không khác baseline. 


---

## 2. Vì sao bối cảnh tiếng Việt làm bài toán thay đổi (extends §2)

Có ba điểm khiến hybrid retrieval cho tiếng Việt **không phải chỉ là "MS MARCO dịch sang tiếng Việt"**:

### 2.1. Word segmentation là tiền đề bắt buộc cho BM25

Tiếng Việt viết tách theo âm tiết, nhưng đơn vị nghĩa thường là từ ghép nhiều âm tiết ("học sinh", "trí tuệ nhân tạo", "bệnh tiểu đường"). BM25 nếu tokenize theo whitespace sẽ tách hỏng từ ghép → giảm chất lượng đáng kể. Bắt buộc phải dùng word segmenter (VnCoreNLP, underthesea, hoặc pyvi) trước khi index BM25. Lựa chọn segmenter nào và đo ảnh hưởng của nó là một sub-experiment đáng có.

### 2.2. Diacritic là biến số thứ hai

Người dùng thật thường gõ thiếu dấu hoặc gõ Telex sai ("benh tieu duong" thay vì "bệnh tiểu đường"). Dense embedding của BGE-M3 robust hơn với việc thiếu dấu, BM25 thì gần như fail hoàn toàn nếu corpus có dấu mà query không có. Đây là lý do tự nhiên để dynamic fusion hữu ích: **MLP có thể học cách giảm trọng số BM25 khi query thiếu dấu.**

### 2.3. BGE-M3 đã có hybrid built-in

Vietnamese_Embedding fine-tuned từ BGE-M3, mà BGE-M3 nguyên bản đã hỗ trợ ba representation: dense, sparse (lexical weight), và multi-vector (ColBERT-style). Tức là baseline mạnh nhất không phải "dense + BM25 với fixed weight" nữa, mà là **BGE-M3 native hybrid**. Đây là baseline phải vượt được — nếu không vượt được thì ý tưởng yếu.

Thực tế, có thể tận dụng cả ba head của BGE-M3 thay vì chỉ dense + BM25 ngoài, và đẩy thành **ba-way dynamic fusion**: `(a, b, c)` với softmax.

---

## 3. Kiến trúc đề xuất (replaces §3-§4)

### 3.1. Hai lựa chọn kiến trúc

**Lựa chọn A — Two-way fusion (đơn giản, an toàn):**
```
score(q, d) = a · norm(s_dense)  +  b · norm(s_bm25)
```
trong đó `s_dense` từ Vietnamese_Embedding, `s_bm25` từ Pyserini/Elasticsearch với Vietnamese tokenizer.

**Lựa chọn B — Three-way fusion:**
```
score(q, d) = a · norm(s_dense)  +  b · norm(s_lex)  +  c · norm(s_colbert)
(a, b, c) = softmax(MLP(features(q)))
```
trong đó cả ba điểm đều lấy từ BGE-M3, tận dụng tối đa unified encoder và tránh mismatch token giữa BM25 ngoài và embedding model.

Lựa chọn B mạnh hơn vì: (1) một forward của encoder cho ra cả ba điểm, không tốn thêm latency; (2) ba head cùng học trên cùng tokenizer nên scale điểm nhất quán hơn; (3) kết quả có "câu chuyện" rõ ràng hơn cho paper — không phải so dense vs BM25 lại lần thứ N, mà là **học cách phối hợp ba representation vốn đã có sẵn trong cùng một model**.

Lựa chọn A vẫn nên giữ làm phương án dự phòng nếu việc trích xuất sparse/multi-vector từ BGE-M3 phức tạp về mặt kỹ thuật.

### 3.2. Sơ đồ luồng inference (cho Lựa chọn B)

```
                          Query (tiếng Việt)
                                │
                                ▼
                        Word segmentation
                                │
                                ▼
                  BGE-M3 / Vietnamese_Embedding
                  (single forward pass)
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
       Dense vector       Lexical weights      Multi-vector
            │                   │                   │
            ▼                   ▼                   ▼
       FAISS top-k         Inverted index      ColBERT scoring
            │                   │                   │
            └─────────┬─────────┴─────────┬─────────┘
                      ▼                   ▼
                Union of candidates + per-source normalize
                                │
                                ▼
              MLP(features(q)) → softmax → (a, b, c)
                                │
                                ▼
              Fused score = a·s_dense + b·s_lex + c·s_colbert
                                │
                                ▼
                      Top-N → LLM generator
```

---

## 4. Các mô hình (Models) sử dụng trong proposal

Dựa trên kiến trúc và yêu cầu xử lý tiếng Việt, dưới đây là danh sách cụ thể các loại mô hình sẽ được tích hợp:

### 4.1. Embedding & Retrieval Models (Mô hình trích xuất đặc trưng và tìm kiếm)
* **BGE-M3**: Đóng vai trò là nền tảng chính (backbone) cho cả Lựa chọn B (Three-way fusion). BGE-M3 mạnh ở khả năng hỗ trợ đa ngôn ngữ và cung cấp cùng lúc ba loại biểu diễn (dense, sparse lexical, multi-vector ColBERT) chỉ qua một lần forward pass.
* **Vietnamese_Embedding (hoặc các biến thể fine-tune từ BGE-M3 cho tiếng Việt)**: Được sử dụng để tối ưu hóa vector dense đặc thù cho ngôn ngữ và domain tiếng Việt, tăng độ chính xác so với BGE-M3 gốc.
* **BM25 (thông qua Pyserini/Elasticsearch)**: Dùng làm baseline và cho Lựa chọn A (Two-way fusion) kết hợp với dense vector.

### 4.2. Tokenizer & Word Segmenter (Mô hình tách từ)
Do tính chất của tiếng Việt, các segmenter là bắt buộc trước khi đưa qua BM25 hoặc trích xuất feature:
* **VnCoreNLP**: Khuyến nghị dùng làm tokenizer chính cho toàn bộ thực nghiệm nhờ độ ổn định cao, được cộng đồng nghiên cứu tiếng Việt sử dụng rộng rãi, giúp kết quả có tính so sánh đối chiếu chuẩn xác.
* **Underthesea / Pyvi**: Các giải pháp thay thế, có thể dùng để làm thí nghiệm (sub-experiment) đánh giá ảnh hưởng của segmentation lên hiệu năng retrieval.

### 4.3. Generator Models (Mô hình LLM sinh văn bản)
Nhiệm vụ của các mô hình này là nhận context đã được retrieve (Top-N) và câu hỏi (Query) để sinh ra câu trả lời cuối cùng:
* **Mô hình thương mại (API-based)**: GPT-4o-mini hoặc Gemini 1.5 Flash. Chi phí rẻ, inference nhanh, chất lượng text sinh ra tiếng Việt rất tốt, dùng làm Upper Bound hoặc tiêu chuẩn để đánh giá RAG quality.
* **Mô hình mã nguồn mở (Open-source)**: Llama-3-8B-Instruct, Qwen-2.5 (đặc biệt bản 7B hỗ trợ tiếng Việt rất tốt), hoặc các mô hình nội địa như VinaLlama / PhoGPT. Việc đánh giá trên một open-source LLM giúp paper mang tính tái tạo (reproducibility) cao hơn.

### 4.4. Fusion Model (Mô hình kết hợp)
* **MLP (Multilayer Perceptron)**: Một mạng nơ-ron nhỏ (khoảng 2-3 lớp ẩn) để học trọng số `(a, b, c)` một cách động (dynamic) dựa vào các *Vietnamese-aware features* trích xuất từ câu hỏi. Mô hình này rất nhẹ, train nhanh và không gây tắc nghẽn (bottleneck) ở quá trình inference.

---

## 5. Vietnamese-aware features cho MLP (extends §5.1)

Giữ nguyên các nhóm feature trong proposal gốc, **bổ sung** một nhóm mới gọi là *Vietnamese-aware features*:

| Feature | Tín hiệu mà nó cung cấp |
|---|-
| Tỉ lệ âm tiết có dấu / tổng âm tiết | Query gõ đủ dấu hay thiếu dấu |
| Tỉ lệ token là từ ghép (≥2 âm tiết) sau segmentation | Query thiên về thuật ngữ chuyên ngành |
| Có chứa từ tiếng Anh không (code-switching) | Query pha từ kỹ thuật/công nghệ, thường gặp trong ViNewsQA |
| Tỉ lệ token thuộc từ điển chuyên ngành (đơn giản, dùng lexicon) | Độ chuyên biệt của query so với general Wikipedia |
| Số mệnh đề (đếm dấu phẩy, từ nối "và", "hoặc", "nếu") | Đại lượng proxy cho multi-hop, liên quan VIMQA |
| Có từ để hỏi không ("ai", "gì", "khi nào", "tại sao", "như thế nào") | Question type, ảnh hưởng dense vs sparse |

Thêm nhóm này có hai tác dụng. Một là tăng khả năng MLP học được pattern hữu ích. Hai là, quan trọng hơn cho paper, tạo ra một **ablation table** sạch sẽ để chứng minh mỗi nhóm feature đóng góp bao nhiêu.

---

## 6. Setup thực nghiệm (replaces §7)

### 6.1. Ba dataset, ba vai trò khác nhau

Việc ba dataset khác nhau rõ ràng về tính chất là **tài sản** chứ không phải gánh nặng — chúng tự cấu thành một bộ test cross-domain mà không cần phải tự chế. Tập trung vào General (Wikipedia) làm core, ViNewsQA làm bước domain shift tự nhiên.

| Dataset | Domain | Đặc điểm câu hỏi | Vai trò trong evaluation |
|---|---|---|---|
| UIT-ViQuAD 2.0 | General (Wikipedia) | Single-hop, factual, SQuAD-style; ~23K QA pairs, chất lượng cao, human-annotated | Train chính + in-domain test |
| VIMQA | General (Wikipedia) | Multi-hop reasoning, cần tổng hợp nhiều đoạn văn; ~20K QA pairs | Test khả năng xử lý query phức tạp |
| ViNewsQA | Tin tức (VnExpress) | Single-hop nhưng ngữ cảnh báo chí, khác phong cách Wikipedia; ~22K QA pairs | Test cross-domain shift (Wikipedia → News) |

> **Lý do bỏ ViHealthQA và XLMRQA:**  
> - *ViHealthQA*: Domain y tế quá xa General Wikipedia, làm lệch câu chuyện nghiên cứu; kết quả sẽ khó giải thích vì terminology gap quá lớn.  
> - *XLMRQA*: Là bản dịch máy, chất lượng không đồng đều, có thể tạo artifact giả tạo trong kết quả. Với mục tiêu Q3 journal, một bộ dữ liệu dịch máy không annotated thủ công sẽ bị reviewer chất vấn.

### 6.2. Hai protocol đánh giá

**Protocol 1 — In-domain:** Train và test trên cùng một dataset. Cho mỗi dataset một bảng kết quả riêng.

**Protocol 2 — Cross-domain (đóng góp chính):** Train MLP **chỉ trên UIT-ViQuAD 2.0**, sau đó zero-shot evaluate trên VIMQA và ViNewsQA. Đây là phép kiểm tra tương đương với BEIR trong proposal gốc, và là phần mà reviewer sẽ tìm.

Lý do split như vậy: UIT-ViQuAD 2.0 lớn nhất, cấu trúc gần SQuAD nhất, và là dataset general-domain nên train trên đó rồi test các domain khác là setup logic. VIMQA kiểm tra khả năng xử lý query phức tạp hơn, ViNewsQA kiểm tra domain shift nhẹ (Wikipedia → news). Nếu MLP chỉ tốt khi train + test cùng dataset, nó có thể chỉ là overfit trên distribution đó.

### 6.3. Baselines

Phải so với đầy đủ các baseline sau, theo thứ tự tăng dần độ khó:

1. BM25 (với Vietnamese tokenizer) only
2. Vietnamese_Embedding dense only
3. Fixed-weight hybrid `a = b = 0.5`
4. **Best fixed-weight** tune trên dev — baseline khó nhất theo proposal gốc
5. **BGE-M3 native hybrid** (multi-head với weight cố định mặc định) — baseline khó nhất theo phiên bản tiếng Việt
6. RRF với k cố định

Vượt được #4 và #5 đồng thời là yêu cầu tối thiểu để paper có giá trị.

### 6.4. Metrics

Giữ nguyên: NDCG@10, MRR@10, Recall@100, latency (p50/p95). Bổ sung **per-query-type breakdown**: chia query theo (a) độ dài, (b) có/không có dấu, (c) có/không multi-hop. Bảng breakdown này thường là phần được trích dẫn trong related work của paper sau.

### 6.5. Phân tích bắt buộc

Hai phân tích dưới đây không thể thiếu, vì chúng là phần biến từ "thử nghiệm có cải thiện số" thành "paper có đóng góp khoa học":

- **Phân phối `(a, b, c)` theo loại query:** scatter plot hoặc heatmap chứng minh MLP thật sự thay đổi weight theo input, không collapse về một giá trị cố định.
- **Khi nào dynamic fusion thắng và khi nào không:** chia tập test thành các bin theo feature, đo gain so với best fixed-weight. Kết luận trung thực — kể cả khi gain bằng 0 trên một số bin — sẽ tăng độ tin cậy của paper.

---

## 7. Phần mở rộng nên giữ và nên bỏ (replaces §6)

Proposal gốc liệt kê bốn extension (adaptive routing, query-aware top-k, learnable RRF, MoE):

**Đưa vào paper chính:**
- Vietnamese-aware features (đã trình bày ở §5 ở trên).
- Three-way fusion với BGE-M3 (đã trình bày ở §3).
- Cross-domain evaluation trên ba dataset (đã trình bày ở §6).

**Một thí nghiệm phụ — chọn một trong hai:**
- *Diacritic robustness study:* lấy test set, drop dấu ngẫu nhiên 0% / 50% / 100%, đo NDCG. Mục tiêu là chứng minh dynamic fusion robust hơn fixed weight khi diacritic noise tăng. Thí nghiệm này rất rẻ, dễ bán cho reviewer, và là contribution Vietnamese-specific rất khó tranh cãi.
- *Adaptive routing đơn giản:* nếu MLP confidence cao về một head (ví dụ `a > 0.8`), bỏ qua hai head còn lại lúc inference. Đo latency saving với negligible NDCG drop.

Khuyên chọn **diacritic robustness** vì gắn với câu chuyện tiếng Việt rõ hơn và rủi ro thực hiện thấp hơn.

**Để vào "Future work" của paper, không làm trong scope này:**
- Distillation từ cross-encoder (cần training compute lớn, không có Vietnamese cross-encoder mạnh sẵn).
- Mixture of Experts.
- Learnable RRF như target chính.

---

## 8. Roadmap điều chỉnh cho sinh viên (replaces §8)

Giả định: một sinh viên làm 15-20 giờ/tuần, có một GPU 24GB (hoặc Google Colab Pro+), không có teammate. Mục tiêu nộp **Q3 journal** (ví dụ: Applied Intelligence, Journal of Intelligent Information Systems, hoặc Expert Systems with Applications). 10 tuần là realistic; 8 tuần khả thi nếu không gặp sự cố kỹ thuật lớn.

| Tuần | Việc | Deliverable cuối tuần |
|---|---|---|
| 1 | Setup môi trường, tải **ba dataset** (UIT-ViQuAD 2.0, VIMQA, ViNewsQA), chuẩn hóa format chung | Ba dataset cùng schema: `{id, question, context, answer}` |
| 2 | Build BM25 index với VnCoreNLP tokenizer, build dense index với Vietnamese_Embedding | Hai retriever chạy được, NDCG@10 riêng từng nguồn trên ViQuAD dev |
| 3 | Implement BGE-M3 multi-head extraction + fixed-weight hybrid baseline | Baseline #1–#5 có số trên UIT-ViQuAD 2.0 dev |
| 4 | Implement feature extractor (chuẩn + Vietnamese-aware) và MLP | MLP chạy được, training loop chạy 1 epoch không lỗi |
| 5 | Train chính trên UIT-ViQuAD 2.0 với pairwise loss, hard negative mining, warm-start | Vượt best fixed-weight trên UIT-ViQuAD 2.0 dev |
| 6 | Cross-domain evaluation trên VIMQA và ViNewsQA (zero-shot) | Bảng kết quả chính (Table 2 trong paper) |
| 7 | Ablation: bỏ từng nhóm feature, đo lại NDCG | Bảng ablation (Table 3) |
| 8 | Diacritic robustness study | Biểu đồ NDCG vs diacritic dropout rate (Figure 3) |
| 9 | Phân tích `(a, b, c)` distribution + per-query-type breakdown | Các figure cho paper (Figure 4–5) |
| 10 | Viết paper (8–10 trang), chuẩn bị code release lên GitHub | Bản nháp paper sẵn sàng nộp |

Tuần 5 là milestone quan trọng nhất. Nếu hết tuần 5 mà chưa vượt được best fixed-weight, phải dừng để debug feature/loss/normalization trước khi đi tiếp — đừng cố làm cross-domain với một model còn yếu.

> **Ghi chú scope cho Q3:** Với ba dataset thay vì bốn, phần thực nghiệm gọn hơn nhưng vẫn đủ thuyết phục cho Q3. Reviewer Q3 thường không yêu cầu BEIR-scale evaluation, nhưng sẽ chú ý vào (1) baseline so sánh đầy đủ, (2) ablation rõ ràng, (3) phân tích khi nào method thắng/thua — ba điểm này được bảo đảm bởi roadmap trên.

---

## 9. Rủi ro mới phát sinh do bối cảnh tiếng Việt (extends §9)

Bốn rủi ro của proposal gốc vẫn áp dụng. Bổ sung thêm:

**Rủi ro 6: Vietnamese tokenizer khác nhau cho ra kết quả BM25 khác nhau đáng kể.**
→ Cố định một tokenizer (đề xuất: VnCoreNLP vì có paper, ổn định, được trích dẫn nhiều) cho toàn bộ thí nghiệm. Báo cáo lựa chọn này rõ ràng trong paper. Không pha trộn nhiều tokenizer trong cùng một bảng kết quả.

**Rủi ro 7: BGE-M3 multi-vector chậm trên corpus lớn.**
→ Giới hạn re-ranking multi-vector ở top-100 sau dense retrieval, không scoring trên toàn corpus. Đây cũng là cách dùng tiêu chuẩn của BGE-M3, không phải workaround đặc biệt.

**Rủi ro 8: ViNewsQA có phong cách câu hỏi khác biệt đáng kể so với UIT-ViQuAD 2.0.**
→ Báo chí thường dùng câu hỏi ngắn, ngôn ngữ thông tục, bối cảnh thời sự. Nếu MLP train trên ViQuAD không generalize sang ViNewsQA, đây là **kết quả đáng báo cáo** chứ không phải thất bại — nó chứng minh rằng dynamic fusion cần training data phù hợp domain. Đưa phân tích này vào phần Limitations/Future Work.

**Rủi ro 9: BGE-M3 native hybrid đã quá mạnh, dynamic fusion không vượt được nhiều.**
→ Đây là rủi ro thật. Mitigation: (a) đảm bảo Vietnamese-aware features đủ phong phú để có nguồn tín hiệu mà BGE-M3 native không có; (b) nếu gain tổng nhỏ, vẫn có thể bán câu chuyện qua phân tích — ví dụ "gain nhỏ trên trung bình, lớn trên query thiếu dấu/multi-hop".

