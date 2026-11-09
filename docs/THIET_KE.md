# Thiết kế kỹ thuật: Gợi ý việc làm IT cho sinh viên từ CV

Tài liệu này giải thích ý tưởng, luồng chạy của code, và lý do đằng sau từng quyết định kỹ thuật. Đọc cùng [README](../README.md); README chỉ ghi cách chạy.

---

## 1. Ý tưởng chính

Bài toán là **xếp hạng**: với một CV, sắp xếp khoảng 160 tin tuyển dụng theo độ phù hợp, rồi giải thích vì sao (kỹ năng đã khớp và còn thiếu).

Ba ý tưởng làm nên hệ thống:

1. **Ba tín hiệu bổ sung cho nhau.** Mỗi tín hiệu giỏi một kiểu và yếu một kiểu:

   | Tín hiệu | Bắt được | Bỏ sót |
   |---|---|---|
   | TF-IDF | Từ khoá trùng chính xác ("Airflow", "PySpark") | Đồng nghĩa, khác ngôn ngữ ("học máy" và "machine learning") |
   | Embedding đa ngôn ngữ | Nghĩa tổng thể, tiếng Việt và tiếng Anh lẫn lộn | Chi tiết kỹ năng cụ thể; thường cho điểm cao mọi tin "AI" |
   | Độ phủ kỹ năng | Có hay không có đúng kỹ năng tin yêu cầu, qua từ điển alias | Bất kỳ thứ gì ngoài từ điển |

   Điểm cuối cùng là `w1·TF-IDF + w2·embedding + w3·kỹ năng`. Bộ trọng số được chọn bằng dữ liệu, không chọn bằng cảm tính.

2. **Lọc cấp bậc trước khi xếp hạng.** Trong lần crawl đầu, có 70/164 tin ghi Senior/Lead ngay ở tên tin. Nếu không lọc, sinh viên sẽ thấy toàn tin không nộp được. Bộ lọc dùng luật đơn giản và giải thích được (xem mục 4.3).

3. **Đánh giá không cần gán nhãn tay.** Dùng LLM làm giám khảo. Để kết quả có thể tin được, có bốn ràng buộc:
   - giám khảo khác nhà cung cấp với LLM trong pipeline;
   - có 2 giám khảo, và đo độ đồng thuận giữa họ;
   - đo độ ổn định khi chấm lại;
   - tính lại toàn bộ kết quả bằng nhãn của giám khảo thứ hai để xem kết luận có đổi không.

---

## 2. Luồng tổng thể

```
                      ┌──────────────────── OFFLINE (chạy 1 lần / mỗi đợt crawl) ────────────────────┐
 ITviec ──crawl──▶ itviec_jobs.json ──build_index──▶ jobs_processed.json + job_embeddings.npy
                                         │             (cấp bậc, số năm KN, lọc, kỹ năng chuẩn hoá)
                                         └─ LLM trích kỹ năng từ "Requirements" ─▶ jd_skills_llm.json

 CV thật (PDF) ──prepare_cvs──▶ data/cvs/*.txt (ẩn danh)        ┐
 LLM sinh CV giả có đáp án ──make_synthetic_cvs──▶ synthetic_cvs.json ┘──evaluate──▶ results/
                                                                       (report.md, metrics.json,
                                                                        best_weights.json, ...)
                      └──────────────────────────────────────────────────────────────────────────────┘

                      ┌──────────────────── ONLINE (mỗi lần người dùng tải CV) ──────────────────────┐
 PDF ─▶ pdfplumber ─▶ ẩn danh (regex) ─▶ trích kỹ năng (từ điển, hoặc LLM nếu bật)
     ─▶ 3 tín hiệu × 164 tin ─▶ min-max theo CV ─▶ cộng theo best_weights ─▶ lọc cấp bậc ─▶ top-10
     ─▶ kỹ năng khớp/thiếu cho từng tin ─▶ đếm kỹ năng thiếu ─▶ (LLM) lộ trình học
                      └──────────────────────────────────────────────────────────────────────────────┘
```

Thứ tự lệnh (chi tiết trong README):

```
python itviec_crawler.py            (chạy trong thư mục data/)
python scripts/build_index.py
python scripts/make_synthetic_cvs.py --n 40      # và/hoặc prepare_cvs.py nếu có CV thật
python scripts/evaluate.py
streamlit run app.py
```

---

## 3. Cấu trúc code

```
itviec_crawler.py      Thu thập dữ liệu (code của bạn, đã sửa 1 lỗi; xem 4.1)
jobmatch/              Thư viện lõi. Không phụ thuộc Streamlit, dùng chung cho script và demo
  config.py            Mọi đường dẫn, tên model, vai trò LLM, ngưỡng, trọng số mặc định
  skills.py            Từ điển kỹ năng + trích kỹ năng bằng regex
  jobs.py              Suy ra cấp bậc, số năm kinh nghiệm, lọc tin, gộp kỹ năng của tin
  cv.py                Đọc PDF, ẩn danh bằng regex
  matcher.py           Ba tín hiệu, kết hợp, xếp hạng, kỹ năng khớp/thiếu
  llm.py               Gọi OpenAI và Gemini, cache, retry, chạy song song
  llm_tasks.py         Toàn bộ prompt
scripts/               Các bước offline (mỗi file chạy độc lập bằng python scripts/xxx.py)
app.py                 Demo Streamlit
```

Nguyên tắc: **mọi logic nằm trong `jobmatch/`**. Script và app chỉ ghép các bước lại với nhau. Nhờ vậy demo và đánh giá dùng *cùng một* code xếp hạng, nên con số trong báo cáo đúng là chất lượng của bản demo.

---

## 4. Giải thích từng phần

### 4.1 Crawler: `itviec_crawler.py`

Giữ nguyên thiết kế của bạn: duyệt `?page=1,2,…` tới khi không còn link mới, mở từng tin, cắt các mục theo tiêu đề cố định (`SECTIONS`). Có hai thay đổi:

- **Sửa lỗi `posted` luôn rỗng.** Trên trang ITviec, "Posted" và "1 day ago" là **hai dòng riêng**, trong khi code cũ tìm cả cụm "Posted … ago" trên một dòng. Bản sửa ghép dòng "Posted" với dòng ngay sau nó, và vẫn nhận dạng cũ nếu ITviec đổi lại.
- **`DELAY = 0.5`** là thay đổi của bạn, giúp crawl nhanh khoảng 4 lần (164 tin cộng khoảng 10 trang danh sách thì mất chừng 1,5–2 phút). Cần biết: `get_soup` gọi `raise_for_status()`, và trong `main` mỗi tin được bọc `try/except`, nên nếu ITviec trả 429/403 thì tin đó chỉ in "lỗi …" rồi **bị bỏ qua, không thử lại**. Sau khi crawl, hãy kiểm tra số dòng "lỗi" trong log. Nếu có, tăng `DELAY` lên 1–2 rồi crawl lại.

### 4.2 Từ điển kỹ năng: `jobmatch/skills.py`

**Mục tiêu:** đưa "PySpark", "Spark", "Apache Spark", "spark streaming" về cùng một tên gốc **Apache Spark**, ở cả phía CV lẫn phía tin.

- `ALIASES` có khoảng 100 tên gốc, mỗi tên kèm các cách viết khác: tiếng Anh, tiếng Việt ("học máy", "kho dữ liệu"), viết tắt, và tên công cụ thuộc về nó (mlflow → MLOps, faiss → Vector Database).
- `canonical(name)` chuẩn hoá một tên. Tên không có trong bảng (tag ITviec lạ, ví dụ "Agentic AI") được giữ nguyên làm tên gốc của chính nó, nên **từ điển tự mở rộng theo dữ liệu crawl**.
- `SkillExtractor` biên dịch mỗi alias thành một regex có ranh giới từ `(?<![\w+#])…(?![\w+#])`. Ranh giới này khiến "Java" không khớp trong "JavaScript", "SQL" không khớp trong "MySQL", còn "C++" và "C#" vẫn khớp đúng. Khoảng trắng và gạch nối được nới lỏng, nên "machine-learning" vẫn khớp "machine learning".

Các luật xử lý ca khó, mỗi luật sinh ra từ một lỗi thật đã gặp khi kiểm tra:

| Luật | Ví dụ |
|---|---|
| `CASE_SENSITIVE`: viết tắt chỉ khớp đúng chữ hoa | "AI" là kỹ năng, nhưng "ai" trong "Ai cũng biết" thì không |
| `NO_FREE_TEXT`: alias quá mơ hồ, không dùng khi quét văn bản | "api", "cloud", "go", "r", "shell" |
| `NOT_FOLLOWED_BY`: cụm tiếng Việt nằm trong từ khác nghĩa | "Khoa **học máy** tính" không được tính là Machine Learning |
| `NON_SKILL_TAGS`: tag ITviec là tên vị trí, không phải kỹ năng | "Data Engineer", "Tester", "Product Owner" |

Còn một lỗi đã sửa: lúc đầu mọi alias nhiều từ ("power bi", "star schema", "học sâu") đều không khớp, vì hai lệnh `str.replace` liên tiếp ghi đè lên nhau. `_pattern` hiện tách alias thành token rồi mới escape và ghép lại.

`mentioned(skill, text)` kiểm tra một kỹ năng (hoặc alias của nó) có thật sự xuất hiện trong văn bản không. Hàm này dùng để đo **grounding rate** của LLM, tức LLM có bịa kỹ năng không có trong JD hay không.

### 4.3 Xử lý kho tin: `jobmatch/jobs.py`

`process()` thêm vào mỗi tin các trường sau:

| Trường | Cách tính |
|---|---|
| `level` | Regex trên tên tin, lấy **cấp thấp nhất** được nhắc tới: "Junior/Middle" → junior, "Middle/Senior" → middle. Không có từ khoá → `None` |
| `required_years` | Tìm mọi cụm "N years / N năm / N+ / N-M năm" trong phần Requirements, **đã nối các dòng lại**, rồi chỉ giữ cụm có "experience/kinh nghiệm" trong phạm vi 80 ký tự. Lấy cận dưới lớn nhất. Phải nối dòng vì JD ITviec hay ngắt giữa câu: `"Minimum\n7 years\nof experience"` |
| `eligible` | Giữ tin nếu thoả một trong hai: (1) có nhãn Fresher/Internship Accepted; (2) không phải senior **và** (`required_years` không có hoặc ≤ 2) |
| `skills_tag` | Tag ITviec đã chuẩn hoá |
| `skills_dict` | Kỹ năng do từ điển trích từ Requirements và Description |
| `skills_llm` | Kỹ năng do LLM trích từ Requirements (nếu có API key) |
| `job_skills` | `skills_tag ∪ skills_llm`; nếu không có LLM thì `skills_tag ∪ skills_dict`. Đây là tập dùng để so khớp |
| `text` | Tên tin, tag, Requirements và Description: văn bản đưa vào TF-IDF và embedding. Bỏ phần benefits và "reasons to join", vì đó là quảng cáo công ty, không nói gì về độ phù hợp |

Lần crawl đầu cho kết quả: 70 tin senior, 18 middle, 8 intern, 3 fresher, 2 junior, 63 tin không ghi cấp. Sau lọc còn **59 tin**.

### 4.4 Matching: `jobmatch/matcher.py`

**TF-IDF.** Dùng unigram và bigram (để giữ cụm "data warehouse"), `sublinear_tf` (một từ lặp 10 lần không đáng gấp 10 lần), `max_df=0.85` (bỏ những từ có trong gần như mọi tin). Trước khi tách từ, `c++`, `c#`, `.net` được đổi thành token chữ thường, vì tokenizer mặc định làm mất dấu `+` và `#`. Vector đã chuẩn hoá L2, nên tích vô hướng chính là cosine.

**Embedding.** Dùng `intfloat/multilingual-e5-small` (118M tham số, khoảng 470 MB, chạy trên CPU). Lý do chọn model này:
- hỗ trợ tiếng Việt;
- nhận tối đa 512 token, trong khi `paraphrase-multilingual-MiniLM` chỉ nhận 128 và sẽ cắt mất gần hết JD.

JD và CV dài hơn 512 token, nên `embed()` **cắt văn bản thành đoạn khoảng 180 từ (chồng nhau 30 từ)**, lấy trung bình vector các đoạn rồi chuẩn hoá lại. Tiền tố `"query: "` được dùng cho cả hai phía, đúng khuyến nghị của e5 cho bài toán so khớp đối xứng.

**Độ phủ kỹ năng.**

```
skill = |kỹ năng tin ∩ kỹ năng CV| / (|kỹ năng tin| + 1)
```

Mẫu số tính theo số kỹ năng của tin, vì câu hỏi cần trả lời là *"bạn đáp ứng được bao nhiêu phần yêu cầu"*. Số **+1** ngăn một tin chỉ có đúng 1 tag "Python" đạt 100% với mọi CV có Python.

**Kết hợp.**
1. Ba tín hiệu có thang khác nhau: TF-IDF thường 0–0,3, embedding thường 0,75–0,9. Vì vậy mỗi tín hiệu được **min-max theo từng CV** (trên 164 tin) về [0, 1] trước khi cộng. Nếu không làm vậy, trọng số sẽ không còn ý nghĩa.
2. Tính `score = Σ wᵢ·tínhiệuᵢ`.
3. Đặt `-inf` cho tin bị loại (nếu bật lọc), rồi sắp xếp.

**Kỹ năng khớp/thiếu.** `skill_gap()` trả `job_skills ∩ CV` và `job_skills − CV`. Trong danh sách thiếu, **tag ITviec đứng trước**, vì đó là kỹ năng nhà tuyển dụng tự đánh dấu là quan trọng.

`recommend()` là API duy nhất mà demo gọi. Trọng số mặc định lấy từ `results/best_weights.json` (do `evaluate.py` ghi ra); nếu chưa chạy đánh giá thì dùng `(0.3, 0.3, 0.4)`.

### 4.5 Gọi LLM: `jobmatch/llm.py`

- **Gọi REST trực tiếp bằng `requests`, không dùng SDK**, để không vướng xung đột phiên bản SDK, và code OpenAI/Gemini có cùng một dạng.
- **JSON mode** ở cả hai bên: `response_format: json_object` với OpenAI, `responseMimeType: application/json` với Gemini. Kết quả trả về là dict, không phải parse văn bản tự do.
- **Cache SQLite** (`data/llm_cache.sqlite`), với key là `sha256(nhà cung cấp | model | temperature | variant | prompt)`:
  - chạy lại `evaluate.py` không tốn tiền cho những phần đã chấm;
  - đổi prompt hoặc model thì key đổi, nên kết quả cũ tự bị bỏ qua;
  - `variant` được dùng để cố ý chấm lại cùng một prompt (đo độ ổn định).
- **Retry** với backoff tăng dần cho lỗi 429/5xx, lỗi mạng và JSON hỏng. Nếu model không nhận `temperature=0` (một số model mới của OpenAI như vậy), code tự gọi lại không kèm tham số này.
- `map_parallel()` chạy nhiều lời gọi song song (`LLM_WORKERS`, mặc định 4). Một lời gọi lỗi chỉ in cảnh báo, không làm hỏng cả đợt.
- **Mỗi vai trò chọn được model riêng.** Có thể ghi `provider` hoặc `provider:model`, ví dụ `JUDGE_MAIN=gemini:gemini-3.5-flash`.
- **Tự chuyển sang Gemini.** `role(name)` gọi thử model của vai trò đó (`healthy()`, 1 request thật, nhớ kết quả trong phiên chạy). Nếu OpenAI không dùng được (hết credit, sai key) thì vai trò chuyển sang model dự phòng trong `FALLBACK_ROLES`: pipeline và giám khảo chính dùng `gemini-3.5-flash`, giám khảo thứ hai dùng `gemini-3.1-flash-lite`, tức hai model khác thế hệ. Nếu không có model nào gọi được, hàm trả `None`. Khi đó demo tự dùng từ điển, còn báo cáo đánh giá tự ghi cảnh báo "cùng nhà cung cấp".
- **Nhiều key Gemini** (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`):
  - Các key được dùng luân phiên.
  - Key nào báo hết quota ngày với một model thì bị bỏ qua cho model đó, request chuyển ngay sang key còn lại.
  - Gặp 429 theo phút mà còn key khác thì đổi key, không phải chờ.
  - Quota miễn phí tính theo **project Google Cloud**, nên key thứ hai phải nằm ở project khác thì mới có thêm quota.
- **Giới hạn tốc độ.**
  - `_throttle()` giãn đều các request theo `GEMINI_RPM` (mặc định 10/phút) cho *từng model*, vì quota Gemini tính riêng theo model.
  - Gặp 429 thì chờ đúng `retryDelay` mà API trả về.
  - Gặp 429 kèm `PerDay` (hết quota ngày) thì dừng ngay và báo rõ, không retry vô ích.
- **Gemini 3** được gửi `thinkingLevel=low` (`GEMINI_THINKING`). Khi đo, `gemini-3.5-flash` mất khoảng 3 giây mỗi lần gọi với low, 10 giây với mức mặc định.

### 4.6 Prompt: `jobmatch/llm_tasks.py`

| Hàm | Dùng ở đâu | Điểm thiết kế |
|---|---|---|
| `extract_jd_skills_batch` | build_index | **Gom `JD_BATCH` = 8 tin vào một lời gọi**, mỗi tin có id `J1…J8`, nên 164 tin chỉ tốn khoảng 21 lời gọi. Danh sách từ vựng chỉ gửi một lần cho cả lô (đỡ token) và giúp LLM dùng đúng tên gốc, nhờ đó kết quả so khớp được với từ điển. Tách `core` và `nice_to_have`. Cấm soft skill. Tin LLM bỏ sót được thử lại một lượt; kết quả cũ trong `jd_skills_llm.json` được giữ theo url, chỉ tin mới mới phải gọi. **Chỉ trích cho tin `eligible`**: tin Senior/Lead/Manager hoặc yêu cầu từ 3 năm kinh nghiệm không hiện trong gợi ý, nên dùng tag + từ điển; cờ `--all-jobs` để trích hết. Lọc theo `eligible` tốt hơn lọc theo danh sách từ "senior/lead/…" trong tên tin: không bỏ nhầm 7 tin như "Fresher/Junior/Senior Tester", và không gọi phí cho 24 tin tên không có từ senior nhưng yêu cầu từ 3 năm |
| `extract_jd_skills` | (bản 1 tin/lời gọi, giữ để tham khảo) | Giống trên nhưng mỗi tin một lời gọi |
| `extract_cv_skills` | chỉ dùng trong đánh giá | Demo **không** gọi hàm này: kỹ năng CV trong demo lấy bằng từ điển |
| `review_cv` | demo, **lời gọi LLM duy nhất cho mỗi CV** | Một lời gọi trả về cùng lúc: nhận xét tổng quan, điểm mạnh/yếu, các lỗi cần sửa trong CV kèm cách sửa, CV hợp với loại vị trí nào, `extra_skills` (kỹ năng CV có mà từ điển bỏ sót) và lộ trình học. Đầu vào gồm top-5 tin đã gợi ý cùng kỹ năng khớp/thiếu, nên nhận xét bám vào đúng thị trường |
| `anonymize_llm` | prepare_cvs | Thay tên người, trường, công ty bằng placeholder; giữ nguyên toàn bộ phần còn lại |
| `roadmap` | chỉ dùng trong đánh giá (demo đã gộp vào `review_cv`) | Đầu vào là kỹ năng thiếu **kèm số tin yêu cầu**, để LLM ưu tiên đúng. Đầu ra là JSON gồm các bước, tài nguyên, mini-project và số tuần |
| `judge_pair` | đánh giá | Rubric 0/1/2. Bắt LLM điền các trường trung gian (số năm, `level_ok`, `domain_ok`, `core_skills`, `matched`, `missing`) **trước** khi ra nhãn, nên nhãn ổn định hơn. Có thể kèm ý kiến của giám khảo kia để chấm lại |
| `judge_roadmap` | đánh giá | 4 tiêu chí, thang 1–5 |
| `synth_cv` | make_synthetic_cvs | Chỉ được nhắc **đúng** các kỹ năng cho trước, viết theo alias được chỉ định, nhúng kỹ năng vào mô tả dự án. temperature 0,9 để các CV đa dạng |

**Tách vai trò LLM** để tránh việc model "tự chấm bài của mình":

| Vai trò | Mặc định | Đổi trong `.env` |
|---|---|---|
| Pipeline (trích kỹ năng, lộ trình học) | Gemini | `PIPELINE_LLM` |
| Giám khảo chính | OpenAI | `JUDGE_MAIN` |
| Giám khảo thứ hai | Gemini | `JUDGE_SECOND` |
| Sinh CV tổng hợp | Xen kẽ OpenAI / Gemini | — |

### 4.7 Đánh giá: `scripts/evaluate.py`

```
1. Nạp Matcher + bộ CV (CV thật nếu có ≥ 10, nếu không thì CV tổng hợp)
2. Tính trước 3 tín hiệu cho mọi cặp CV × tin
3. POOLING: với mỗi CV, gộp
      top-10 của: ngẫu nhiên, 3 tín hiệu đơn, hybrid trọng số đều (có lọc và không lọc)
    + top-5 của cả 66 tổ hợp trọng số (có lọc và không lọc)
4. Hai giám khảo chấm mọi cặp trong pool (song song, có cache)
5. ĐỐI CHIẾU: cặp lệch 2 bậc (0 và 2) → giám khảo chính chấm lại khi được xem lý do của giám khảo kia
6. Chạy 10 hệ thống; với hệ thống cần trọng số: 5-fold theo CV, grid search tối đa NDCG@5 trên 4 fold, đo trên fold còn lại
7. CHẤM BÙ: tin nào lọt top-10 mà chưa có nhãn → chấm → quay lại bước 6 (tối đa 4 vòng)
8. Độ đo + bootstrap CI + Wilcoxon
9. Độ tin cậy: kappa giữa hai giám khảo, độ ổn định (100 cặp × 3 lần), độ nhạy (làm lại bước 6–8 bằng nhãn giám khảo 2)
10. Kỹ năng thiếu so với giám khảo; trích kỹ năng (CV tổng hợp + JD); lộ trình học
11. Ghi results/
```

Các quyết định cần giải thích được khi bảo vệ:

- **Vì sao pooling.** Chấm hết 30 CV × 164 tin ≈ 4.900 cặp × 2 giám khảo thì tốn. Hầu hết tin không lọt top của hệ thống nào, nên chỉ cần chấm những tin có cơ hội xuất hiện. Pool gồm top-5 của *mọi* tổ hợp trọng số, nên lúc chọn trọng số, top-5 luôn đã có nhãn. Bước chấm bù (bước 7) đảm bảo top-10 cuối cùng cũng có đủ nhãn.
- **Vì sao chọn trọng số bằng cross-validation.** Nếu chọn trọng số trên chính bộ test thì Hybrid chắc chắn thắng, vì nó được phép "nhìn đáp án". Chọn trên 4 fold rồi đo trên fold thứ 5 mới cho con số trung thực. `best_weights.json` (chọn trên toàn bộ CV) chỉ dùng cho demo, **không** phải con số báo cáo.
- **Khi các tổ hợp hoà điểm**, lưới đã được sắp theo khoảng cách tới `(⅓, ⅓, ⅓)`, nên trọng số cân bằng được ưu tiên thay vì nghiêng hẳn về một tín hiệu.
- **NDCG@5** dùng gain `2^rel − 1`, nên nhãn 2 có giá trị gấp 3 lần nhãn 1. IDCG tính từ toàn bộ nhãn của CV đó. CV không có tin phù hợp nào bị loại khỏi phép tính trung bình, và số CV bị loại được ghi trong báo cáo.
- **"Tin sai cấp bậc trong top-10"** dùng trường `level_ok` của **giám khảo**, không dùng bộ lọc của hệ thống. Nếu dùng bộ lọc của chính hệ thống thì kết quả luôn là 0%, không đo được gì.
- **Wilcoxon** được dùng thay cho t-test vì chỉ có khoảng 30 CV và NDCG không có phân phối chuẩn. Phép kiểm định ghép cặp theo từng CV.
- **Đo trích kỹ năng từ CV.** CV tổng hợp được sinh *từ* một danh sách kỹ năng, nên danh sách đó chính là đáp án. Nhờ vậy precision và recall tính được mà không cần gán tay.
- **Đo trích kỹ năng từ JD.** Không có đáp án, nên dùng ba chỉ số thay thế:
  - recall so với tag ITviec (tag do người gắn);
  - số kỹ năng mới tìm thêm ngoài tag;
  - grounding rate (kỹ năng trích ra có thật trong JD không).

**Đầu ra:**
- `report.md`: bảng để dán vào báo cáo;
- `metrics.json`: toàn bộ số liệu;
- `per_cv.csv`: kết quả từng CV, dùng để phân tích lỗi;
- `judgments.jsonl`: mọi nhãn kèm lý do, để minh bạch;
- `best_weights.json`.

### 4.8 Demo: `app.py`

App có ba tab:
- **Gợi ý việc làm**, theo thứ tự từ trên xuống:
  - Phần đầu: tiêu đề và dòng trạng thái (số tin, số tin AI đã phân tích, model AI đang dùng).
  - Ô tải CV (PDF) hoặc dán văn bản, kèm nút **"Thử với CV mẫu"** để demo (có thể mở thẳng bằng `?sample=CV mẫu Data Engineer`).
  - Hai cột: bên trái là **kỹ năng trong CV** (từ điển) và **nên học tiếp** (thanh ngang, đếm số tin cần kỹ năng đó); bên phải là **Nhận xét từ AI**.
  - Bên dưới là **thẻ tin** cho từng tin: hạng, tên tin (link về ITviec), công ty, thành phố, ngày đăng, nhãn cấp bậc và số năm kinh nghiệm. Mỗi thẻ có ba thanh điểm *Giống về nghĩa / Trùng từ khoá / Đủ kỹ năng x/y*, kỹ năng **Đã có** (nền xanh) và **Còn thiếu** (viền đứt).
  - Kỹ năng do AI trích từ JD, hoặc AI phát hiện thêm trong CV, có nhãn nhỏ **AI**.
- **Nhận xét từ AI:** nút "Nhận xét CV của tôi" gọi `review_cv` **đúng một lần**. Kết quả được lưu trong `st.session_state` theo hash của CV, nên Streamlit chạy lại cũng không gọi lại. Khi hết quota, app báo bằng câu dễ hiểu thay vì lỗi kỹ thuật.
- **Thị trường:** biểu đồ 20 kỹ năng được yêu cầu nhiều nhất (một màu, có tooltip, xem được dạng bảng), chọn phạm vi "tin hợp sinh viên" hoặc "tất cả", kèm bảng cấp bậc.
- **Đánh giá:** hiển thị `results/report.md`.

Sidebar có: lọc cấp bậc, số tin hiển thị, trọng số (gấp trong mục "Trọng số xếp hạng") và một đoạn ngắn "Cách hệ thống chấm điểm". App **không** kiểm tra API key lúc khởi động, vì việc đó tốn quota. Khi bấm nút, app thử model cấu hình trước, lỗi thì thử model Gemini dự phòng.

**Giao diện:**
- Font Be Vietnam Pro (thiết kế cho tiếng Việt); không dùng emoji.
- Màu khai báo bằng một bảng token cho chế độ sáng và tối, chọn theo `st.context.theme`. Màu nhấn là xanh `#2a78d6`.
- Thẻ tin, chip kỹ năng và thanh điểm là HTML/CSS tự viết; mọi nội dung động đều qua `html.escape`.
- Theme chung đặt trong `.streamlit/config.toml`.

Hai chi tiết kỹ thuật:
- `Matcher` và model embedding được nạp **một lần** nhờ `st.cache_resource`.
- Khối tab "Gợi ý" được đặt **cuối file**, vì nó gọi `st.stop()` khi chưa có CV. Nếu đặt đầu file, hai tab còn lại sẽ không bao giờ được vẽ.

---

## 5. Bảng quyết định và đánh đổi

| Quyết định | Phương án khác | Lý do chọn |
|---|---|---|
| JD: LLM trích kỹ năng theo lô. CV: từ điển, cộng các kỹ năng AI phát hiện thêm trong lần nhận xét | Mỗi tin/CV một lời gọi riêng | Quota miễn phí tính theo **số request** (20/ngày/model), không theo token, nên gom lô giảm số lời gọi khoảng 8 lần. Phía CV, gộp trích kỹ năng vào lời gọi nhận xét, nên mỗi CV chỉ tốn 1 lời gọi. Từ điển vẫn cần để chuẩn hoá tên, chạy khi không có LLM và làm mốc so sánh |
| Gom nhiều tin vào một prompt | Một tin một prompt | Rủi ro: LLM có thể bỏ sót tin hoặc trộn kỹ năng giữa các tin. Giảm rủi ro bằng id cho từng tin, yêu cầu "treat every job independently", và thử lại các tin bị bỏ sót |
| e5-small | bge-m3, e5-large | Chạy trên CPU laptop trong vài giây, nhận 512 token; model lớn chậm hơn 3–5 lần |
| Min-max theo từng CV | z-score, rank | Đơn giản, đưa về [0, 1] nên trọng số dễ diễn giải |
| Lọc cứng theo cấp bậc | Trừ điểm mềm | Sinh viên không nộp được tin Senior thì tin đó không nên xuất hiện. Ablation "không lọc" đo tác dụng của bước này |
| Giám khảo chấm từng cặp | Đưa cả danh sách cho LLM xếp hạng | Tránh thiên vị vị trí, dễ cache, dễ tính kappa |
| Pooling + chấm bù | Chấm toàn bộ cặp | Rẻ hơn 3–5 lần, nhưng vẫn đảm bảo các tin trong top được đo đều có nhãn |
| REST + cache tự viết | SDK, LangChain | Ít phụ thuộc, kiểm soát được cache key và retry |

---

## 6. Tình trạng hiện tại (08/11/2026)

**Đã chạy và kiểm tra:**
- **Crawl lại** với crawler đã sửa: 164/164 tin, không lỗi tin nào, trường `posted` đã có dữ liệu.
- **`build_index.py`:** 59/164 tin qua bộ lọc. LLM chỉ trích được kỹ năng cho **26/164 tin**, vì Gemini hết quota (xem dưới); 138 tin còn lại tạm dùng từ điển.
- **Demo Streamlit** (chạy bằng `AppTest`, không có lỗi): với CV mẫu Data Engineer, top 5 là các tin Data Engineer hoặc Data Analyst, kỹ năng khớp và thiếu hợp lý. LLM không gọi được nên app tự dùng từ điển.
- **Smoke test toàn bộ `make_synthetic_cvs` và `evaluate.py`** bằng LLM giả: chạy thông mọi bước (pool khoảng 38 cặp/CV, kappa, độ ổn định, độ nhạy, báo cáo). Số liệu của LLM giả không có ý nghĩa nên không được lưu vào `results/`.
- **Trích kỹ năng JD theo lô** (LLM giả): 164 tin → 24 lời gọi, gồm 21 lô và 3 lời gọi thử lại.
- **Luân phiên key Gemini** (API giả): key 1 báo hết quota ngày thì mọi request sau đó đi qua key 2; khi mọi key đều hết quota thì báo lỗi rõ ràng.
- **Nút "Nhận xét CV bằng AI"** (LLM giả, `AppTest`): đúng 1 lời gọi LLM; kỹ năng AI phát hiện thêm được tô tím và có thể dùng để xếp hạng lại.

**Phần đánh giá: tạm hoãn** theo quyết định ngày 08/11/2026. `evaluate.py` giữ nguyên, chưa chạy với LLM thật. Với cấu hình hiện tại, nó cần khoảng 2.800 lời gọi, trong đó khoảng 80% là chấm từng cặp CV–tin. Các cách giảm đã liệt kê (gom cặp vào một lời gọi, 20 CV, pool top-5, giám khảo 2 chỉ chấm một mẫu) **chưa** được áp dụng.

**Quota API:**
- **OpenAI:** key hợp lệ nhưng hết credit, nên code tự chuyển sang Gemini.
- **Gemini:** gói miễn phí, **20 request/ngày cho mỗi model và mỗi project**. Hôm nay đã dùng hết; quota được làm mới sau khoảng 17 tiếng (tính từ 08/11 tối).

**Việc cần làm tiếp:**
1. *(Tuỳ chọn)* điền `GEMINI_API_KEY_2` trong `.env`, dùng key tạo ở **project khác**.
2. `python scripts/build_index.py`: trích nốt **49 tin hợp sinh viên** còn lại, khoảng **7 lời gọi**. 89 tin Senior/≥3 năm được bỏ qua và dùng tag + từ điển.
3. `streamlit run app.py`: mỗi lần bấm "Nhận xét CV bằng AI" tốn 1 lời gọi.
