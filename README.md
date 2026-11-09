# Gợi ý việc làm IT cho sinh viên từ CV

> Ý tưởng, luồng hoạt động và giải thích kỹ thuật chi tiết: [docs/THIET_KE.md](docs/THIET_KE.md)

Nhận CV (PDF) của sinh viên, trả về top 10 tin tuyển dụng AI/ML và Data Engineering đang mở trên ITviec phù hợp nhất. Mỗi tin kèm điểm phù hợp, kỹ năng đã khớp, kỹ năng còn thiếu, và một lộ trình học do LLM gợi ý. Chất lượng gợi ý được đo bằng Precision@5 và NDCG@5, nhãn do LLM chấm (LLM-as-a-judge, dùng 2 giám khảo khác nhà cung cấp).

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate             # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # rồi điền OPENAI_API_KEY và GEMINI_API_KEY
```

Chạy trên CPU, không cần GPU. Lần đầu sẽ tải model embedding `intfloat/multilingual-e5-small` (khoảng 470 MB).

## Chạy từng bước

| # | Lệnh | Kết quả |
|---|---|---|
| 1 | `cd data && python ../itviec_crawler.py && cd ..` | `data/itviec_jobs.json` (khoảng 164 tin, ~8 phút) |
| 2 | `python scripts/build_index.py` | `data/jobs_processed.json`, `data/job_embeddings.npy`, `data/jd_skills_llm.json` |
| 3a | *(có CV thật)* bỏ PDF vào `data/cvs_raw/`, chạy `python scripts/prepare_cvs.py` | `data/cvs/cv_XX.txt` đã ẩn danh |
| 3b | `python scripts/make_synthetic_cvs.py --n 40` | `data/synthetic_cvs.json` (CV giả có đáp án kỹ năng) |
| 4 | `python scripts/evaluate.py` | `results/report.md`, `metrics.json`, `best_weights.json`, `per_cv.csv`, `judgments.jsonl` |
| 5 | `streamlit run app.py` | Demo web ở http://localhost:8501 |

- Bước 3b luôn nên chạy, vì CV tổng hợp là thứ dùng để đo precision/recall của bước trích kỹ năng.
- Nếu có từ 10 CV thật trở lên, `evaluate.py` dùng CV thật để đánh giá xếp hạng; nếu ít hơn thì dùng CV tổng hợp. Có thể chọn tay bằng `--cvs real|synthetic`.
- `build_index.py --no-llm` chạy được khi chưa có API key, lúc đó chỉ dùng tag ITviec và từ điển. Demo vẫn chạy, chỉ tắt các tính năng LLM.
- Mọi lời gọi LLM được cache trong `data/llm_cache.sqlite`, nên chạy lại chỉ tốn tiền cho phần mới.
- `python scripts/make_sample_pdfs.py` tạo PDF từ các CV mẫu trong `examples/` để thử demo.

## Cấu trúc

```
itviec_crawler.py          crawl ITviec (2 danh mục AI/ML Engineer, Data Engineer)
jobmatch/
  config.py                đường dẫn, model, vai trò LLM, trọng số mặc định
  skills.py                từ điển kỹ năng = tag ITviec + ~100 nhóm alias Việt/Anh, trích bằng regex
  jobs.py                  suy ra cấp bậc và số năm kinh nghiệm, lọc tin Intern/Fresher/Junior
  cv.py                    đọc PDF (pdfplumber), ẩn danh email/SĐT/link/ngày sinh
  matcher.py               3 tín hiệu + hybrid + kỹ năng khớp/thiếu
  llm.py                   client OpenAI/Gemini (REST), JSON mode, cache, retry
  llm_tasks.py             prompt: trích kỹ năng JD/CV, ẩn danh, lộ trình học, giám khảo, sinh CV
scripts/                   build_index, prepare_cvs, make_synthetic_cvs, evaluate, make_sample_pdfs
app.py                     demo Streamlit (gợi ý, tổng quan thị trường, kết quả đánh giá)
```

## Phương pháp

**Lọc cấp bậc.** Cấp bậc lấy từ tên tin: lấy mức thấp nhất được nhắc tới, ví dụ "Junior/Middle" thành junior. Số năm kinh nghiệm tối thiểu lấy bằng regex trên những dòng của phần yêu cầu có nhắc tới "experience/kinh nghiệm". Một tin được giữ lại nếu thoả một trong hai điều kiện:
- có nhãn "Fresher/Internship Accepted";
- không phải Senior/Lead và yêu cầu không quá 2 năm kinh nghiệm.

**Ba tín hiệu.** Mỗi tín hiệu được min-max theo từng CV, rồi cộng theo trọng số `w1·tfidf + w2·emb + w3·skill`:
1. **TF-IDF:** unigram và bigram, sublinear tf, cosine giữa CV và phần tên tin, tag, yêu cầu, mô tả.
2. **Embedding:** `multilingual-e5-small`. Văn bản dài được cắt thành đoạn khoảng 180 từ, sau đó lấy trung bình vector các đoạn.
3. **Kỹ năng:** `|kỹ năng khớp| / (|kỹ năng tin yêu cầu| + 1)`. Kỹ năng của tin gồm tag ITviec và kỹ năng LLM trích từ phần yêu cầu. Cả hai phía đều được chuẩn hoá qua alias, ví dụ pyspark → Apache Spark, học máy → Machine Learning.

## Đánh giá (không gán nhãn tay)

- **Pooling:** với mỗi CV, gộp top-10 của mọi hệ thống và top-5 của cả 66 tổ hợp trọng số. Tin nào lọt vào top-10 sau khi chọn trọng số mà chưa có nhãn thì được chấm bù.
- **Giám khảo:** giám khảo chính là OpenAI, giám khảo thứ hai là Gemini, còn pipeline dùng Gemini. Chấm từng cặp, mù, temperature 0, theo rubric 0/1/2. Cặp nào hai giám khảo lệch 2 bậc thì giám khảo chính chấm lại sau khi xem lý do của giám khảo kia.
- **Độ đo:** P@5 (tính nhãn ≥ 1), NDCG@5 (gain = 2^rel − 1), tỷ lệ tin sai cấp bậc trong top-10, khoảng tin cậy 95% bằng bootstrap.
- **Chọn trọng số:** 5-fold cross-validation theo CV, grid search bước 0.1. Kiểm định Wilcoxon giữa Hybrid và tín hiệu đơn tốt nhất.
- **Độ tin cậy nhãn:** weighted kappa giữa hai giám khảo, độ ổn định khi chấm lại 3 lần, và kiểm tra độ nhạy bằng cách tính lại mọi bảng với nhãn của giám khảo thứ hai.
- **Trích kỹ năng:** trên CV tổng hợp, đo P/R/F1 so với đáp án, cho từ điển và cho LLM. Trên JD, đo recall so với tag ITviec, số kỹ năng mới tìm thêm, và grounding rate (tỷ lệ kỹ năng trích ra thật sự xuất hiện trong JD).
- **Lộ trình học:** giám khảo chấm thang 1–5 theo 4 tiêu chí.

## Deploy (Streamlit Community Cloud)

1. Commit cả `data/jobs_processed.json`, `data/job_embeddings.npy` và `results/` (không commit `.env`, `data/cvs*`, cache).
2. Trên share.streamlit.io, chọn repo và file `app.py`.
3. Ở mục *Secrets*, thêm `OPENAI_API_KEY = "..."` và `GEMINI_API_KEY = "..."`. Streamlit đưa secret cấp gốc vào biến môi trường, nên `config.py` đọc được luôn.
