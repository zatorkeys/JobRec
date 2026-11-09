"""Cấu hình chung. Mọi giá trị có thể ghi đè bằng biến môi trường hoặc file .env ở thư mục gốc."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _load_dotenv(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# ---- Dữ liệu ----
JOBS_RAW = DATA / "itviec_jobs.json"          # output của itviec_crawler.py
JOBS_PROCESSED = DATA / "jobs_processed.json"  # đã thêm cấp bậc, kỹ năng chuẩn hoá, kỹ năng LLM
JOB_EMB = DATA / "job_embeddings.npy"
CV_DIR = DATA / "cvs"                          # CV thật đã ẩn danh (.txt), tạo bởi scripts/prepare_cvs.py
CV_RAW_DIR = DATA / "cvs_raw"                  # CV thật gốc (.pdf) — KHÔNG commit
SYNTH_CVS = DATA / "synthetic_cvs.json"        # CV tổng hợp có danh sách kỹ năng gold
LLM_CACHE = DATA / "llm_cache.sqlite"

# ---- LLM ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
# Nhiều key Gemini: dùng luân phiên, key nào hết quota ngày thì tự chuyển sang key khác.
# Quota miễn phí tính theo PROJECT → key thứ 2 phải tạo ở project Google Cloud khác mới có thêm quota.
GEMINI_API_KEYS = [k for k in (GEMINI_API_KEY, os.getenv("GEMINI_API_KEY_2", ""), os.getenv("GEMINI_API_KEY_3", ""))
                   if k]
JD_BATCH = int(os.getenv("JD_BATCH", "8"))  # số tin gom vào 1 lời gọi khi LLM trích kỹ năng JD
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low")  # mức "suy nghĩ" của Gemini 3; để trống = mặc định

# Vai trò: pipeline (trích kỹ năng, lộ trình học) dùng một nhà cung cấp,
# giám khảo chính dùng nhà cung cấp KIA để tránh "tự chấm cho mình".
# Mỗi vai trò có thể ghi "provider" hoặc "provider:model", vd. JUDGE_MAIN=gemini:gemini-3.8-flash
PIPELINE_LLM = os.getenv("PIPELINE_LLM", "gemini")
JUDGE_MAIN = os.getenv("JUDGE_MAIN", "openai")
JUDGE_SECOND = os.getenv("JUDGE_SECOND", "gemini")
# Dự phòng khi vai trò cấu hình không gọi được (vd. OpenAI hết credit) → mọi vai trò dùng Gemini.
# Hai giám khảo dùng 2 model Gemini KHÁC THẾ HỆ để bớt chấm giống nhau.
FALLBACK_ROLES = {
    "pipeline": os.getenv("FALLBACK_PIPELINE", "gemini"),
    "judge_main": os.getenv("FALLBACK_JUDGE_MAIN", "gemini:gemini-3.5-flash"),
    "judge_second": os.getenv("FALLBACK_JUDGE_SECOND", "gemini:gemini-3.1-flash-lite"),
}
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "4"))
# Giới hạn request/phút cho MỖI model (gói miễn phí Gemini khoảng 10–15 RPM). 0 = không giới hạn
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "10"))
OPENAI_RPM = float(os.getenv("OPENAI_RPM", "0"))

# ---- Matching ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# Trọng số mặc định (TF-IDF, embedding, kỹ năng). scripts/evaluate.py ghi đè bằng results/best_weights.json
DEFAULT_WEIGHTS = (0.3, 0.3, 0.4)
BEST_WEIGHTS_FILE = RESULTS / "best_weights.json"
MAX_YEARS_FOR_STUDENT = 2   # tin yêu cầu >= 3 năm kinh nghiệm bị loại
TOP_K = 10
