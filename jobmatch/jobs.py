"""Xử lý kho tin: suy ra cấp bậc + số năm kinh nghiệm, lọc tin hợp với sinh viên, chuẩn hoá kỹ năng."""
import json
import re

from . import config
from .skills import SkillExtractor, canonical_set

LEVELS = ["intern", "fresher", "junior", "middle", "senior"]
_LEVEL_RX = {
    "intern": r"\b(intern|internship|thực tập|trainee)\b",
    "fresher": r"\b(fresher|graduate|mới tốt nghiệp)\b",
    "junior": r"\b(junior|jr\.?)(?=\W|$)",
    "middle": r"\b(middle|mid-level|mid level|mid)\b",
    "senior": r"\b(senior|sr\.?|lead|leader|principal|head|manager|architect|staff|chief|director|expert|"
              r"chuyên gia|trưởng nhóm|trưởng phòng)(?=\W|$)",
}
_YEARS_RX = re.compile(
    r"(?<![\d.])(\d{1,2}(?:[.,]5)?)\s*\+?\s*(?:(?:-|–|~|to|đến|or|hoặc)\s*\d{1,2}\s*\+?\s*)?"
    r"(?:years?|yrs?|năm)", re.IGNORECASE)
_EXP_RX = re.compile(r"experience|exp\b|kinh nghiệm|worked|working", re.IGNORECASE)


def infer_level(title):
    """Cấp thấp nhất xuất hiện trong tên tin ("Junior/Middle" → junior); không có → None."""
    t = title.lower()
    found = [lv for lv in LEVELS if re.search(_LEVEL_RX[lv], t)]
    return found[0] if found else None


def required_years(text):
    """Số năm kinh nghiệm tối thiểu: cận dưới lớn nhất trong các cụm "N năm" có nhắc tới kinh nghiệm
    trong vòng ~80 ký tự. Nối dòng trước vì JD ITviec hay ngắt giữa câu ("Minimum\\n7 years\\nof experience")."""
    flat = re.sub(r"\s+", " ", text or "")
    best = None
    for m in _YEARS_RX.finditer(flat):
        if not _EXP_RX.search(flat[max(0, m.start() - 80): m.end() + 80]):
            continue
        y = float(m.group(1).replace(",", "."))
        if 0 < y <= 15:
            best = y if best is None else max(best, y)
    return best


def is_student_friendly(job):
    if job.get("fresher") or job.get("internship"):
        return True
    if job["level"] == "senior":
        return False
    y = job["required_years"]
    return y is None or y <= config.MAX_YEARS_FOR_STUDENT


def job_text(job):
    return "\n".join([job["title"], ", ".join(job.get("skills", [])),
                      job.get("requirements", ""), job.get("description", "")])


def process(raw_jobs, llm_skills=None, extractor=None):
    """llm_skills: {url: [kỹ năng]} từ LLM (tuỳ chọn). Trả list job đã thêm các trường mới."""
    extractor = extractor or SkillExtractor.from_jobs(raw_jobs)
    seen, out = set(), []
    for j in raw_jobs:
        if j["url"] in seen:
            continue
        seen.add(j["url"])
        j = dict(j)
        j["level"] = infer_level(j["title"])
        j["required_years"] = required_years(j.get("requirements", ""))
        j["eligible"] = is_student_friendly(j)
        j["skills_tag"] = sorted(canonical_set(j.get("skills", [])))
        j["skills_dict"] = sorted(extractor.extract(j.get("requirements", "") + "\n" + j.get("description", "")))
        llm = (llm_skills or {}).get(j["url"])
        j["skills_llm"] = sorted(canonical_set(llm)) if llm is not None else None
        extra = j["skills_llm"] if j["skills_llm"] is not None else j["skills_dict"]
        j["job_skills"] = sorted(set(j["skills_tag"]) | set(extra))
        j["text"] = job_text(j)
        out.append(j)
    return out


def load_raw(path=config.JOBS_RAW):
    return json.loads(path.read_text(encoding="utf-8"))


def load_processed(path=config.JOBS_PROCESSED):
    return json.loads(path.read_text(encoding="utf-8"))
