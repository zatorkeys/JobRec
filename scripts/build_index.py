"""Bước 2 (sau khi crawl): xử lý kho tin → data/jobs_processed.json + data/job_embeddings.npy

    python scripts/build_index.py            # có API key: LLM trích kỹ năng sâu từ phần yêu cầu (gom JD_BATCH tin/lời gọi)
    python scripts/build_index.py --no-llm   # chỉ dùng tag ITviec + từ điển
    python scripts/build_index.py --refresh  # trích lại toàn bộ bằng LLM (mặc định giữ kết quả cũ theo url)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobmatch import config, llm, llm_tasks  # noqa: E402
from jobmatch.jobs import load_raw, process  # noqa: E402
from jobmatch.matcher import embed  # noqa: E402
from jobmatch.skills import SkillExtractor  # noqa: E402

LLM_SKILLS_FILE = config.DATA / "jd_skills_llm.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="bỏ kết quả LLM cũ, trích lại toàn bộ")
    ap.add_argument("--all-jobs", action="store_true",
                    help="trích bằng LLM cả tin Senior/Lead (mặc định chỉ tin hợp sinh viên)")
    args = ap.parse_args()

    raw = load_raw()
    print(f"Đọc {len(raw)} tin từ {config.JOBS_RAW.name}")
    extractor = SkillExtractor.from_jobs(raw)

    # Kết quả LLM của lần chạy trước được giữ lại (theo url) → chỉ gọi LLM cho tin mới / tin còn thiếu
    detail = {}
    if LLM_SKILLS_FILE.exists() and not args.refresh:
        old = json.loads(LLM_SKILLS_FILE.read_text(encoding="utf-8"))
        detail = {j["url"]: old[j["url"]] for j in raw if j["url"] in old}
    # Chỉ gọi LLM cho tin sinh viên nộp được (eligible: không Senior/Lead/Manager..., ≤ 2 năm KN, hoặc có nhãn
    # Fresher/Internship) — tin bị lọc không hiện trong gợi ý nên không cần trích sâu, đỡ tốn quota.
    eligible = {j["url"]: j["eligible"] for j in process(raw, None, extractor)}

    def need_llm(j):
        return j["url"] not in detail and (args.all_jobs or eligible[j["url"]])

    todo = [j for j in raw if need_llm(j)]
    skipped = sum(1 for j in raw if j["url"] not in detail and not eligible[j["url"]]) if not args.all_jobs else 0
    if skipped:
        print(f"Bỏ qua LLM cho {skipped} tin không hợp sinh viên (Senior/Lead, ≥3 năm KN) → dùng tag + từ điển")
    pipe = None if args.no_llm or not todo else llm.role("pipeline")
    if pipe:
        vocab = extractor.vocabulary()
        bs = config.JD_BATCH
        for attempt in (1, 2):  # lượt 2: thử lại các tin LLM bỏ sót / lô bị lỗi
            batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
            print(f"LLM ({llm.model_name(pipe)}) trích kỹ năng {len(todo)} tin, {bs} tin/lời gọi → "
                  f"{len(batches)} lời gọi" + (" (thử lại)" if attempt == 2 else ""))
            for res in llm.map_parallel(lambda b: llm_tasks.extract_jd_skills_batch(b, vocab, pipe), batches,
                                        desc="JD batch"):
                detail.update(res or {})
            LLM_SKILLS_FILE.write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
            todo = [j for j in raw if need_llm(j)]
            if not todo:
                break
    elif todo and not args.no_llm:
        print("! Không có LLM nào dùng được → các tin chưa có kết quả LLM dùng tag + từ điển")
    if todo:
        print(f"! {len(todo)} tin hợp sinh viên chưa có kết quả LLM → dùng từ điển. Chạy lại script để bổ sung.")
    print(f"Kỹ năng LLM: {len(detail)}/{len(raw)} tin "
          f"({sum(1 for u in detail if eligible.get(u))}/{sum(eligible.values())} tin hợp sinh viên)")
    llm_skills = {u: r["core"] + r["nice_to_have"] for u, r in detail.items()} or None

    jobs = process(raw, llm_skills, extractor)
    config.JOBS_PROCESSED.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Tính embedding cho các tin...")
    np.save(config.JOB_EMB, embed([j["text"] for j in jobs]))

    lv = Counter(j["level"] or "không ghi" for j in jobs)
    print(f"\nĐã lưu {len(jobs)} tin → {config.JOBS_PROCESSED.name}")
    print("Cấp bậc theo tên tin:", dict(lv))
    print(f"Phù hợp sinh viên (sau lọc): {sum(j['eligible'] for j in jobs)} tin "
          f"(Fresher Accepted: {sum(j['fresher'] for j in jobs)}, Internship: {sum(j['internship'] for j in jobs)})")
    top = Counter(s for j in jobs if j["eligible"] for s in j["job_skills"]).most_common(15)
    print("Kỹ năng được yêu cầu nhiều nhất:", ", ".join(f"{s} ({n})" for s, n in top))


if __name__ == "__main__":
    main()
