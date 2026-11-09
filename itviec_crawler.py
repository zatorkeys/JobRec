"""
Crawl tin tuyển dụng ITviec: trang danh mục → lấy link các tin → mở từng tin → lưu itviec_jobs.json

Chạy:     python itviec_crawler.py
Notebook: from itviec_crawler import main; jobs = main()
"""
import json
import re
import time
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

CATEGORIES = [
    "https://itviec.com/it-jobs/ai-machine-learning-engineer",
    "https://itviec.com/it-jobs/data-engineer",
]
HEADERS = {"User-Agent": "student-job-matcher/0.1", "Accept-Language": "en-US,en;q=0.8"}
DELAY = 0.5  # giây nghỉ giữa các request, để không bị chặn
SECTIONS = ["Top 3 reasons to join us", "Job description",
            "Your skills and experience", "Why you'll love working here"]


def get_soup(url):
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def get_job_links(category_url):
    """Mỗi trang chỉ có 20 tin nên duyệt ?page=1, 2, ... tới khi không còn link mới."""
    links, page = [], 1
    while True:
        try:
            soup = get_soup(f"{category_url}?page={page}")
        except requests.HTTPError:
            return links
        new = []
        for a in soup.select('a[href*="/it-jobs/"][href*="lab_feature=preview_jd_page"]'):
            url = urljoin(category_url, a["href"].split("?")[0])
            if url not in links and url not in new:
                new.append(url)
        if not new:
            return links
        links += new
        page += 1


def between(lines, start, stops):
    """Các dòng sau dòng `start`, dừng ở dòng đầu tiên nằm trong `stops`."""
    if start not in lines:
        return []
    out = []
    for line in lines[lines.index(start) + 1:]:
        if line in stops:
            break
        out.append(line)
    return out


def posted_days(text):
    """'Posted 5 days ago' → 5, 'Posted 14 hours ago' → 0.58"""
    m = re.search(r"(\d+|an?) (minute|hour|day|week|month)", text or "")
    if not m:
        return None
    n = 1 if m[1] in ("a", "an") else int(m[1])
    return round(n * {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30}[m[2]], 2)


def parse_job(url):
    soup = get_soup(url)
    lines = [s.replace("’", "'") for s in soup.get_text("\n", strip=True).split("\n")]
    title = soup.h1.get_text(" ", strip=True)
    og_title = soup.find("meta", property="og:title")["content"]  # "<tin> at <công ty> | ITviec"
    company = og_title.rsplit(" | ITviec", 1)[0].rsplit(" at ", 1)[-1]

    # Phần đầu trang (từ tên tin tới mục đầu tiên): ngày đăng, skills, cấp bậc...
    start = lines.index(title) if title in lines else 0
    end = next((i for i, s in enumerate(lines) if s in SECTIONS), len(lines))
    head = lines[start:end]
    # ITviec tách "Posted" và "1 day ago" thành 2 dòng; vẫn nhận dạng cũ "Posted 1 day ago" trên 1 dòng
    posted = next((f"Posted {head[i + 1]}" for i, s in enumerate(head[:-1]) if s == "Posted"),
                  next((s for s in head if re.match(r"Posted .+ ago", s)), None))
    maps = soup.select_one('a[href*="google.com/maps"]')
    stops = SECTIONS + ["Skills:", "Job Expertise:", "Job Domain:",
                        company, "Company type", "More jobs for you"]

    return {
        "url": url,
        "title": title,
        "company": company,
        "posted": posted,
        "posted_days": posted_days(posted),
        "address": parse_qs(urlparse(maps["href"]).query).get("q", [None])[0] if maps else None,
        "working_model": next((s for s in head if s in ("At office", "Hybrid", "Remote")), None),
        "fresher": "Fresher Accepted" in head,
        "internship": "Internship Accepted" in head,
        "skills": between(head, "Skills:", stops),
        "domains": between(head, "Job Domain:", stops),
        "reasons": "\n".join(between(lines, SECTIONS[0], stops)),
        "description": "\n".join(between(lines, SECTIONS[1], stops)),
        "requirements": "\n".join(between(lines, SECTIONS[2], stops)),
        "benefits": "\n".join(between(lines, SECTIONS[3], stops)),
    }


def main(out_file="itviec_jobs.json"):
    links = []
    for cat in CATEGORIES:
        found = get_job_links(cat)
        print(f"{cat}: {len(found)} tin")
        links += [u for u in found if u not in links]  # 1 tin có thể nằm ở cả 2 danh mục

    jobs = []
    for i, url in enumerate(links, 1):
        try:
            jobs.append(parse_job(url))
            print(f"[{i}/{len(links)}] {jobs[-1]['title']}")
        except Exception as e:
            print(f"[{i}/{len(links)}] lỗi {url}: {e}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(jobs)} tin vào {out_file}")
    return jobs


if __name__ == "__main__":
    main()
