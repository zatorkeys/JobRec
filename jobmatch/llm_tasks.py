"""Các tác vụ dùng LLM. Mỗi hàm nhận `provider` ("openai" | "gemini") và trả dict đã parse."""
import json

from . import config
from . import llm
from .llm import ask_json


def _cut(text, n):
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n[...]"


# ---------------------------------------------------------------- trích kỹ năng
JD_SKILLS_PROMPT = """Extract the technical and professional skills REQUIRED by this IT job posting.
Use the "Requirements" section as the main source; the description only for context.
When a skill matches a name in the VOCABULARY, use that exact name; otherwise use the common English name.
Split combined items ("Python/Scala" → "Python", "Scala"). Do not include soft skills like "teamwork",
"communication", "problem solving". Do not invent skills that are not stated.

VOCABULARY: {vocab}

Return JSON: {{"core": ["must-have skills"], "nice_to_have": ["preferred / plus skills"]}}

JOB TITLE: {title}
REQUIREMENTS:
{requirements}
DESCRIPTION:
{description}"""


def extract_jd_skills(job, vocab, provider=None):
    p = JD_SKILLS_PROMPT.format(vocab=", ".join(vocab), title=job["title"],
                                requirements=_cut(job.get("requirements"), 4000),
                                description=_cut(job.get("description"), 2000))
    d = ask_json(provider or llm.role("pipeline"), p)
    return {"core": list(d.get("core", [])), "nice_to_have": list(d.get("nice_to_have", []))}


JD_BATCH_PROMPT = """Extract the technical and professional skills REQUIRED by EACH of the IT job postings below.
Treat every job independently. Use its "Requirements" as the main source; its description only for context.
When a skill matches a name in the VOCABULARY, use that exact name; otherwise use the common English name.
Split combined items ("Python/Scala" → "Python", "Scala"). No soft skills ("teamwork", "communication",
"problem solving"). Do not invent skills that are not stated.

VOCABULARY: {vocab}

Return JSON with one entry per job, using the job ids given:
{{"jobs": [{{"id": "J1", "core": ["must-have skills"], "nice_to_have": ["preferred / plus skills"]}}, ...]}}

{jobs}"""


def extract_jd_skills_batch(jobs, vocab, provider=None):
    """Gom nhiều tin vào 1 lời gọi. Trả {url: {"core": [...], "nice_to_have": [...]}} — tin nào LLM bỏ sót thì vắng."""
    blocks = [f"=== JOB J{i} ===\nTITLE: {j['title']}\nREQUIREMENTS:\n{_cut(j.get('requirements'), 2500)}\n"
              f"DESCRIPTION:\n{_cut(j.get('description'), 800)}" for i, j in enumerate(jobs, 1)]
    d = ask_json(provider or llm.role("pipeline"),
                 JD_BATCH_PROMPT.format(vocab=", ".join(vocab), jobs="\n\n".join(blocks)))
    out = {}
    for item in d.get("jobs", []):
        try:
            j = jobs[int(str(item.get("id", "")).lstrip("Jj")) - 1]
        except (ValueError, IndexError):
            continue
        out[j["url"]] = {"core": list(item.get("core", [])), "nice_to_have": list(item.get("nice_to_have", []))}
    return out


CV_SKILLS_PROMPT = """List every technical and professional skill this candidate demonstrably has, based on the CV
(skills section, projects, experience, courses). CV may mix Vietnamese and English.
When a skill matches a name in the VOCABULARY, use that exact name; otherwise use the common English name.
Only skills actually stated or clearly used in a project. No soft skills.

VOCABULARY: {vocab}

Return JSON: {{"skills": ["..."]}}

CV:
{cv}"""


def extract_cv_skills(cv_text, vocab, provider=None):
    d = ask_json(provider or llm.role("pipeline"),
                 CV_SKILLS_PROMPT.format(vocab=", ".join(vocab), cv=_cut(cv_text, 8000)))
    return list(d.get("skills", []))


# ---------------------------------------------------------------- ẩn danh CV
ANON_PROMPT = """Anonymize this CV. Replace: person names → [NAME]; university/school names → [SCHOOL];
company names where the candidate worked/interned → [COMPANY]; street addresses → [ADDRESS];
emails, phones, links → [CONTACT]. Keep EVERYTHING else exactly as written (skills, projects, dates, grades,
technologies). Do not summarize or translate.
Return JSON: {{"text": "<anonymized CV>"}}

CV:
{cv}"""


def anonymize_llm(cv_text, provider=None):
    return ask_json(provider or llm.role("pipeline"), ANON_PROMPT.format(cv=_cut(cv_text, 12000)))["text"]


# ---------------------------------------------------------------- lộ trình học
ROADMAP_PROMPT = """Bạn là mentor cho sinh viên IT Việt Nam muốn ứng tuyển vị trí Intern/Fresher/Junior.
Kỹ năng hiện có của sinh viên: {have}
Các kỹ năng còn thiếu, kèm số tin trong top gợi ý yêu cầu kỹ năng đó: {missing}
Các vị trí mục tiêu: {targets}

Lập lộ trình học 8–12 tuần, ưu tiên kỹ năng được nhiều tin yêu cầu và phù hợp nền tảng hiện có.
Mỗi bước: kỹ năng, lý do, nội dung cần học cụ thể, 1–3 tài nguyên miễn phí có thật (tên khoá học/tài liệu chính thức),
một mini-project để đưa vào CV, số tuần.
Trả JSON bằng tiếng Việt:
{{"summary": "...", "steps": [{{"skill": "...", "why": "...", "topics": ["..."], "resources": ["..."],
"project": "...", "weeks": 2}}]}}"""


def roadmap(have, missing_counts, targets, provider=None):
    miss = ", ".join(f"{s} ({n})" for s, n in missing_counts[:12])
    p = ROADMAP_PROMPT.format(have=", ".join(sorted(have)) or "(không rõ)", missing=miss,
                              targets="; ".join(targets[:10]))
    return ask_json(provider or llm.role("pipeline"), p)


# ---------------------------------------------------------------- nhận xét CV (1 lời gọi duy nhất trong demo)
REVIEW_PROMPT = """Bạn là chuyên viên tuyển dụng IT kiêm mentor, giúp sinh viên Việt Nam ứng tuyển Intern/Fresher/Junior
mảng AI/ML và Data Engineering. Hãy nhận xét CV dưới đây DỰA TRÊN các tin tuyển dụng mà hệ thống đã gợi ý.

Kỹ năng hệ thống nhận diện được từ CV (bằng từ điển): {have}
Top tin được gợi ý (tên tin — kỹ năng đã khớp | còn thiếu):
{jobs}
Kỹ năng còn thiếu, kèm số tin trong top yêu cầu: {missing}

Yêu cầu:
- Nhận xét cụ thể, dựa vào nội dung CV (trích dẫn ngắn khi cần), không chung chung.
- "cv_fixes": lỗi trình bày / nội dung trong CV và cách sửa (vd. dự án thiếu số liệu kết quả, kỹ năng chỉ liệt kê
  mà không có dự án chứng minh, thiếu link GitHub...).
- "extra_skills": kỹ năng kỹ thuật CV CÓ thể hiện rõ nhưng KHÔNG nằm trong danh sách hệ thống nhận diện
  (dùng tên tiếng Anh phổ biến; để [] nếu không có). Không bịa.
- "roadmap": 3-5 bước học 8-12 tuần cho các kỹ năng còn thiếu quan trọng nhất; mỗi bước có tài nguyên miễn phí
  có thật và một mini-project để đưa vào CV.
Trả JSON bằng tiếng Việt:
{{"summary": "2-3 câu đánh giá tổng quan", "strengths": ["..."], "weaknesses": ["..."],
"cv_fixes": [{{"issue": "...", "fix": "..."}}], "extra_skills": ["..."], "fit": "CV hợp nhất với loại vị trí nào và vì sao",
"roadmap": [{{"skill": "...", "why": "...", "topics": ["..."], "resources": ["..."], "project": "...", "weeks": 2}}]}}

=== CV ===
{cv}"""


def review_cv(cv_text, have, recs, missing_counts, provider=None):
    """Nhận xét CV + kỹ năng từ điển bỏ sót + lộ trình học, tất cả trong 1 lời gọi."""
    jobs = "\n".join(f"- {r['job']['title']} ({r['job']['company']}) — {', '.join(r['matched'][:8]) or '∅'} | "
                     f"{', '.join(r['missing'][:8]) or '∅'}" for r in recs[:5])
    p = REVIEW_PROMPT.format(have=", ".join(sorted(have)) or "(không có)", jobs=jobs,
                             missing=", ".join(f"{s} ({n})" for s, n in missing_counts[:12]) or "(không có)",
                             cv=_cut(cv_text, 8000))
    return ask_json(provider or llm.role("pipeline"), p)


# ---------------------------------------------------------------- giám khảo
JUDGE_PROMPT ="""You are a technical recruiter in Vietnam evaluating whether a job posting fits a student's CV.
Target candidates are Intern / Fresher / Junior level.

RUBRIC
2 = Good fit: same field (AI/ML or Data), level is Intern/Fresher/Junior (or <=1 year required),
    and the CV covers most CORE skills in the requirements.
1 = Partial fit: same field, but missing several core skills, OR requires 1-2 years experience.
0 = No fit: different field, OR Senior/Lead/Manager, OR requires >=3 years experience.

Think step by step by filling the fields in order. Output JSON only:
{{
  "required_years": <number or null>,
  "level_ok": <true|false>,
  "domain_ok": <true|false>,
  "core_skills": ["core skills from the job requirements"],
  "matched_skills": ["core skills evidenced in the CV"],
  "missing_skills": ["core skills not evidenced in the CV"],
  "label": <0|1|2>,
  "reason": "<one sentence>"
}}
{extra}
=== CV ===
{cv}

=== JOB ===
Title: {title}
Company: {company}
Skill tags: {skills}
Requirements:
{requirements}
Description:
{description}"""


def judge_pair(cv_text, job, provider, variant="", other_opinion=None):
    extra = ""
    if other_opinion:
        extra = (f"\nAnother reviewer labeled this pair {other_opinion['label']} because: "
                 f"\"{other_opinion.get('reason', '')}\". Re-check carefully; keep your label if you still disagree.\n")
    p = JUDGE_PROMPT.format(extra=extra, cv=_cut(cv_text, 6000), title=job["title"], company=job["company"],
                            skills=", ".join(job.get("skills", [])),
                            requirements=_cut(job.get("requirements"), 4000),
                            description=_cut(job.get("description"), 2500))
    d = ask_json(provider, p, variant=variant)
    try:
        d["label"] = max(0, min(2, int(d.get("label", 0))))
    except (TypeError, ValueError):
        d["label"] = 0
    return d


ROADMAP_JUDGE_PROMPT = """Evaluate a learning roadmap generated for an IT student (target: Intern/Fresher/Junior).
Student skills: {have}
Skills the student is missing (with demand counts): {missing}
ROADMAP (JSON): {roadmap}

Score each criterion 1-5 (5 = excellent):
- relevance: covers the most-demanded missing skills, nothing irrelevant
- specificity: concrete topics and real, identifiable resources
- feasibility: realistic for a student in the stated weeks
- ordering: logical prerequisite order, builds on existing skills
Return JSON: {{"relevance": n, "specificity": n, "feasibility": n, "ordering": n, "comment": "..."}}"""


def judge_roadmap(have, missing_counts, rm, provider):
    p = ROADMAP_JUDGE_PROMPT.format(have=", ".join(sorted(have)),
                                    missing=", ".join(f"{s} ({n})" for s, n in missing_counts[:12]),
                                    roadmap=json.dumps(rm, ensure_ascii=False))
    return ask_json(provider, p)


# ---------------------------------------------------------------- CV tổng hợp
SYNTH_CV_PROMPT = """Write a realistic CV (plain text, 350-600 words) for a Vietnamese IT student.
Persona: {persona}
The CV must demonstrate EXACTLY these skills and NO other technical skills/tools/libraries/platforms:
{skills}
Rules:
- Mix Vietnamese and English naturally like real Vietnamese student CVs (section headers may be either language).
- Use the given surface form for each skill (e.g. write "PySpark" if told so), not the canonical name.
- Put at least half of the skills inside project/experience descriptions instead of only a skills list.
- Use placeholders [NAME], [SCHOOL], [CONTACT] for personal info.
- Do not mention any programming language, library, database, cloud or tool not in the list
  (not even Git, Excel, Jupyter, Linux, English certificates) unless it is in the list.
Return JSON: {{"cv_text": "..."}}"""


def synth_cv(persona, surface_skills, provider, variant=""):
    p = SYNTH_CV_PROMPT.format(persona=persona, skills="\n".join(f"- {s}" for s in surface_skills))
    return ask_json(provider, p, temperature=0.9, variant=variant)["cv_text"]
