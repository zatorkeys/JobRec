"""Sinh CV tổng hợp có đáp án kỹ năng (gold) → data/synthetic_cvs.json

Dùng cho (1) đo precision/recall bước trích kỹ năng từ CV, (2) bộ test xếp hạng khi chưa đủ CV thật.
Một nửa CV do OpenAI viết, một nửa do Gemini viết để không thiên về một model.

    python scripts/make_synthetic_cvs.py --n 40
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobmatch import config, llm, llm_tasks  # noqa: E402
from jobmatch.skills import ALIASES, CASE_SENSITIVE, NO_FREE_TEXT  # noqa: E402

TRACKS = {
    "AI/Machine Learning": {
        "core": ["Python", "Machine Learning", "Deep Learning"],
        "pool": ["PyTorch", "TensorFlow", "Scikit-learn", "NLP", "Computer Vision", "LLM", "RAG", "Transformers",
                 "OpenCV", "Pandas", "NumPy", "SQL", "Docker", "Git", "Statistics", "FastAPI", "LangChain",
                 "Fine-tuning", "YOLO", "XGBoost", "MLOps", "Linux", "Time Series", "Recommender Systems",
                 "Vector Database", "Data Visualization", "AWS", "English", "Algorithms"],
    },
    "Data Engineering": {
        "core": ["Python", "SQL", "ETL"],
        "pool": ["Apache Spark", "Apache Airflow", "Apache Kafka", "Data Warehouse", "PostgreSQL", "MySQL",
                 "Docker", "Git", "Linux", "AWS", "GCP", "Azure", "BigQuery", "dbt", "Hadoop", "Hive", "Pandas",
                 "Data Modeling", "MongoDB", "Power BI", "Databricks", "Snowflake", "Kubernetes", "Scala",
                 "Data Lake", "Java", "English", "Bash"],
    },
    "Data Analytics → Data Engineering": {
        "core": ["SQL", "Data Analysis"],
        "pool": ["Python", "Excel", "Power BI", "Tableau", "Pandas", "Statistics", "Data Visualization", "ETL",
                 "PostgreSQL", "MySQL", "BigQuery", "Apache Airflow", "Git", "English", "Machine Learning",
                 "Data Warehouse", "Looker"],
    },
}
YEARS = ["sinh viên năm 3", "sinh viên năm 4", "vừa tốt nghiệp (fresher)"]
LEVELS = {"yếu (ít dự án, chủ yếu môn học)": (3, 5), "trung bình (2-3 dự án cá nhân)": (6, 9),
          "khá (có 1 kỳ thực tập + dự án)": (10, 13)}


def surface(skill, rng):
    forms = [skill] + [a for a in ALIASES.get(skill, []) if a.lower() not in NO_FREE_TEXT
                       and a not in CASE_SENSITIVE and len(a) > 2]
    return rng.choice(forms)  # vd. "Apache Spark" có thể thành "pyspark" — kiểm tra khả năng chuẩn hoá alias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    providers = [p for p in ("openai", "gemini") if llm.healthy(p)]  # chỉ dùng provider gọi được thật
    if not providers:
        sys.exit("Không có LLM nào dùng được (kiểm tra API key / credit trong .env)")
    print("Sinh CV bằng:", ", ".join(f"{p} ({llm.model_name(p)})" for p in providers))

    rng = random.Random(args.seed)
    specs = []
    for i in range(args.n):
        track = list(TRACKS)[i % len(TRACKS)]
        level, (lo, hi) = rng.choice(list(LEVELS.items()))
        t = TRACKS[track]
        k = rng.randint(lo, hi)
        skills = list(dict.fromkeys(t["core"] + rng.sample(t["pool"], max(0, k - len(t["core"])))))
        persona = f"{rng.choice(YEARS)} ngành CNTT, định hướng {track}, mức độ {level}"
        specs.append({"id": f"syn_{i + 1:02d}", "track": track, "persona": persona,
                      "gold_skills": sorted(skills), "surface": [surface(s, rng) for s in skills],
                      "provider": providers[i % len(providers)]})

    texts = llm.map_parallel(lambda s: llm_tasks.synth_cv(s["persona"], s["surface"], s["provider"], s["id"]),
                             specs, desc="synthetic CV")
    out = [dict(s, text=t) for s, t in zip(specs, texts) if t]
    config.SYNTH_CVS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã lưu {len(out)} CV tổng hợp → {config.SYNTH_CVS}")


if __name__ == "__main__":
    main()
