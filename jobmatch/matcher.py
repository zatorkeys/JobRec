"""Ba tín hiệu so khớp CV–tin + xếp hạng hybrid.

    m = Matcher.load()                       # đọc data/jobs_processed.json + embeddings
    recs = m.recommend(cv_text, top_k=10)    # list dict: job, score, các điểm thành phần, matched, missing
"""
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config
from .jobs import load_processed
from .skills import SkillExtractor

SIGNALS = ("tfidf", "emb", "skill")
_emb_model = None


def _prep(text):
    t = text.lower()
    t = t.replace("c++", " cplusplus ").replace("c#", " csharp ").replace(".net", " dotnet ")
    return t


def get_embedder():
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        _emb_model.max_seq_length = 512
    return _emb_model


def _chunks(text, size=180, overlap=30):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else [""]
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, size - overlap)]


def embed(texts, batch_size=16):
    """Văn bản dài được cắt thành đoạn, lấy trung bình vector các đoạn (e5 dùng tiền tố 'query: ')."""
    model = get_embedder()
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for c in _chunks(t):
            pieces.append("query: " + c)
            owner.append(i)
    vecs = model.encode(pieces, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(pieces) > 64)
    owner = np.array(owner)
    out = np.stack([vecs[owner == i].mean(axis=0) for i in range(len(texts))])
    return normalize(out)


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Matcher:
    def __init__(self, jobs, job_emb=None, extractor=None):
        self.jobs = jobs
        self.extractor = extractor or SkillExtractor.from_jobs(jobs)
        self.tfidf = TfidfVectorizer(preprocessor=_prep, ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=1, max_df=0.85)
        self.job_tfidf = self.tfidf.fit_transform([j["text"] for j in jobs])
        self.job_emb = job_emb
        self.eligible = np.array([j["eligible"] for j in jobs])
        self.job_skill_sets = [set(j["job_skills"]) for j in jobs]
        self.tag_sets = [set(j["skills_tag"]) for j in jobs]

    @classmethod
    def load(cls):
        jobs = load_processed()
        emb = np.load(config.JOB_EMB) if config.JOB_EMB.exists() else None
        return cls(jobs, emb)

    def ensure_embeddings(self):
        if self.job_emb is None or len(self.job_emb) != len(self.jobs):
            self.job_emb = embed([j["text"] for j in self.jobs])
        return self.job_emb

    # ---------- tín hiệu thô ----------
    def raw_signals(self, cv_text, cv_skills=None, cv_emb=None):
        cv_skills = set(cv_skills) if cv_skills is not None else self.extractor.extract(cv_text)
        tfidf = (self.job_tfidf @ self.tfidf.transform([cv_text]).T).toarray().ravel()
        if cv_emb is None:
            cv_emb = embed([cv_text])[0]
        emb = self.ensure_embeddings() @ cv_emb
        # độ phủ yêu cầu: số kỹ năng khớp / (số kỹ năng tin yêu cầu + 1) — +1 để tin chỉ có 1 tag không được 100%
        skill = np.array([len(s & cv_skills) / (len(s) + 1) for s in self.job_skill_sets])
        return {"tfidf": tfidf, "emb": emb, "skill": skill}, cv_skills

    @staticmethod
    def combine(raw, weights):
        norm = {k: minmax(v) for k, v in raw.items()}
        return sum(w * norm[k] for k, w in zip(SIGNALS, weights)), norm

    def rank(self, raw, weights, use_filter=True, k=None):
        score, _ = self.combine(raw, weights)
        score = np.where(self.eligible, score, -np.inf) if use_filter else score
        order = np.argsort(-score, kind="stable")
        if use_filter:
            order = order[self.eligible[order]]
        return order[:k] if k else order

    def skill_gap(self, idx, cv_skills):
        js = self.job_skill_sets[idx]
        matched = sorted(js & cv_skills)
        tags = self.tag_sets[idx]
        missing = sorted(js - cv_skills, key=lambda s: (s not in tags, s))  # tag ITviec lên trước
        return matched, missing

    # ---------- API cho demo ----------
    def recommend(self, cv_text, weights=None, top_k=config.TOP_K, use_filter=True, cv_skills=None):
        weights = weights or load_weights()
        raw, cv_skills = self.raw_signals(cv_text, cv_skills)
        score, norm = self.combine(raw, weights)
        out = []
        for i in self.rank(raw, weights, use_filter, top_k):
            matched, missing = self.skill_gap(i, cv_skills)
            out.append({"job": self.jobs[i], "score": float(score[i]),
                        **{k: float(norm[k][i]) for k in SIGNALS},
                        "matched": matched, "missing": missing})
        return out, cv_skills


def load_weights():
    try:
        return tuple(json.loads(config.BEST_WEIGHTS_FILE.read_text(encoding="utf-8"))["weights"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return config.DEFAULT_WEIGHTS
