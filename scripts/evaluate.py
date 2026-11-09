"""Đánh giá toàn bộ hệ thống bằng LLM-as-a-judge → results/report.md, results/metrics.json, results/best_weights.json

    python scripts/evaluate.py                      # tự chọn CV thật (data/cvs) nếu có ≥10, không thì CV tổng hợp
    python scripts/evaluate.py --cvs synthetic --max-cvs 30
    python scripts/evaluate.py --skip-roadmap --skip-extraction

Mọi lời gọi LLM được cache (data/llm_cache.sqlite) nên chạy lại chỉ tốn tiền cho phần mới.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobmatch import config, llm, llm_tasks  # noqa: E402
from jobmatch.matcher import Matcher, embed  # noqa: E402
from jobmatch.skills import canonical_set  # noqa: E402

K = 5
EQUAL = (1 / 3, 1 / 3, 1 / 3)
# lưới trọng số w1+w2+w3 = 1, bước 0.1 (66 tổ hợp); sắp theo độ gần trọng số đều để khi hoà thì chọn bộ cân bằng
GRID = sorted({(a / 10, b / 10, round((10 - a - b) / 10, 1)) for a in range(11) for b in range(11 - a)},
              key=lambda w: sum((x - 1 / 3) ** 2 for x in w))

SYSTEMS = [  # tên, kiểu, trọng số / điều kiện lưới, có lọc cấp bậc
    ("Ngẫu nhiên (sau lọc)", "random", None, True),
    ("Chỉ TF-IDF", "fixed", (1.0, 0.0, 0.0), True),
    ("Chỉ embedding", "fixed", (0.0, 1.0, 0.0), True),
    ("Chỉ kỹ năng", "fixed", (0.0, 0.0, 1.0), True),
    ("TF-IDF + embedding", "cv", lambda w: w[2] == 0, True),
    ("TF-IDF + kỹ năng", "cv", lambda w: w[1] == 0, True),
    ("Embedding + kỹ năng", "cv", lambda w: w[0] == 0, True),
    ("Hybrid, trọng số đều", "fixed", EQUAL, True),
    ("Hybrid (chọn trọng số bằng CV)", "cv", lambda w: True, True),
    ("Hybrid (CV), không lọc cấp bậc", "cv", lambda w: True, False),
]
HYBRID = "Hybrid (chọn trọng số bằng CV)"
SINGLES = ["Chỉ TF-IDF", "Chỉ embedding", "Chỉ kỹ năng"]


# ------------------------------------------------------------------ độ đo
def gains(r):
    return 2 ** r - 1


def ndcg(rels, ideal, k=K):
    dcg = sum(gains(r) / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(gains(r) / np.log2(i + 2) for i, r in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def precision(rels, k=K):
    return sum(r >= 1 for r in rels[:k]) / k


def mean_ci(vals, n_boot=1000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0
    idx = np.random.default_rng(seed).integers(0, len(vals), (n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ------------------------------------------------------------------ dữ liệu
def load_cvs(source, max_cvs):
    real = sorted(config.CV_DIR.glob("*.txt")) if config.CV_DIR.exists() else []
    synth = json.loads(config.SYNTH_CVS.read_text(encoding="utf-8")) if config.SYNTH_CVS.exists() else []
    if source == "auto":
        source = "real" if len(real) >= 10 else "synthetic"
    if source == "real":
        cvs = [{"id": p.stem, "text": p.read_text(encoding="utf-8")} for p in real]
    else:
        cvs = [{"id": s["id"], "text": s["text"]} for s in synth]
    if not cvs:
        sys.exit("Không có CV nào. Chạy scripts/prepare_cvs.py (CV thật) hoặc scripts/make_synthetic_cvs.py trước.")
    return source, cvs[:max_cvs]


class Evaluator:
    def __init__(self, m, cvs, cv_skill_mode="dict"):
        self.m, self.cvs = m, cvs
        self.text = {c["id"]: c["text"] for c in cvs}
        self.job = {j["url"]: j for j in m.jobs}
        embs = embed([c["text"] for c in cvs])
        vocab = m.extractor.vocabulary()
        llm_skills = [None] * len(cvs)
        if cv_skill_mode == "llm":  # CV nào LLM lỗi → None → matcher tự dùng từ điển
            llm_skills = llm.map_parallel(lambda c: canonical_set(llm_tasks.extract_cv_skills(c["text"], vocab)),
                                          cvs, desc="CV skills (LLM)")
        self.raw, self.cv_skills = {}, {}
        for c, e, skills in zip(cvs, embs, llm_skills):
            self.raw[c["id"]], self.cv_skills[c["id"]] = m.raw_signals(c["text"], skills, e)
        self._rank = {}

    def top(self, cid, weights, use_filter, k=10):
        key = (cid, weights, use_filter)
        if key not in self._rank:
            if weights == "random":
                idx = [i for i in range(len(self.m.jobs)) if self.m.eligible[i]]
                random.Random(f"rand-{cid}").shuffle(idx)
            else:
                idx = self.m.rank(self.raw[cid], weights, use_filter, k=10)
            self._rank[key] = [self.m.jobs[i]["url"] for i in idx[:10]]
        return self._rank[key][:k]

    def pool(self, cid):
        p = set(self.top(cid, "random", True)) | set(self.top(cid, EQUAL, False))
        for name, kind, w, filt in SYSTEMS:
            if kind == "fixed":
                p |= set(self.top(cid, w, filt))
        for w in GRID:
            p |= set(self.top(cid, w, True, K)) | set(self.top(cid, w, False, K))
        return {(cid, u) for u in p}

    # ----- chạy toàn bộ các hệ thống với một bộ nhãn -----
    def run(self, labels, level_bad):
        ids = [c["id"] for c in self.cvs]
        ideal = {cid: [v for (c, _), v in labels.items() if c == cid] for cid in ids}
        valid = [cid for cid in ids if any(v >= 1 for v in ideal[cid])]

        def rels(cid, urls):
            return [labels.get((cid, u), 0) for u in urls]

        def mean_ndcg(cids, w, filt):
            return np.mean([ndcg(rels(c, self.top(c, w, filt, K)), ideal[c]) for c in cids]) if cids else 0

        folds = list(KFold(max(2, min(5, len(ids))), shuffle=True, random_state=42).split(ids))
        out = {}
        for name, kind, spec, filt in SYSTEMS:
            chosen, weights_of = [], {}
            if kind == "random":
                weights_of = {c: "random" for c in ids}
            elif kind == "fixed":
                weights_of = {c: spec for c in ids}
            else:
                grid = [w for w in GRID if spec(w)]
                for tr, te in folds:
                    tr_ids = [ids[i] for i in tr]
                    best = max(grid, key=lambda w: mean_ndcg(tr_ids, w, filt))
                    chosen.append(best)
                    weights_of.update({ids[i]: best for i in te})
            per_cv = {}
            for c in ids:
                urls = self.top(c, weights_of[c], filt)
                judged_lv = [level_bad[(c, u)] for u in urls if (c, u) in level_bad]
                per_cv[c] = {"top10": urls, "p5": precision(rels(c, urls)), "ndcg5": ndcg(rels(c, urls), ideal[c]),
                             "senior10": float(np.mean(judged_lv)) if judged_lv else 0.0}
            out[name] = {"per_cv": per_cv, "fold_weights": chosen, "filter": filt}
        return out, valid

    def best_weights_all(self, labels, use_filter=True):
        ids = [c["id"] for c in self.cvs]
        ideal = {cid: [v for (c, _), v in labels.items() if c == cid] for cid in ids}

        def score(w):
            return np.mean([ndcg([labels.get((c, u), 0) for u in self.top(c, w, use_filter, K)], ideal[c])
                            for c in ids])
        best = max(GRID, key=score)
        return best, float(score(best))


# ------------------------------------------------------------------ giám khảo
def run_judges(ev, pairs, providers, store):
    for p in providers:
        todo = sorted(x for x in pairs if x not in store[p])
        if not todo:
            continue
        print(f"Giám khảo {p} ({llm.model_name(p)}): chấm {len(todo)} cặp")
        res = llm.map_parallel(lambda x: llm_tasks.judge_pair(ev.text[x[0]], ev.job[x[1]], p), todo,
                               desc=f"judge {p}")
        store[p].update({x: r for x, r in zip(todo, res) if r is not None})


def reconcile(ev, store, main, second):
    """Cặp lệch 2 bậc (0 vs 2) → giám khảo chính chấm lại khi được xem lý do của giám khảo kia."""
    final = {x: r["label"] for x, r in store[main].items()}
    if not second:
        return final, 0
    todo = [x for x, r in store[main].items() if x in store[second] and abs(r["label"] - store[second][x]["label"]) == 2]
    res = llm.map_parallel(lambda x: llm_tasks.judge_pair(ev.text[x[0]], ev.job[x[1]], main,
                                                          other_opinion=store[second][x]), todo, desc="re-judge")
    for x, r in zip(todo, res):
        if r is not None:
            final[x] = r["label"]
    return final, len(todo)


# ------------------------------------------------------------------ báo cáo
def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def summarize(res, valid):
    rows, summary = [], {}
    for name, _, _, _ in SYSTEMS:
        pc = res[name]["per_cv"]
        p5 = mean_ci([pc[c]["p5"] for c in valid])
        nd = mean_ci([pc[c]["ndcg5"] for c in valid])
        sr = float(np.mean([pc[c]["senior10"] for c in pc]))
        summary[name] = {"P@5": p5, "NDCG@5": nd, "senior@10": sr, "fold_weights": res[name]["fold_weights"]}
        rows.append([name, f"{p5[0]:.3f} [{p5[1]:.3f}–{p5[2]:.3f}]", f"{nd[0]:.3f} [{nd[1]:.3f}–{nd[2]:.3f}]",
                     f"{sr:.0%}"])
    return summary, table(["Hệ thống", "P@5 [95% CI]", "NDCG@5 [95% CI]", "Tin sai cấp bậc trong top-10"], rows)


def wtest(res, valid, a, b):
    x = [res[a]["per_cv"][c]["ndcg5"] for c in valid]
    y = [res[b]["per_cv"][c]["ndcg5"] for c in valid]
    try:
        p = float(wilcoxon(x, y).pvalue)
    except ValueError:  # mọi chênh lệch = 0
        p = 1.0
    return float(np.mean(x) - np.mean(y)), p


def fmt_w(w):
    return "(" + ", ".join(f"{x:.1f}" for x in w) + ")"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvs", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--max-cvs", type=int, default=30)
    ap.add_argument("--cv-skills", choices=["auto", "dict", "llm"], default="auto",
                    help="cách trích kỹ năng CV khi xếp hạng (llm: LLM pipeline — mặc định khi có key; dict: từ điển)")
    ap.add_argument("--stability-n", type=int, default=100)
    ap.add_argument("--roadmap-n", type=int, default=10)
    ap.add_argument("--skip-roadmap", action="store_true")
    ap.add_argument("--skip-extraction", action="store_true")
    args = ap.parse_args()

    # Vai trò được kiểm tra bằng 1 lời gọi thật; OpenAI không dùng được → tự chuyển sang Gemini
    main_j, second_j, pipe = llm.role("judge_main"), llm.role("judge_second"), llm.role("pipeline")
    if not main_j:
        sys.exit("Không có LLM nào dùng được làm giám khảo. Kiểm tra API key / credit trong .env")
    if not second_j or second_j == main_j:
        print("! Không có giám khảo thứ hai khác giám khảo chính → bỏ qua kappa và kiểm tra độ nhạy")
        second_j = None
    providers = [p for p in (main_j, second_j) if p]
    print(f"Giám khảo chính: {main_j} ({llm.model_name(main_j)}) | giám khảo 2: "
          f"{second_j and llm.model_name(second_j)} | pipeline: {pipe and llm.model_name(pipe)}")

    if args.cv_skills == "auto":
        args.cv_skills = "llm" if pipe else "dict"
    config.RESULTS.mkdir(exist_ok=True)
    m = Matcher.load()
    m.ensure_embeddings()
    source, cvs = load_cvs(args.cvs, args.max_cvs)
    print(f"Bộ test: {len(cvs)} CV ({source}), {len(m.jobs)} tin, {int(m.eligible.sum())} tin sau lọc cấp bậc")
    ev = Evaluator(m, cvs, args.cv_skills)
    store = {p: {} for p in providers}

    # 1) pooling + chấm
    pairs = set().union(*(ev.pool(c["id"]) for c in cvs))
    print(f"Pool: {len(pairs)} cặp CV–tin (~{len(pairs) / len(cvs):.0f}/CV)")
    run_judges(ev, pairs, providers, store)
    labels, n_rejudge = reconcile(ev, store, main_j, second_j)

    # 2) chạy hệ thống; chấm bù các tin lọt top-10 mà chưa có nhãn, lặp tới khi đủ
    for it in range(4):
        level_bad = {x: (r.get("level_ok") is False) for x, r in store[main_j].items()}
        res, valid = ev.run(labels, level_bad)
        holes = {(c, u) for s in res.values() for c, v in s["per_cv"].items() for u in v["top10"]} - set(labels)
        if not holes or it == 3:
            break
        print(f"Chấm bù {len(holes)} cặp mới lọt vào top-10")
        run_judges(ev, holes, providers, store)
        labels, n_rejudge = reconcile(ev, store, main_j, second_j)

    summary, main_table = summarize(res, valid)
    best_w, best_score = ev.best_weights_all(labels)
    config.BEST_WEIGHTS_FILE.write_text(json.dumps({"weights": best_w, "ndcg5_on_all_cvs": best_score,
                                                    "note": "Chọn bằng grid search trên toàn bộ bộ test — dùng cho demo"},
                                                   indent=2), encoding="utf-8")
    best_single = max(SINGLES, key=lambda s: summary[s]["NDCG@5"][0])
    tests = {f"{HYBRID} vs {b}": wtest(res, valid, HYBRID, b)
             for b in [best_single, "Hybrid, trọng số đều", "Ngẫu nhiên (sau lọc)"]}

    # 3) độ tin cậy nhãn
    rel = {"pairs_judged": len(labels), "rejudged_2_level_disagreements": n_rejudge}
    label_dist = Counter(labels.values())
    if second_j:
        common = sorted(set(store[main_j]) & set(store[second_j]))
        a = [store[main_j][x]["label"] for x in common]
        b = [store[second_j][x]["label"] for x in common]
        rel.update(kappa_linear=float(cohen_kappa_score(a, b, weights="linear")),
                   raw_agreement=float(np.mean(np.array(a) == np.array(b))),
                   confusion=confusion_matrix(a, b, labels=[0, 1, 2]).tolist())
    sample = random.Random(0).sample(sorted(store[main_j]), min(args.stability_n, len(store[main_j])))
    reruns = [llm.map_parallel(lambda x, v=v: llm_tasks.judge_pair(ev.text[x[0]], ev.job[x[1]], main_j, variant=v),
                               sample, desc=f"stability {v}") for v in ("run2", "run3")]
    same = [store[main_j][x]["label"] == r2["label"] == r3["label"]
            for x, r2, r3 in zip(sample, *reruns) if r2 and r3]
    rel["stability_same_label_3_runs"] = float(np.mean(same)) if same else None

    sens_table, sens_summary = "", None
    if second_j:
        labels_b = {x: r["label"] for x, r in store[second_j].items()}
        res_b, valid_b = ev.run(labels_b, {x: (r.get("level_ok") is False) for x, r in store[second_j].items()})
        sens_summary, sens_table = summarize(res_b, valid_b)

    # 4) kỹ năng còn thiếu: so với giám khảo trên top-5 của hybrid
    tp = fp = fn = mtp = mfp = 0
    for c, v in res[HYBRID]["per_cv"].items():
        for u in v["top10"][:K]:
            j = store[main_j].get((c, u))
            if not j:
                continue
            idx = next(i for i, jj in enumerate(m.jobs) if jj["url"] == u)
            matched, missing = m.skill_gap(idx, ev.cv_skills[c])
            jm, jmiss = canonical_set(j.get("matched_skills", [])), canonical_set(j.get("missing_skills", []))
            s_miss = set(missing)
            tp, fp, fn = tp + len(s_miss & jmiss), fp + len(s_miss - jmiss), fn + len(jmiss - s_miss)
            mtp, mfp = mtp + len(set(matched) & jm), mfp + len(set(matched) - jm)
    gap = {"missing_vs_judge_PRF": prf(tp, fp, fn), "matched_precision_vs_judge": mtp / (mtp + mfp) if mtp + mfp else None}

    # 5) trích kỹ năng
    extraction = run_extraction_eval(m) if not args.skip_extraction else None

    # 6) lộ trình học
    roadmap_eval = None
    if not args.skip_roadmap and pipe:
        roadmap_eval = run_roadmap_eval(m, cvs[:args.roadmap_n], best_w, main_j, args.cv_skills)

    # ---- ghi kết quả ----
    with open(config.RESULTS / "per_cv.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["system", "cv", "P@5", "NDCG@5", "senior@10", "top5"])
        for name, s in res.items():
            for c, v in s["per_cv"].items():
                w.writerow([name, c, v["p5"], round(v["ndcg5"], 4), v["senior10"], " ".join(v["top10"][:5])])
    with open(config.RESULTS / "judgments.jsonl", "w", encoding="utf-8") as f:
        for p in providers:
            for (c, u), r in store[p].items():
                f.write(json.dumps({"judge": p, "model": llm.model_name(p), "cv": c, "url": u, **r},
                                   ensure_ascii=False) + "\n")
    metrics = {"date": str(date.today()), "cv_source": source, "n_cvs": len(cvs), "n_valid_cvs": len(valid),
               "n_jobs": len(m.jobs), "n_eligible": int(m.eligible.sum()), "cv_skill_mode": args.cv_skills,
               "judges": {"main": [main_j, llm.model_name(main_j)],
                          "second": [second_j, llm.model_name(second_j)] if second_j else None},
               "pipeline_llm": [pipe, llm.model_name(pipe)] if pipe else None,
               "same_provider": len({llm.split_spec(p)[0] for p in (main_j, second_j, pipe) if p}) == 1,
               "label_distribution": dict(label_dist), "systems": summary, "wilcoxon": tests,
               "reliability": rel, "sensitivity_judge2": sens_summary, "best_weights": best_w,
               "skill_gap": gap, "extraction": extraction, "roadmap": roadmap_eval}
    (config.RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metrics, main_table, sens_table, best_single)
    print(f"\n{main_table}\nĐã ghi results/report.md, metrics.json, per_cv.csv, judgments.jsonl, best_weights.json")


def run_extraction_eval(m):
    out = {}
    vocab = m.extractor.vocabulary()
    pl = llm.role("pipeline")
    if config.SYNTH_CVS.exists():
        synth = json.loads(config.SYNTH_CVS.read_text(encoding="utf-8"))
        llm_preds = (llm.map_parallel(lambda s: canonical_set(llm_tasks.extract_cv_skills(s["text"], vocab)), synth,
                                      desc="CV skills (LLM)") if pl else [None] * len(synth))
        cnt = {"dict": [0, 0, 0], "llm": [0, 0, 0], "llm+dict": [0, 0, 0]}
        for s, lp in zip(synth, llm_preds):
            gold = set(s["gold_skills"])
            dp = m.extractor.extract(s["text"])
            for name, pred in (("dict", dp), ("llm", lp), ("llm+dict", lp | dp if lp is not None else None)):
                if pred is None:
                    continue
                cnt[name][0] += len(pred & gold)
                cnt[name][1] += len(pred - gold)
                cnt[name][2] += len(gold - pred)
        out["cv"] = {"n": len(synth), **{k: prf(*v) for k, v in cnt.items() if sum(v)}}
    llm_file = config.DATA / "jd_skills_llm.json"
    if llm_file.exists():
        detail = json.loads(llm_file.read_text(encoding="utf-8"))
        rows = {"dict": [0, 0, 0, 0], "llm": [0, 0, 0, 0]}  # hit tag, tổng tag, số kỹ năng mới, grounded
        n_pred = {"dict": 0, "llm": 0}
        n = 0
        for j in m.jobs:
            if j["url"] not in detail:
                continue
            n += 1
            tags = set(j["skills_tag"])
            body = j.get("requirements", "") + "\n" + j.get("description", "")
            d = detail[j["url"]]
            for name, pred in (("dict", set(j["skills_dict"])),
                               ("llm", canonical_set(d["core"] + d["nice_to_have"]))):
                rows[name][0] += len(pred & tags)
                rows[name][1] += len(tags)
                rows[name][2] += len(pred - tags)
                rows[name][3] += sum(m.extractor.mentioned(s, body) for s in pred)
                n_pred[name] += len(pred)
        out["jd"] = {"n": n, **{k: {"recall_vs_tags": v[0] / v[1] if v[1] else 0,
                                    "new_skills_per_job": v[2] / n if n else 0,
                                    "grounding_rate": v[3] / n_pred[k] if n_pred[k] else 0}
                                for k, v in rows.items()}}
    return out


def run_roadmap_eval(m, cvs, weights, judge, cv_skill_mode):
    vocab = m.extractor.vocabulary()

    def one(c):
        skills = canonical_set(llm_tasks.extract_cv_skills(c["text"], vocab)) if cv_skill_mode == "llm" else None
        recs, have = m.recommend(c["text"], weights, cv_skills=skills)
        miss = Counter(s for r in recs for s in r["missing"]).most_common()
        rm = llm_tasks.roadmap(have, miss, [r["job"]["title"] for r in recs])
        return llm_tasks.judge_roadmap(have, miss, rm, judge)

    scores = [s for s in llm.map_parallel(one, cvs, desc="roadmap") if s]
    crit = ["relevance", "specificity", "feasibility", "ordering"]
    return {"n": len(scores), **{k: float(np.mean([float(s.get(k, 0)) for s in scores])) for k in crit}}


def write_report(mt, main_table, sens_table, best_single):
    r = mt["reliability"]
    L = [f"# Kết quả đánh giá — {mt['date']}\n",
         f"- Bộ test: **{mt['n_cvs']} CV ({'thật, đã ẩn danh' if mt['cv_source'] == 'real' else 'tổng hợp'})**, "
         f"{mt['n_valid_cvs']} CV có ít nhất một tin phù hợp được dùng để tính trung bình.",
         f"- Kho tin: {mt['n_jobs']} tin, còn {mt['n_eligible']} tin sau lọc cấp bậc.",
         f"- Giám khảo chính: `{mt['judges']['main'][1]}`"
         + (f"; giám khảo thứ hai: `{mt['judges']['second'][1]}`" if mt["judges"]["second"] else ""),
         f"- LLM trong pipeline: `{mt['pipeline_llm'][1] if mt['pipeline_llm'] else 'không có'}`. "
         f"Kỹ năng CV khi xếp hạng: {mt['cv_skill_mode']}.",
         *(["- ⚠️ Pipeline và cả hai giám khảo cùng một nhà cung cấp (OpenAI không dùng được nên chuyển hết sang "
            "Gemini). Hai giám khảo là hai model khác nhau, nhưng vẫn có thể thiên vị theo cùng một kiểu; "
            "nên chạy lại với giám khảo OpenAI khi có credit."] if mt.get("same_provider") else []),
         f"- Phân bố nhãn (0/1/2): {mt['label_distribution']}\n",
         "## 1. Chất lượng gợi ý\n",
         "Trọng số của các hệ thống kết hợp được chọn bằng 5-fold cross-validation theo CV "
         "(grid search, bước 0.1). Số trong ngoặc là khoảng tin cậy 95% (bootstrap theo CV).\n",
         main_table,
         f"\nTrọng số chọn được ở từng fold (Hybrid): "
         f"{', '.join(fmt_w(w) for w in mt['systems'][HYBRID]['fold_weights'])}. "
         f"Trọng số dùng cho demo (chọn trên toàn bộ CV): **{fmt_w(mt['best_weights'])}** (TF-IDF, embedding, kỹ năng).\n",
         "### Kiểm định Wilcoxon (NDCG@5, ghép cặp theo CV)\n",
         table(["So sánh", "Chênh lệch NDCG@5", "p-value"],
               [[k, f"{d:+.3f}", f"{p:.4f}" + (" ✱" if p < 0.05 else "")] for k, (d, p) in mt["wilcoxon"].items()]),
         f"\n✱ = có ý nghĩa thống kê ở mức 0.05. Tín hiệu đơn tốt nhất: {best_single}.\n",
         "## 2. Độ tin cậy của nhãn LLM\n",
         f"- Số cặp được chấm: {r['pairs_judged']}; số cặp hai giám khảo lệch 2 bậc được chấm lại: "
         f"{r['rejudged_2_level_disagreements']}."]
    if "kappa_linear" in r:
        L += [f"- Cohen's kappa có trọng số (linear) giữa hai giám khảo: **{r['kappa_linear']:.3f}**; "
              f"tỷ lệ trùng nhãn: {r['raw_agreement']:.1%}.",
              "- Ma trận nhầm lẫn (hàng: giám khảo chính, cột: giám khảo thứ hai, nhãn 0/1/2): "
              f"`{r['confusion']}`"]
    if r.get("stability_same_label_3_runs") is not None:
        L.append(f"- Độ ổn định: chấm lại 3 lần thì {r['stability_same_label_3_runs']:.1%} số cặp giữ nguyên nhãn.")
    if sens_table:
        L += ["\n### Kiểm tra độ nhạy: tính lại bằng nhãn của giám khảo thứ hai\n", sens_table]
    g = mt["skill_gap"]
    p, rc, f1 = g["missing_vs_judge_PRF"]
    L += ["\n## 3. Kỹ năng đã khớp / còn thiếu (top-5 của Hybrid, so với giám khảo chính)\n",
          f"- Kỹ năng còn thiếu: precision {p:.3f}, recall {rc:.3f}, F1 {f1:.3f}.",
          f"- Kỹ năng đã khớp: precision {g['matched_precision_vs_judge'] or 0:.3f}.",
          "- Ghi chú: hệ thống tính cả kỹ năng \"nice to have\", còn giám khảo chỉ liệt kê kỹ năng cốt lõi, "
          "nên precision của phần còn thiếu thường thấp hơn recall."]
    ex = mt.get("extraction") or {}
    if "cv" in ex:
        L += [f"\n## 4. Trích kỹ năng từ CV ({ex['cv']['n']} CV tổng hợp có đáp án)\n",
              table(["Cách trích", "Precision", "Recall", "F1"],
                    [[k, *(f"{x:.3f}" for x in ex["cv"][k])] for k in ("dict", "llm", "llm+dict")
                     if k in ex["cv"]])]
    if "jd" in ex:
        L += [f"\n## 5. Trích kỹ năng từ JD ({ex['jd']['n']} tin, so với tag ITviec)\n",
              table(["Cách trích", "Recall so với tag", "Kỹ năng mới/tin", "Grounding rate"],
                    [[k, f"{ex['jd'][k]['recall_vs_tags']:.3f}", f"{ex['jd'][k]['new_skills_per_job']:.1f}",
                      f"{ex['jd'][k]['grounding_rate']:.1%}"] for k in ("dict", "llm")]),
              "\nGrounding rate là tỷ lệ kỹ năng trích ra mà tên hoặc alias thật sự xuất hiện trong JD "
              "(từ điển luôn đạt 100% do cách làm)."]
    if mt.get("roadmap"):
        rm = mt["roadmap"]
        L += [f"\n## 6. Lộ trình học (LLM chấm thang 1–5, {rm['n']} CV)\n",
              table(["Liên quan", "Cụ thể", "Khả thi", "Thứ tự"],
                    [[f"{rm[k]:.2f}" for k in ("relevance", "specificity", "feasibility", "ordering")]])]
    L += ["\n## Hạn chế\n",
          "- Nhãn do LLM gán chỉ là silver label (gần đúng), không phải đánh giá của chuyên gia tuyển dụng. "
          "Dùng hai giám khảo từ hai nhà cung cấp và kiểm tra độ nhạy để giảm rủi ro.",
          "- Pooling: tin không lọt vào pool được coi là không phù hợp; NDCG chỉ tính trên các tin đã chấm.",
          "- CV tổng hợp có thể vô tình nhắc thêm kỹ năng ngoài đáp án, khi đó bị tính là false positive."]
    (config.RESULTS / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
