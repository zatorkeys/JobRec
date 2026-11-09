"""Demo web: streamlit run app.py"""
import hashlib
import html
from collections import Counter
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from jobmatch import config, llm, llm_tasks
from jobmatch.cv import anonymize, read_pdf
from jobmatch.matcher import Matcher, load_weights
from jobmatch.skills import canonical_set

st.set_page_config(page_title="Gợi ý việc làm IT", page_icon=":material/work:", layout="wide")


# ================================================================== dữ liệu & LLM
@st.cache_resource(show_spinner="Đang tải kho tin và mô hình ngôn ngữ...")
def get_matcher():
    m = Matcher.load()
    m.ensure_embeddings()
    from jobmatch.matcher import get_embedder
    get_embedder()
    return m


if not config.JOBS_PROCESSED.exists():
    st.error("Chưa có data/jobs_processed.json. Chạy `python itviec_crawler.py` (trong thư mục data) "
             "rồi `python scripts/build_index.py`.")
    st.stop()
m = get_matcher()

# LLM chỉ được gọi khi bấm "Nhận xét CV" (không kiểm tra lúc khởi động để khỏi tốn quota).
# Thử model cấu hình trước, lỗi (vd. OpenAI hết credit) thì thử model Gemini dự phòng.
LLM_CANDIDATES = [s for s in dict.fromkeys([config.PIPELINE_LLM, config.FALLBACK_ROLES["pipeline"]])
                  if llm.available(s)]
llm_ok = bool(LLM_CANDIDATES)


def review_with_fallback(*args):
    err = None
    for spec in LLM_CANDIDATES:
        try:
            return llm_tasks.review_cv(*args, provider=spec), spec
        except Exception as e:
            err = e
    raise err or RuntimeError("Không có API key nào")


def friendly_error(e):
    s = str(e)
    if "hết quota" in s or "PerDay" in s:
        return ("Gemini đã dùng hết lượt miễn phí hôm nay (20 lượt/ngày cho mỗi model). Hãy thử lại sau khi quota "
                "được làm mới, hoặc thêm GEMINI_API_KEY_2 (tạo ở project Google Cloud khác) vào file .env.")
    if "credit" in s or "insufficient_quota" in s:
        return "Tài khoản OpenAI đã hết credit và không có model Gemini nào gọi được."
    return f"Không gọi được AI: {s[:240]}"


ELIGIBLE = [j for j in m.jobs if j["eligible"]]
N_AI_JOBS = sum(1 for j in ELIGIBLE if j["skills_llm"] is not None)
try:
    UPDATED = datetime.fromtimestamp(config.JOBS_RAW.stat().st_mtime).strftime("%d/%m/%Y")
except OSError:
    UPDATED = "?"


# ================================================================== giao diện
def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = is_dark()
T = ({"bg": "#141413", "surface": "#1c1c1a", "border": "#2e2d2a", "text": "#f0efec", "muted": "#a3a29b",
      "accent": "#5598e7", "accent_soft": "#1b2a3f", "have_bg": "#17301f", "have_text": "#b4e6c0",
      "need_border": "#4a4945", "bar_bg": "#2a2a27", "ai": "#9d8df0"}
     if DARK else
     {"bg": "#fcfcfb", "surface": "#ffffff", "border": "#e7e5e0", "text": "#1a1a19", "muted": "#6b6a64",
      "accent": "#2a78d6", "accent_soft": "#eaf2fc", "have_bg": "#e8f5ec", "have_text": "#1d5a32",
      "need_border": "#d4d2cc", "bar_bg": "#efeee9", "ai": "#6250d6"})

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button,
.stApp input, .stApp textarea, .stApp [data-testid="stMarkdownContainer"] {{
  font-family: 'Be Vietnam Pro', system-ui, sans-serif; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.015em; }}
.hero {{ padding: 4px 0 18px; }}
.eyebrow {{ font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: {T['accent']}; margin-bottom: 6px; }}
.hero h1 {{ font-size: 2.05rem; font-weight: 700; margin: 0 0 6px; padding: 0; color: {T['text']}; }}
.hero p {{ color: {T['muted']}; font-size: 1rem; margin: 0; max-width: 720px; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 6px; }}
.stat {{ border: 1px solid {T['border']}; background: {T['surface']}; border-radius: 999px;
  padding: 4px 12px; font-size: .82rem; color: {T['muted']}; }}
.stat b {{ color: {T['text']}; font-weight: 600; }}
.section-title {{ font-size: 1.15rem; font-weight: 650; margin: 26px 0 4px; color: {T['text']}; }}
.section-sub {{ color: {T['muted']}; font-size: .87rem; margin-bottom: 12px; }}
.card {{ border: 1px solid {T['border']}; background: {T['surface']}; border-radius: 14px;
  padding: 18px 20px; margin-bottom: 12px; }}
.chip {{ display: inline-block; font-size: .8rem; line-height: 1; padding: 6px 10px; border-radius: 999px;
  margin: 0 6px 6px 0; background: {T['accent_soft']}; color: {T['text']}; }}
.chip.have {{ background: {T['have_bg']}; color: {T['have_text']}; }}
.chip.need {{ background: transparent; border: 1px dashed {T['need_border']}; color: {T['muted']}; }}
.chip sup {{ font-size: .6rem; font-weight: 700; color: {T['ai']}; margin-left: 3px; letter-spacing: .03em; }}
.job {{ border: 1px solid {T['border']}; background: {T['surface']}; border-radius: 14px;
  padding: 16px 20px 10px; margin-bottom: 12px; }}
.job-head {{ display: flex; gap: 14px; align-items: flex-start; }}
.rank {{ flex: none; width: 30px; height: 30px; border-radius: 8px; background: {T['accent_soft']};
  color: {T['accent']}; font-weight: 700; font-size: .9rem; display: flex; align-items: center;
  justify-content: center; margin-top: 2px; }}
.job-main {{ flex: 1; min-width: 0; }}
.job-title {{ font-weight: 650; font-size: 1.04rem; color: {T['text']} !important; text-decoration: none; }}
.job-title:hover {{ color: {T['accent']} !important; text-decoration: underline; }}
.job-meta {{ color: {T['muted']}; font-size: .85rem; margin: 3px 0 8px; }}
.pill {{ display: inline-block; font-size: .74rem; font-weight: 500; padding: 3px 8px; border-radius: 6px;
  border: 1px solid {T['border']}; color: {T['muted']}; margin: 0 6px 4px 0; }}
.pill.ok {{ border-color: transparent; background: {T['have_bg']}; color: {T['have_text']}; }}
.score {{ flex: none; text-align: right; min-width: 74px; }}
.score-num {{ font-size: 1.7rem; font-weight: 700; color: {T['text']}; line-height: 1; }}
.score-num span {{ font-size: .95rem; color: {T['muted']}; font-weight: 500; }}
.score-label {{ font-size: .74rem; color: {T['muted']}; margin-top: 3px; }}
.signals {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 12px 0 12px 44px; }}
.sig-label {{ font-size: .75rem; color: {T['muted']}; margin-bottom: 5px; display: flex;
  justify-content: space-between; }}
.sig-label b {{ color: {T['text']}; font-weight: 600; }}
.bar {{ height: 6px; border-radius: 3px; background: {T['bar_bg']}; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 3px; background: {T['accent']}; }}
.skills-row {{ display: flex; gap: 10px; margin-left: 44px; }}
.skills-label {{ flex: none; width: 70px; font-size: .78rem; color: {T['muted']}; padding-top: 5px; }}
.rank-list .row {{ display: grid; grid-template-columns: 150px 1fr 48px; gap: 12px; align-items: center;
  font-size: .87rem; padding: 5px 0; color: {T['text']}; }}
.rank-list .row span:last-child {{ color: {T['muted']}; text-align: right; }}
.review-summary {{ font-size: 1rem; line-height: 1.6; color: {T['text']}; margin-bottom: 14px; }}
.review-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.review-h {{ font-size: .78rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
  color: {T['muted']}; margin: 6px 0 6px; }}
.review-list {{ margin: 0; padding-left: 18px; color: {T['text']}; font-size: .92rem; line-height: 1.55; }}
.fix {{ border-left: 3px solid {T['accent']}; padding: 4px 0 4px 12px; margin: 8px 0; font-size: .92rem; }}
.fix b {{ color: {T['text']}; }}
.fix div {{ color: {T['muted']}; }}
.ai-tag {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .05em; color: {T['ai']};
  border: 1px solid {T['ai']}; border-radius: 5px; padding: 1px 6px; margin-left: 8px; vertical-align: middle; }}
.muted {{ color: {T['muted']}; font-size: .85rem; }}
@media (max-width: 760px) {{
  .signals {{ grid-template-columns: 1fr; margin-left: 0; }} .skills-row {{ margin-left: 0; flex-direction: column; gap: 2px; }}
  .review-grid {{ grid-template-columns: 1fr; }} .rank-list .row {{ grid-template-columns: 110px 1fr 40px; }} }}
</style>""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def chip_html(skill, kind="", ai=False):
    return f'<span class="chip {kind}">{esc(skill)}{"<sup>AI</sup>" if ai else ""}</span>'


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{esc(title)}</div>'
                + (f'<div class="section-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def city(addr):
    if not addr:
        return ""
    last = addr.split(",")[-1].strip()
    return {"Ho Chi Minh": "TP.HCM", "Thành phố Hồ Chí Minh": "TP.HCM", "Ha Noi": "Hà Nội"}.get(last, last)


WORK_MODEL = {"At office": "Tại văn phòng", "Hybrid": "Hybrid", "Remote": "Từ xa"}


def posted_vi(days):
    if days is None:
        return ""
    if days < 1:
        return f"Đăng {max(1, round(days * 24))} giờ trước"
    if days < 30:
        return f"Đăng {round(days)} ngày trước"
    return f"Đăng {round(days / 30)} tháng trước"


def job_card(rank, r):
    j = r["job"]
    tags, llm_sk = set(j["skills_tag"]), set(j["skills_llm"] or [])
    is_ai = lambda s: s in llm_sk and s not in tags  # kỹ năng chỉ có nhờ AI trích từ JD
    pills = []
    if j["internship"]:
        pills.append('<span class="pill ok">Nhận Intern</span>')
    if j["fresher"]:
        pills.append('<span class="pill ok">Nhận Fresher</span>')
    if j["level"]:
        pills.append(f'<span class="pill">{esc(j["level"].capitalize())}</span>')
    if j["required_years"]:
        pills.append(f'<span class="pill">{j["required_years"]:g}+ năm KN</span>')
    if j.get("working_model"):
        pills.append(f'<span class="pill">{esc(WORK_MODEL.get(j["working_model"], j["working_model"]))}</span>')
    meta = " · ".join(x for x in [esc(j["company"]), esc(city(j.get("address"))),
                                  posted_vi(j.get("posted_days"))] if x)
    n_total = len(r["matched"]) + len(r["missing"])
    have = "".join(chip_html(s, "have", is_ai(s)) for s in r["matched"]) or '<span class="muted">—</span>'
    need = "".join(chip_html(s, "need", is_ai(s)) for s in r["missing"][:10]) or '<span class="muted">Không thiếu</span>'
    more = f'<span class="muted">+{len(r["missing"]) - 10}</span>' if len(r["missing"]) > 10 else ""

    def sig(label, val, right):
        return (f'<div><div class="sig-label">{label}<b>{right}</b></div>'
                f'<div class="bar"><i style="width:{max(2, round(val * 100))}%"></i></div></div>')

    return ("".join([
        '<div class="job"><div class="job-head">',
        f'<div class="rank">{rank}</div><div class="job-main">',
        f'<a class="job-title" href="{esc(j["url"])}" target="_blank">{esc(j["title"])}</a>',
        f'<div class="job-meta">{meta}</div><div>{"".join(pills)}</div></div>',
        f'<div class="score"><div class="score-num">{round(r["score"] * 100)}<span>%</span></div>'
        '<div class="score-label">phù hợp</div></div></div>',
        '<div class="signals">',
        sig("Giống về nghĩa", r["emb"], f'{r["emb"]:.0%}'),
        sig("Trùng từ khoá", r["tfidf"], f'{r["tfidf"]:.0%}'),
        sig("Đủ kỹ năng", len(r["matched"]) / n_total if n_total else 0, f'{len(r["matched"])}/{n_total}'),
        '</div>',
        f'<div class="skills-row"><div class="skills-label">Đã có</div><div>{have}</div></div>',
        f'<div class="skills-row"><div class="skills-label">Còn thiếu</div><div>{need}{more}</div></div>',
        '</div>']))


# ================================================================== sidebar
with st.sidebar:
    st.markdown("### Tuỳ chỉnh")
    use_filter = st.toggle("Chỉ tin Intern / Fresher / Junior", value=True)
    top_k = st.slider("Số tin hiển thị", 5, 30, config.TOP_K)
    with st.expander("Trọng số xếp hạng"):
        w0 = load_weights()
        w2 = st.slider("Giống về nghĩa (embedding)", 0.0, 1.0, float(w0[1]), 0.05)
        w1 = st.slider("Trùng từ khoá (TF-IDF)", 0.0, 1.0, float(w0[0]), 0.05)
        w3 = st.slider("Đủ kỹ năng", 0.0, 1.0, float(w0[2]), 0.05)
        st.caption("Mặc định " + ("chọn bằng đánh giá" if config.BEST_WEIGHTS_FILE.exists() else "chưa qua đánh giá"))
    tot = (w1 + w2 + w3) or 1
    weights = (w1 / tot, w2 / tot, w3 / tot)
    st.divider()
    st.markdown("**Cách hệ thống chấm điểm**")
    st.caption("Mỗi tin được so với CV theo 3 tiêu chí: **giống về nghĩa** (mô hình ngôn ngữ đa ngữ, hiểu cả "
               "tiếng Việt lẫn tiếng Anh), **trùng từ khoá** (TF-IDF) và **đủ kỹ năng** (so danh sách kỹ năng). "
               "Ba điểm được cộng theo trọng số. Các tin Senior/Lead hoặc yêu cầu từ 3 năm kinh nghiệm bị lọc.")
    st.caption(f"Kỹ năng của tin = tag ITviec + kỹ năng AI trích từ phần yêu cầu (nhãn AI). "
               f"Hiện AI đã phân tích {N_AI_JOBS}/{len(ELIGIBLE)} tin.")

tab_rec, tab_market, tab_eval = st.tabs(["Gợi ý việc làm", "Thị trường", "Đánh giá"])

# ================================================================== thị trường
with tab_market:
    st.markdown(f'<div class="hero"><div class="eyebrow">Thị trường · cập nhật {UPDATED}</div>'
                f'<h1>Kỹ năng nhà tuyển dụng đang cần</h1>'
                f'<p>Thống kê từ các tin AI/ML Engineer và Data Engineer trên ITviec.</p></div>',
                unsafe_allow_html=True)
    scope = st.segmented_control("Phạm vi", ["Tin hợp sinh viên", "Tất cả tin"], default="Tin hợp sinh viên",
                                 label_visibility="collapsed") or "Tin hợp sinh viên"
    pool = ELIGIBLE if scope == "Tin hợp sinh viên" else m.jobs
    cnt = Counter(s for j in pool for s in j["job_skills"])
    df = pd.DataFrame(cnt.most_common(20), columns=["Kỹ năng", "Số tin"])
    df["Tỷ lệ"] = df["Số tin"] / len(pool)
    st.markdown('<div class="stats">'
                f'<span class="stat"><b>{len(pool)}</b> tin</span>'
                f'<span class="stat"><b>{sum(j["fresher"] for j in pool)}</b> nhận Fresher</span>'
                f'<span class="stat"><b>{sum(j["internship"] for j in pool)}</b> nhận Intern</span>'
                f'<span class="stat"><b>{len(cnt)}</b> kỹ năng khác nhau</span></div>', unsafe_allow_html=True)
    section("20 kỹ năng được yêu cầu nhiều nhất", f"Số tin (trong {len(pool)} tin) có yêu cầu kỹ năng đó")
    chart = (alt.Chart(df).mark_bar(color=T["accent"], cornerRadiusEnd=4, height={"band": 0.62})
             .encode(x=alt.X("Số tin:Q", title=None, axis=alt.Axis(grid=True, tickCount=5)),
                     y=alt.Y("Kỹ năng:N", sort="-x", title=None),
                     tooltip=["Kỹ năng", "Số tin", alt.Tooltip("Tỷ lệ:Q", format=".0%")])
             .properties(height=26 * len(df))
             .configure_axis(labelFontSize=12, labelColor=T["muted"], domain=False, ticks=False,
                             gridColor=T["border"], labelFont="Be Vietnam Pro")
             .configure_view(strokeWidth=0))
    st.altair_chart(chart, width="stretch")
    with st.expander("Xem dạng bảng"):
        st.dataframe(df.assign(**{"Tỷ lệ": df["Tỷ lệ"].map("{:.0%}".format)}), hide_index=True, width="stretch")
    section("Cấp bậc trong toàn bộ kho tin", "Suy ra từ tên tin")
    lv = Counter((j["level"] or "Không ghi") for j in m.jobs)
    st.dataframe(pd.DataFrame([(k.capitalize(), v) for k, v in lv.most_common()], columns=["Cấp bậc", "Số tin"]),
                 hide_index=True)

# ================================================================== đánh giá
with tab_eval:
    rep = config.RESULTS / "report.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.markdown('<div class="hero"><div class="eyebrow">Đánh giá</div><h1>Chưa có kết quả đánh giá</h1>'
                    '<p>Chạy <code>python scripts/evaluate.py</code> để đo Precision@5 và NDCG@5 '
                    'của hệ thống gợi ý.</p></div>', unsafe_allow_html=True)

# ================================================================== gợi ý (đặt cuối vì có st.stop())
with tab_rec:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">ITviec · AI/ML & Data Engineering · cập nhật {UPDATED}</div>'
        '<h1>Tìm việc IT phù hợp với CV của bạn</h1>'
        '<p>Tải CV lên, hệ thống so khớp với các tin Intern, Fresher và Junior đang mở, chỉ ra kỹ năng bạn đã có, '
        'kỹ năng còn thiếu, và nhờ AI nhận xét CV.</p>'
        '<div class="stats">'
        f'<span class="stat"><b>{len(m.jobs)}</b> tin đang mở</span>'
        f'<span class="stat"><b>{len(ELIGIBLE)}</b> tin hợp sinh viên</span>'
        f'<span class="stat">AI đã phân tích <b>{N_AI_JOBS}/{len(ELIGIBLE)}</b> tin</span>'
        f'<span class="stat">AI: <b>{esc(llm.model_name(LLM_CANDIDATES[0])) if llm_ok else "chưa có API key"}</b></span>'
        '</div></div>', unsafe_allow_html=True)

    up = st.file_uploader("CV của bạn (PDF)", type="pdf")
    c_paste, c_sample = st.columns([3, 2], vertical_alignment="top")
    with c_paste.expander("Hoặc dán nội dung CV"):
        pasted = st.text_area("Nội dung CV", height=200, label_visibility="collapsed",
                              placeholder="Dán toàn bộ nội dung CV vào đây...")
    samples = {"CV mẫu Data Engineer": config.ROOT / "examples" / "sample_cv_data_engineer.txt",
               "CV mẫu AI Engineer": config.ROOT / "examples" / "sample_cv_ai.txt"}
    samples = {k: p for k, p in samples.items() if p.exists()}
    qs = st.query_params.get("sample")  # vd. ?sample=CV mẫu AI Engineer — tiện khi demo
    sample = c_sample.pills("Chưa có CV? Thử với", list(samples), default=qs if qs in samples else None)

    cv_text = ""
    if up is not None:
        cv_text = read_pdf(up.getvalue())
        if len(cv_text) < 100:
            st.warning("Đọc được rất ít chữ từ PDF. CV có thể là ảnh scan — hãy dán nội dung CV vào ô bên trên.")
    elif pasted.strip():
        cv_text = pasted
    elif sample:
        cv_text = samples[sample].read_text(encoding="utf-8")
    if not cv_text:
        st.stop()

    cv_text = anonymize(cv_text)
    cv_key = hashlib.sha256(cv_text.encode()).hexdigest()
    reviews = st.session_state.setdefault("reviews", {})  # nhận xét lưu theo CV → rerun không gọi lại LLM
    review = reviews.get(cv_key)

    dict_skills = m.extractor.extract(cv_text)
    extra = canonical_set(review.get("extra_skills", [])) - dict_skills if review else set()
    use_extra = bool(extra) and st.session_state.get("use_extra", True)
    with st.spinner("Đang so khớp với các tin tuyển dụng..."):
        recs, have = m.recommend(cv_text, weights, int(top_k), use_filter,
                                 dict_skills | extra if use_extra else dict_skills)
    missing = Counter(s for r in recs for s in r["missing"]).most_common()

    # ---------- hồ sơ + nhận xét AI ----------
    col_profile, col_ai = st.columns([2, 3], gap="large")
    with col_profile:
        section("Kỹ năng trong CV", f"{len(dict_skills)} kỹ năng nhận diện"
                + (f" · {len(extra)} do AI phát hiện thêm (nhãn AI)" if extra else ""))
        st.markdown('<div class="card">'
                    + ("".join(chip_html(s) for s in sorted(dict_skills))
                       + "".join(chip_html(s, ai=True) for s in sorted(extra))
                       or '<span class="muted">Chưa nhận diện được kỹ năng nào.</span>')
                    + "</div>", unsafe_allow_html=True)
        if extra:
            st.toggle(f"Tính cả {len(extra)} kỹ năng AI phát hiện khi xếp hạng", value=True, key="use_extra")
        section("Nên học tiếp", "Số tin trong danh sách gợi ý cần kỹ năng mà CV chưa có")
        top_miss = missing[:8]
        if top_miss:
            mx = top_miss[0][1]
            rows = "".join(f'<div class="row"><span>{esc(s)}</span><div class="bar"><i style="width:'
                           f'{round(n / mx * 100)}%"></i></div><span>{n} tin</span></div>' for s, n in top_miss)
            st.markdown(f'<div class="card rank-list">{rows}</div>', unsafe_allow_html=True)

    with col_ai:
        section("Nhận xét từ AI" + ("" if not review else ""),
                "Điểm mạnh, điểm cần cải thiện, cách sửa CV và lộ trình học — dựa trên các tin được gợi ý")
        if not llm_ok:
            st.info("Thêm GEMINI_API_KEY vào file .env để bật nhận xét bằng AI.")
        elif review is None:
            with st.container(border=True):
                st.markdown("AI đọc CV cùng các tin phù hợp nhất để đưa ra nhận xét cụ thể: CV mạnh ở đâu, "
                            "còn thiếu gì, nên sửa gì và nên học gì tiếp theo.")
                if st.button("Nhận xét CV của tôi", type="primary", width="stretch"):
                    try:
                        with st.spinner("AI đang đọc CV của bạn..."):
                            review, spec = review_with_fallback(cv_text, have, recs, missing)
                        review["_model"] = llm.model_name(spec)
                        reviews[cv_key] = review
                        st.rerun()  # vẽ lại để áp dụng kỹ năng AI phát hiện thêm vào xếp hạng
                    except Exception as e:
                        st.warning(friendly_error(e))
                st.caption("Dùng 1 lượt gọi AI cho mỗi CV; kết quả được lưu lại.")
        else:
            strengths = "".join(f"<li>{esc(x)}</li>" for x in review.get("strengths", []))
            weaknesses = "".join(f"<li>{esc(x)}</li>" for x in review.get("weaknesses", []))
            fixes = "".join(f'<div class="fix"><b>{esc(f.get("issue", ""))}</b><div>{esc(f.get("fix", ""))}</div></div>'
                            for f in review.get("cv_fixes", []))
            st.markdown(
                '<div class="card">'
                f'<div class="review-summary">{esc(review.get("summary", ""))}</div>'
                '<div class="review-grid">'
                f'<div><div class="review-h">Điểm mạnh</div><ul class="review-list">{strengths}</ul></div>'
                f'<div><div class="review-h">Cần cải thiện</div><ul class="review-list">{weaknesses}</ul></div></div>'
                + (f'<div class="review-h" style="margin-top:14px">Phù hợp nhất với</div>'
                   f'<div class="review-list" style="padding:0">{esc(review["fit"])}</div>' if review.get("fit") else "")
                + (f'<div class="review-h" style="margin-top:14px">Nên sửa trong CV</div>{fixes}' if fixes else "")
                + f'<div class="muted" style="margin-top:12px">Nhận xét bởi {esc(review.get("_model", "AI"))}</div>'
                '</div>', unsafe_allow_html=True)
            steps = review.get("roadmap", [])
            if steps:
                st.markdown('<div class="review-h" style="margin-top:4px">Lộ trình học đề xuất</div>',
                            unsafe_allow_html=True)
                for i, s in enumerate(steps, 1):
                    with st.expander(f"{i}. {s.get('skill', '')} · {s.get('weeks', '?')} tuần", expanded=i == 1):
                        st.markdown(f"{s.get('why', '')}")
                        if s.get("topics"):
                            st.markdown("**Nội dung:** " + "; ".join(s["topics"]))
                        if s.get("resources"):
                            st.markdown("**Tài nguyên:**\n" + "\n".join(f"- {x}" for x in s["resources"]))
                        if s.get("project"):
                            st.markdown(f"**Mini-project:** {s['project']}")
            if st.button("Ẩn nhận xét", type="tertiary"):
                reviews.pop(cv_key, None)
                st.rerun()

    # ---------- danh sách tin ----------
    section(f"{len(recs)} tin phù hợp nhất",
            "Điểm phù hợp = tổng có trọng số của 3 tiêu chí; mỗi thanh so với các tin khác trong kho. "
            'Kỹ năng có nhãn <sup style="color:' + T["ai"] + ';font-weight:700">AI</sup> do AI trích từ phần yêu cầu của tin.')
    for rank, r in enumerate(recs, 1):
        st.markdown(job_card(rank, r), unsafe_allow_html=True)
