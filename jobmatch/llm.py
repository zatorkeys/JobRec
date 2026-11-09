"""Client tối giản cho OpenAI và Gemini (gọi REST trực tiếp), trả JSON, có cache SQLite.

    from jobmatch.llm import ask_json
    data = ask_json("gemini", prompt)          # dict
    data = ask_json("openai", prompt, variant="run2")   # variant khác → không dùng lại cache
"""
import hashlib
import itertools
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config

_lock = threading.Lock()
_conn = None


def _db():
    global _conn
    if _conn is None:
        config.LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.LLM_CACHE, check_same_thread=False)
        _conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, model TEXT, v TEXT, ts REAL)")
    return _conn


def split_spec(spec):
    """'gemini' → ('gemini', GEMINI_MODEL);  'gemini:gemini-3.5-flash' → ('gemini', 'gemini-3.5-flash')"""
    provider, _, model = spec.partition(":")
    return provider, model or {"openai": config.OPENAI_MODEL, "gemini": config.GEMINI_MODEL}[provider]


def model_name(spec):
    return split_spec(spec)[1]


def available(spec):
    return bool({"openai": config.OPENAI_API_KEY, "gemini": config.GEMINI_API_KEYS}[split_spec(spec)[0]])


_rl_lock = threading.Lock()
_next_slot = {}
_dead = set()          # (chỉ số key, model) đã hết quota ngày
_rr = itertools.count()  # luân phiên giữa các key Gemini


def _pick_key(provider, model):
    """Chỉ số key dùng cho request tiếp theo; None nếu mọi key Gemini đều hết quota ngày với model này."""
    if provider != "gemini":
        return 0
    alive = [i for i in range(len(config.GEMINI_API_KEYS)) if (i, model) not in _dead]
    if not alive:
        return None
    with _rl_lock:
        return alive[next(_rr) % len(alive)]


def _alive_keys(model):
    return [i for i in range(len(config.GEMINI_API_KEYS)) if (i, model) not in _dead]


def _throttle(provider, model, kid):
    """Giãn đều request theo RPM cho từng (key, model) — quota Gemini tính riêng theo project & model."""
    rpm = config.GEMINI_RPM if provider == "gemini" else config.OPENAI_RPM
    if rpm <= 0:
        return
    with _rl_lock:
        now = time.time()
        slot = max(now, _next_slot.get((kid, model), 0.0))
        _next_slot[(kid, model)] = slot + 60.0 / rpm
    if slot > now:
        time.sleep(slot - now)


def _retry_delay(resp):
    """Đọc retryDelay (vd. '37s') trong lỗi 429 của Gemini."""
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', resp.text)
    return float(m.group(1)) if m else None


_health = {}


def healthy(spec):
    """Có key VÀ gọi được thật (key hết credit / sai model → False). Kết quả nhớ trong phiên chạy."""
    if spec not in _health:
        try:
            _health[spec] = available(spec) and ask_json(spec, 'Return JSON {"ok": true}', use_cache=False,
                                                         retries=2).get("ok") is True
        except Exception as e:
            print(f"! {spec} ({model_name(spec)}) không dùng được: {str(e)[:200]}")
            _health[spec] = False
    return _health[spec]


_ROLE_CFG = {"pipeline": "PIPELINE_LLM", "judge_main": "JUDGE_MAIN", "judge_second": "JUDGE_SECOND"}
_roles = {}


def role(name):
    """Spec LLM cho một vai trò. Nếu cấu hình (vd. openai) không gọi được → chuyển sang Gemini dự phòng.
    Trả None nếu không có LLM nào dùng được."""
    if name not in _roles:
        spec = getattr(config, _ROLE_CFG[name])
        if not healthy(spec):
            fb = config.FALLBACK_ROLES[name]
            if fb != spec and healthy(fb):
                print(f"→ {name}: {spec} không dùng được, chuyển sang {fb}")
                spec = fb
            else:
                spec = None
        _roles[name] = spec
    return _roles[name]


def _call_openai(prompt, temperature, model, kid=0):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        body["temperature"] = temperature
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=120,
                      headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"})
    if r.status_code == 400 and "temperature" in r.text and temperature is not None:
        return _call_openai(prompt, None, model)  # một số model chỉ nhận temperature mặc định
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(prompt, temperature, model, kid=0):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    if temperature is not None:
        body["generationConfig"]["temperature"] = temperature
    if config.GEMINI_THINKING and model.startswith("gemini-3"):  # giảm "suy nghĩ" để nhanh & rẻ hơn
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": config.GEMINI_THINKING}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, json=body, timeout=120, headers={"x-goog-api-key": config.GEMINI_API_KEYS[kid]})
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _parse(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def ask_json(spec, prompt, temperature=0.0, variant="", use_cache=True, retries=8):
    """spec: 'openai' | 'gemini' | 'provider:model'"""
    if not spec:
        raise RuntimeError("Không có LLM nào dùng được (kiểm tra API key / credit trong .env).")
    provider, model = split_spec(spec)
    if not available(spec):
        raise RuntimeError(f"Chưa có API key cho {provider}. Thêm vào file .env (xem .env.example).")
    key = hashlib.sha256(f"{provider}|{model}|{temperature}|{variant}|{prompt}".encode()).hexdigest()
    if use_cache:
        with _lock:
            row = _db().execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

    call = _call_openai if provider == "openai" else _call_gemini
    last = None
    for attempt in range(retries):
        wait = min(60, 2 ** attempt * 2)
        kid = _pick_key(provider, model)
        if kid is None:
            raise RuntimeError(f"{spec}: mọi key Gemini đều đã hết quota trong ngày của model {model}. "
                               "Chạy lại vào ngày mai (phần đã xong được cache), thêm key ở project khác, "
                               "hoặc bật billing.")
        try:
            _throttle(provider, model, kid)
            data = _parse(call(prompt, temperature, model, kid))
            with _lock:
                _db().execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                              (key, model, json.dumps(data, ensure_ascii=False), time.time()))
                _db().commit()
            return data
        except requests.HTTPError as e:
            last = e
            code, body = e.response.status_code, e.response.text
            if code not in (429, 500, 502, 503, 504) or "insufficient_quota" in body:
                raise RuntimeError(f"{spec} lỗi {code}: {body[:300]}") from e
            if code == 429 and provider == "gemini" and "PerDay" in body:
                _dead.add((kid, model))  # key này hết quota ngày với model này → thử key khác ngay
                if _alive_keys(model):
                    print(f"  → key Gemini #{kid + 1} hết quota ngày ({model}), chuyển sang key khác")
                    continue
                raise RuntimeError(f"{spec}: mọi key Gemini đều đã hết quota trong ngày của model {model}. "
                                   "Chạy lại vào ngày mai (phần đã xong được cache), thêm key ở project khác, "
                                   "hoặc bật billing.") from e
            if code == 429:
                # còn key khác → đổi key gần như ngay; chỉ còn 1 key → chờ đúng retryDelay
                wait = 1 if len(_alive_keys(model)) > 1 else (_retry_delay(e.response) or wait) + 1
        except (json.JSONDecodeError, KeyError, IndexError, requests.ConnectionError, requests.Timeout) as e:
            last = e
        time.sleep(wait)
    raise RuntimeError(f"{provider} thất bại sau {retries} lần: {last}")


def map_parallel(fn, items, workers=None, desc=""):
    """Chạy fn(item) song song, in tiến độ, trả list kết quả (None nếu lỗi)."""
    workers = workers or config.LLM_WORKERS
    out = [None] * len(items)

    def run(i):
        try:
            out[i] = fn(items[i])
        except Exception as e:  # không để 1 lỗi làm hỏng cả đợt
            print(f"  ! {desc} #{i}: {e}")

    with ThreadPoolExecutor(workers) as ex:
        for n, _ in enumerate(ex.map(run, range(len(items))), 1):
            if n % 25 == 0 or n == len(items):
                print(f"  {desc}: {n}/{len(items)}")
    return out
