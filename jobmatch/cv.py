"""Đọc CV PDF và ẩn danh thông tin cá nhân."""
import io
import re

import pdfplumber

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_URL = re.compile(r"(https?://\S+|www\.\S+|\b(?:github|gitlab|linkedin|facebook|fb)\.com/\S*)", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\d)")
_DOB = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-](19|20)\d{2}\b")


def read_pdf(source):
    """source: đường dẫn hoặc bytes (file upload của Streamlit)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with pdfplumber.open(source) as pdf:
        pages = [p.extract_text(x_tolerance=1.5) or "" for p in pdf.pages]
    return "\n".join(pages).strip()


def anonymize(text):
    """Ẩn email, link, số điện thoại, ngày sinh. Tên người & trường: dùng llm_tasks.anonymize_llm."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[LINK]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _DOB.sub("[DOB]", text)
    return text
