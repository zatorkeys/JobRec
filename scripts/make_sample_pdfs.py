"""Tạo PDF từ các CV mẫu examples/*.txt (để thử demo).  python scripts/make_sample_pdfs.py"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
FONT = Path("C:/Windows/Fonts/arial.ttf")  # cần font Unicode cho tiếng Việt; đổi đường dẫn nếu dùng Linux/Mac

for txt in sorted((ROOT / "examples").glob("*.txt")):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", fname=str(FONT))
    pdf.set_font("Arial", size=10.5)
    for line in txt.read_text(encoding="utf-8").splitlines():
        pdf.multi_cell(0, 5.5, line or " ", new_x="LMARGIN", new_y="NEXT")
    out = txt.with_suffix(".pdf")
    pdf.output(str(out))
    print(out.name)
