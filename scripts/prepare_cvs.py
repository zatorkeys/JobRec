"""Chuẩn bị CV thật cho đánh giá: data/cvs_raw/*.pdf → ẩn danh → data/cvs/cv_XX.txt

    python scripts/prepare_cvs.py          # regex + LLM ẩn danh tên người/trường/công ty
    python scripts/prepare_cvs.py --no-llm # chỉ regex (email, SĐT, link, ngày sinh)

Tên file đầu ra không chứa tên thật. Bảng đối chiếu không được lưu lại.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jobmatch import config, llm, llm_tasks  # noqa: E402
from jobmatch.cv import anonymize, read_pdf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    pdfs = sorted(config.CV_RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Không có PDF nào trong {config.CV_RAW_DIR}. Bỏ CV thật (đã xin phép) vào đó rồi chạy lại.")
        return
    config.CV_DIR.mkdir(parents=True, exist_ok=True)
    use_llm = not args.no_llm and llm.role("pipeline") is not None
    for i, pdf in enumerate(pdfs, 1):
        text = anonymize(read_pdf(pdf))
        if len(text) < 200:
            print(f"! {pdf.name}: đọc được quá ít chữ (CV dạng ảnh?) — bỏ qua")
            continue
        if use_llm:
            text = anonymize(llm_tasks.anonymize_llm(text))
        out = config.CV_DIR / f"cv_{i:02d}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"{out.name}: {len(text)} ký tự")
    print(f"Xong. {'Đã' if use_llm else 'CHƯA'} ẩn danh tên người/trường bằng LLM.")


if __name__ == "__main__":
    main()
