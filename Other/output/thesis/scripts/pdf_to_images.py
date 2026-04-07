"""
Convert PDF(s) under output/thesis to raster images under output/imgs.

Requires: pip install pymupdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _output_root() -> Path:
    # output/scripts/this_file.py -> output/
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render PDF pages from output/thesis to PNG files in output/imgs."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Specific PDF path (default: all *.pdf in output/thesis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Render resolution in dots per inch (default: 150).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpeg", "jpg"),
        default="png",
        help="Output image format (default: png).",
    )
    args = parser.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Missing dependency. Install with: pip install pymupdf", file=sys.stderr)
        return 1

    root = _output_root()
    thesis_dir = root / "thesis"
    imgs_dir = root / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf is not None:
        pdf_path = args.pdf.resolve()
        if not pdf_path.is_file():
            print(f"PDF not found: {pdf_path}", file=sys.stderr)
            return 1
        pdfs = [pdf_path]
    else:
        if not thesis_dir.is_dir():
            print(f"Thesis folder not found: {thesis_dir}", file=sys.stderr)
            return 1
        pdfs = sorted(thesis_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files in {thesis_dir}", file=sys.stderr)
            return 1

    ext = "jpg" if args.format in ("jpeg", "jpg") else "png"
    zoom = args.dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    total_pages = 0
    for pdf_path in pdfs:
        doc = fitz.open(pdf_path)
        stem = pdf_path.stem
        n = len(doc)
        for i in range(n):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_name = f"{stem}_page_{i + 1:04d}.{ext}"
            out_path = imgs_dir / out_name
            pix.save(out_path.as_posix())
            total_pages += 1
        doc.close()
        print(f"{pdf_path.name}: {n} page(s) -> {imgs_dir}")

    print(f"Done. {total_pages} image file(s) in {imgs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
