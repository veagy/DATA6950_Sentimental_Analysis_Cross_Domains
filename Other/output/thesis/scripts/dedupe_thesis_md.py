"""
Deduplicate THESIS_REPORT.md -> THESIS_REPORT_NEW.md

1. Remove page footers: standalone "December 3, 2025" and lone 1–2 digit page numbers.
2. Remove consecutive duplicate lines (length > 8).
3. Remove blank-line-delimited paragraphs that repeat exactly (normalized), min 48 chars.
4. Global: drop lines whose normalized form appears 3+ times in the file (keep first
   occurrence only). Min line length 22; skips |, Fig./TABLE lines, and short numbered
   list markers to reduce collateral removal.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "markdown" / "THESIS_REPORT.md"
DST = ROOT / "markdown" / "THESIS_REPORT_NEW.md"

FOOTER_DATE = re.compile(r"^December 3, 2025\s*$")
FOOTER_PAGE = re.compile(r"^\s*\d{1,2}\s*$")
PAGE_HEADER = re.compile(r"^## Page \d+\s*$")
IMAGE_LINE = re.compile(r"^!\[")
HR_LINE = re.compile(r"^---\s*$")


def norm_para(s: str) -> str:
    return " ".join(s.split())


def line_key(line: str) -> str:
    return " ".join(line.strip().split())


def should_skip_global_dedupe(line: str) -> bool:
    s = line.strip()
    if len(s) < 22:
        return True
    if s.startswith("|"):
        return True
    if re.match(r"^(Fig\.|FIG\.|TABLE|Table|Figure)\b", s):
        return True
    if re.match(r"^\d+\s*[\).]", s) and len(s) < 80:
        return True
    return False


def extract_body_lines_for_counting(text: str) -> list[str]:
    """Lines inside page bodies only (not headers, images, preamble)."""
    parts = re.split(r"## Page \d+\s*\n", text)
    out: list[str] = []
    for chunk in parts[1:]:
        m = re.search(r"!\[[^\]]*\]\([^)]+\)\s*\n", chunk)
        if not m:
            continue
        rest = chunk[m.end() :]
        sep = rest.rfind("\n---")
        body = rest[:sep] if sep != -1 else rest
        for line in body.splitlines():
            if (
                not line.strip()
                or PAGE_HEADER.match(line)
                or IMAGE_LINE.match(line)
                or HR_LINE.match(line)
            ):
                continue
            out.append(line)
    return out


def count_frequent_lines(text: str) -> set[str]:
    lines = extract_body_lines_for_counting(text)
    keys = [line_key(L) for L in lines if not should_skip_global_dedupe(L)]
    cnt = Counter(k for k in keys if k)
    return {k for k, n in cnt.items() if n >= 3}


def strip_footer_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        t = line.strip()
        if FOOTER_DATE.match(t) or FOOTER_PAGE.match(t):
            continue
        out.append(line)
    return out


def dedupe_consecutive_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    prev: str | None = None
    for line in lines:
        cur = line.strip()
        if cur and prev is not None and cur == prev and len(cur) > 8:
            continue
        out.append(line)
        prev = cur if cur else prev
    return out


def process_body(
    body_lines: list[str],
    seen_paras: set[str],
    frequent_lines: set[str],
    emitted_frequent: set[str],
) -> list[str]:
    body_lines = strip_footer_lines(body_lines)
    body_lines = dedupe_consecutive_lines(body_lines)

    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        raw = "\n".join(buf)
        key = norm_para(raw)
        if len(key) >= 48 and key in seen_paras:
            buf = []
            return
        if len(key) >= 48:
            seen_paras.add(key)
        for ln in buf:
            k = line_key(ln)
            if k and k in frequent_lines:
                if k in emitted_frequent:
                    continue
                emitted_frequent.add(k)
            out.append(ln)
        buf = []

    for line in body_lines:
        if not line.strip():
            flush()
            buf = []
            if out and out[-1] != "":
                out.append("")
        else:
            buf.append(line)
    flush()

    while out and out[-1] == "":
        out.pop()
    return out


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    frequent_lines = count_frequent_lines(text)
    emitted_frequent: set[str] = set()

    parts = re.split(r"(## Page \d+\s*\n)", text)
    out_chunks: list[str] = [parts[0]]
    seen_paras: set[str] = set()

    i = 1
    while i < len(parts):
        header = parts[i].strip() + "\n"
        body_and_rest = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        m_img = re.search(r"(!\[[^\]]*\]\([^)]+\))\s*\n", body_and_rest)
        if not m_img:
            out_chunks.append(header + body_and_rest)
            continue

        img_line = m_img.group(1)
        after = body_and_rest[m_img.end() :]
        sep = after.rfind("\n---")
        if sep == -1:
            body_text = after
            tail = ""
        else:
            body_text = after[:sep]
            tail = after[sep:]

        body_lines = body_text.splitlines()
        processed = process_body(
            body_lines, seen_paras, frequent_lines, emitted_frequent
        )
        body_out = "\n".join(processed)
        rebuilt = f"{header}\n{img_line}\n\n{body_out}{tail}"
        out_chunks.append(rebuilt)

    DST.write_text("".join(out_chunks), encoding="utf-8")
    print(f"Wrote {DST} ({DST.stat().st_size} bytes)")
    print(f"Frequent-line keys (>=3): {len(frequent_lines)}")


if __name__ == "__main__":
    main()
