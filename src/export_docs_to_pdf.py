from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def register_fonts() -> str:
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont("ArialUnicode", str(FONT_PATH)))
        return "ArialUnicode"
    return "Helvetica"


def strip_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)


def inline_markdown_to_html(text: str) -> str:
    text = strip_markdown_links(text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def markdown_to_flowables(markdown: str, font_name: str) -> list:
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "NormalVietnamese",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )
    h1 = ParagraphStyle("H1Vietnamese", parent=normal, fontSize=18, leading=23, spaceBefore=8, spaceAfter=12)
    h2 = ParagraphStyle("H2Vietnamese", parent=normal, fontSize=14, leading=19, spaceBefore=10, spaceAfter=8)
    h3 = ParagraphStyle("H3Vietnamese", parent=normal, fontSize=12, leading=16, spaceBefore=8, spaceAfter=6)
    bullet = ParagraphStyle("BulletVietnamese", parent=normal, leftIndent=14, firstLineIndent=-10)
    code = ParagraphStyle(
        "Code",
        parent=normal,
        fontName="Courier",
        fontSize=8.5,
        leading=11,
        leftIndent=10,
        rightIndent=10,
        backColor="#F4F4F4",
        borderPadding=6,
    )

    flowables = []
    in_code = False
    code_lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flowables.append(Preformatted("\n".join(code_lines), code))
                flowables.append(Spacer(1, 6))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            flowables.append(Spacer(1, 4))
            continue
        if line.startswith("# "):
            flowables.append(Paragraph(inline_markdown_to_html(line[2:].strip()), h1))
        elif line.startswith("## "):
            flowables.append(Paragraph(inline_markdown_to_html(line[3:].strip()), h2))
        elif line.startswith("### "):
            flowables.append(Paragraph(inline_markdown_to_html(line[4:].strip()), h3))
        elif re.match(r"^\d+\.\s+", line):
            text = re.sub(r"^\d+\.\s+", "", line)
            flowables.append(Paragraph(inline_markdown_to_html(text), bullet, bulletText="•"))
        elif line.startswith("- "):
            flowables.append(Paragraph(inline_markdown_to_html(line[2:].strip()), bullet, bulletText="•"))
        else:
            flowables.append(Paragraph(inline_markdown_to_html(line), normal))
    if code_lines:
        flowables.append(Preformatted("\n".join(code_lines), code))
    return flowables


def export_markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    font_name = register_fonts()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=markdown_path.stem,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    doc.build(markdown_to_flowables(markdown, font_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export project Markdown documents to PDF.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("report/final_report_outline.md"), Path("docs/PROJECT_WORKFLOW.md")])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for markdown_path in args.paths:
        pdf_path = markdown_path.with_suffix(".pdf")
        export_markdown_to_pdf(markdown_path, pdf_path)
        print(f"Wrote {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
