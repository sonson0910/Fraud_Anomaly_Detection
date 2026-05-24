from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
PAGE = landscape((7.5 * inch, 13.333 * inch))
BLUE = colors.HexColor("#2F6B8F")
RED = colors.HexColor("#C46243")
GREEN = colors.HexColor("#88A868")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
LIGHT = colors.HexColor("#EEF3F6")


def font_name() -> str:
    if FONT_PATH.exists() and "ArialUnicode" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("ArialUnicode", str(FONT_PATH)))
    return "ArialUnicode" if FONT_PATH.exists() else "Helvetica"


def wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def draw_text(c: canvas.Canvas, text: str, x: float, y: float, size: int = 18, color=INK, max_width: int = 80, leading: int | None = None) -> float:
    leading = leading or int(size * 1.35)
    c.setFillColor(color)
    c.setFont(font_name(), size)
    for line in wrap(text, max_width):
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_bullets(c: canvas.Canvas, bullets: list[str], x: float, y: float, size: int = 17, max_width: int = 78) -> float:
    c.setFont(font_name(), size)
    c.setFillColor(INK)
    for bullet in bullets:
        lines = wrap(bullet, max_width)
        c.drawString(x, y, "•")
        c.drawString(x + 0.25 * inch, y, lines[0])
        y -= size * 1.35
        for line in lines[1:]:
            c.drawString(x + 0.25 * inch, y, line)
            y -= size * 1.35
        y -= 0.08 * inch
    return y


def slide_header(c: canvas.Canvas, title: str, subtitle: str | None = None) -> None:
    c.setFillColor(BLUE)
    c.rect(0, PAGE[1] - 0.18 * inch, PAGE[0], 0.18 * inch, fill=1, stroke=0)
    draw_text(c, title, 0.65 * inch, PAGE[1] - 0.72 * inch, 26, INK, 70)
    if subtitle:
        draw_text(c, subtitle, 0.65 * inch, PAGE[1] - 1.08 * inch, 13, MUTED, 110)


def add_metric(c: canvas.Canvas, label: str, value: str, x: float, y: float, color=BLUE) -> None:
    c.setFillColor(LIGHT)
    c.roundRect(x, y, 2.65 * inch, 0.86 * inch, 6, fill=1, stroke=0)
    c.setFillColor(color)
    c.setFont(font_name(), 22)
    c.drawString(x + 0.18 * inch, y + 0.45 * inch, value)
    c.setFillColor(MUTED)
    c.setFont(font_name(), 10)
    c.drawString(x + 0.18 * inch, y + 0.18 * inch, label)


def add_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float) -> None:
    if path.exists():
        c.drawImage(str(path), x, y, width=w, height=h, preserveAspectRatio=True, anchor="c")


def build_pdf(output_path: Path, metrics: dict, root_cause: pd.DataFrame, figures_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=PAGE)
    c.setTitle("GContest Fraud Anomaly Detection Slide Deck")

    slide_header(c, "Fraud & Anomaly Detection", "Cause-first risk framework using real G'Contest banking data")
    add_metric(c, "Transactions scored", f"{metrics['row_counts']['transactions_scored']:,}", 0.65 * inch, 4.85 * inch)
    add_metric(c, "High/Critical cases", f"{metrics['review_queue']['high_or_critical_transactions']:,}", 3.55 * inch, 4.85 * inch, RED)
    add_metric(c, "Customers impacted", f"{metrics['review_queue']['customers_with_high_or_critical']:,}", 6.45 * inch, 4.85 * inch, GREEN)
    draw_bullets(
        c,
        [
            "No synthetic data and no fake fraud labels.",
            "Output is a review queue with explainable risk reasons.",
            "Framework aligns with risk-based banking controls: step-up authentication, manual review, AML escalation.",
        ],
        0.8 * inch,
        4.15 * inch,
        18,
        84,
    )
    c.showPage()

    slide_header(c, "Why Cause-First", "BGK note: identify root causes before introducing the model")
    draw_bullets(
        c,
        [
            "Account takeover / identity compromise: new device, new IP, night access, new beneficiary, late-stage digital activity.",
            "Unauthorized transfer / capital outflow: outside-bank transfer, amount above customer baseline, daily burst, cash-out versus CASA.",
            "AML network / mule-account pattern: shared IP/device/beneficiary, round high-value transfers, repeated external transfers.",
        ],
        0.8 * inch,
        5.55 * inch,
        17,
        92,
    )
    c.showPage()

    slide_header(c, "Framework", "Data dictionary -> behavioral baseline -> root-cause rules -> anomaly model -> xAI review queue")
    steps = [
        ("1. Data", "Customer, Transaction, Activity, Deposit, Lending, Card"),
        ("2. Baseline", "Customer amount P95, daily frequency, known device/IP/beneficiary"),
        ("3. Root causes", "ATO, unauthorized transfer, AML network scores"),
        ("4. Model", "Isolation Forest catches unusual combinations without fraud labels"),
        ("5. xAI", "Reason codes, SHAP surrogate, recommended action"),
    ]
    x = 0.65 * inch
    for idx, (title, body) in enumerate(steps):
        y = 5.4 * inch - idx * 0.86 * inch
        c.setFillColor(LIGHT if idx % 2 == 0 else colors.white)
        c.roundRect(x, y, 11.9 * inch, 0.65 * inch, 5, fill=1, stroke=0)
        draw_text(c, title, x + 0.22 * inch, y + 0.40 * inch, 15, BLUE, 24)
        draw_text(c, body, x + 2.0 * inch, y + 0.40 * inch, 14, INK, 82)
    c.showPage()

    slide_header(c, "Key Insights", "Top-risk queue is explainable by root-cause branch")
    add_image(c, figures_dir / "root_cause_high_critical.png", 0.55 * inch, 1.1 * inch, 5.9 * inch, 4.2 * inch)
    add_image(c, figures_dir / "risk_band_counts.png", 6.75 * inch, 1.1 * inch, 5.7 * inch, 4.2 * inch)
    c.showPage()

    slide_header(c, "Temporal Stability", "2019 monthly/quarterly backtest for review-rate stability")
    add_image(c, figures_dir / "monthly_stability_backtest.png", 0.8 * inch, 1.0 * inch, 11.7 * inch, 4.8 * inch)
    draw_bullets(
        c,
        [
            "Only 2019 data is available, so this is a temporal robustness check rather than a crisis-period validation.",
            "The monitoring KPI is not accuracy without labels; it is review queue volume, high-risk rate, and reason-code consistency over time.",
        ],
        0.9 * inch,
        1.05 * inch,
        14,
        95,
    )
    c.showPage()

    slide_header(c, "xAI Engine", "SHAP explains a tree surrogate of the hybrid risk score")
    add_image(c, figures_dir / "shap_feature_importance.png", 0.7 * inch, 1.05 * inch, 5.8 * inch, 4.5 * inch)
    add_image(c, figures_dir / "shap_summary_beeswarm.png", 6.7 * inch, 1.05 * inch, 5.8 * inch, 4.5 * inch)
    c.showPage()

    slide_header(c, "Operational Recommendations", "Turn model output into bank actions")
    draw_bullets(
        c,
        [
            "Critical: temporary hold, near-real-time manual review, customer verification, step-up authentication.",
            "High: same-day review queue and device/IP/beneficiary investigation.",
            "AML branch: network escalation rather than single-transaction review only.",
            "Medium: enhanced monitoring; escalate if repeated within the next review window.",
            "Feedback loop: investigator decisions become labels for Precision@K, Recall@K and supervised model tuning.",
        ],
        0.8 * inch,
        5.55 * inch,
        17,
        92,
    )
    c.showPage()

    slide_header(c, "Live Demo", "Streamlit advisor for transaction/customer review")
    draw_bullets(
        c,
        [
            "Run `streamlit run src/demo_app.py`.",
            "Select a top-risk transaction or enter a CUSTOMER_NUMBER.",
            "The advisor returns risk band, root-cause branch, reason codes, recommended action and SHAP evidence.",
            "This is designed for a short judging demo or screen-recorded walkthrough.",
        ],
        0.8 * inch,
        5.55 * inch,
        18,
        88,
    )
    c.save()


def set_text(box, text: str, font_size: int = 18, color: RGBColor | None = None, bold: bool = False) -> None:
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def build_pptx(output_path: Path, metrics: dict, figures_dir: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    slides = [
        ("Fraud & Anomaly Detection", "Cause-first framework using real data, no synthetic labels.", []),
        (
            "Cause-First Hypothesis",
            "Three root-cause branches before modeling.",
            [
                "Account takeover / identity compromise",
                "Unauthorized transfer / capital outflow",
                "AML network / mule-account pattern",
            ],
        ),
        (
            "Framework",
            "Data -> baseline -> rules -> Isolation Forest -> SHAP/xAI -> review queue.",
            [
                "Behavioral baseline by customer",
                "60% root-cause rules + 40% anomaly score",
                "Human-readable reason codes and recommended actions",
            ],
        ),
        (
            "Key Metrics",
            f"{metrics['row_counts']['transactions_scored']:,} transactions scored; {metrics['review_queue']['high_or_critical_transactions']:,} High/Critical.",
            [],
        ),
    ]
    for title, subtitle, bullets in slides:
        slide = prs.slides.add_slide(blank)
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.16))
        band.fill.solid()
        band.fill.fore_color.rgb = RGBColor(47, 107, 143)
        band.line.fill.background()
        title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.55), Inches(12), Inches(0.55))
        set_text(title_box, title, 30, RGBColor(31, 41, 51), True)
        sub_box = slide.shapes.add_textbox(Inches(0.58), Inches(1.15), Inches(12), Inches(0.45))
        set_text(sub_box, subtitle, 16, RGBColor(82, 96, 109))
        if bullets:
            body = slide.shapes.add_textbox(Inches(0.75), Inches(2.0), Inches(11.7), Inches(4.2))
            tf = body.text_frame
            tf.clear()
            for idx, bullet in enumerate(bullets):
                p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                p.text = bullet
                p.font.name = "Arial"
                p.font.size = Pt(21)
                p.level = 0
    for title, image in [
        ("Root-Cause Review Queue", "root_cause_high_critical.png"),
        ("Temporal Stability", "monthly_stability_backtest.png"),
        ("SHAP Explainability", "shap_feature_importance.png"),
    ]:
        slide = prs.slides.add_slide(blank)
        title_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.4), Inches(12), Inches(0.5))
        set_text(title_box, title, 28, RGBColor(31, 41, 51), True)
        image_path = figures_dir / image
        if image_path.exists():
            slide.shapes.add_picture(str(image_path), Inches(0.9), Inches(1.2), width=Inches(11.5), height=Inches(5.7))
    prs.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final slide deck as PDF and PPTX.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads((args.outputs_dir / "model_metrics.json").read_text(encoding="utf-8"))
    root_cause = pd.read_csv(args.outputs_dir / "root_cause_summary.csv")
    build_pdf(args.report_dir / "final_slide_deck.pdf", metrics, root_cause, args.figures_dir)
    build_pptx(args.report_dir / "final_slide_deck.pptx", metrics, args.figures_dir)
    print(f"Wrote {(args.report_dir / 'final_slide_deck.pdf').resolve()}")
    print(f"Wrote {(args.report_dir / 'final_slide_deck.pptx').resolve()}")


if __name__ == "__main__":
    main()
