from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def build_notebook(output_path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# G'Contest 2026 - Fraud & Anomaly Detection\n\n"
            "Notebook này dùng dữ liệu thật trong `Processed_Data/`. Dữ liệu không có nhãn fraud đã xác minh, "
            "vì vậy bài làm không tạo synthetic label. Mục tiêu là xây dựng framework phát hiện bất thường, "
            "xếp hạng giao dịch cần review và giải thích nguyên nhân theo nghiệp vụ ngân hàng."
        ),
        nbf.v4.new_markdown_cell("## 1. Setup"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "sys.path.append(str(PROJECT_ROOT / 'src'))\n\n"
            "from fraud_pipeline import PipelineConfig, run_pipeline, load_reference_tables, aggregate_activity, validate_schema\n\n"
            "RAW_DIR = PROJECT_ROOT / 'Processed_Data'\n"
            "OUTPUT_DIR = PROJECT_ROOT / 'outputs'\n"
            "sns.set_theme(style='whitegrid')"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Assignment Interpretation\n\n"
            "Theo assignment, fraud track cần kết hợp transaction metadata với digital footprint để:\n\n"
            "- xây behavioral baseline cho từng khách hàng,\n"
            "- phát hiện account takeover và unauthorized transfers,\n"
            "- mở rộng sang money laundering patterns nếu dữ liệu cho phép,\n"
            "- xuất được human-readable reason cho từng dự báo.\n\n"
            "Do không có confirmed fraud label, notebook coi đây là bài toán unsupervised/risk-ranking, không phải supervised fraud classification."
        ),
        nbf.v4.new_markdown_cell("## 3. Data Overview And Schema Check"),
        nbf.v4.new_code_cell(
            "tables = load_reference_tables(RAW_DIR)\n"
            "row_counts = {name: len(df) for name, df in tables.items()}\n"
            "row_counts"
        ),
        nbf.v4.new_code_cell(
            "# Activity table is large, so the production pipeline aggregates it by customer-date.\n"
            "# Run this cell if you want a fresh schema report before the full pipeline.\n"
            "activity_daily, activity_no_threshold = aggregate_activity(RAW_DIR, chunksize=1_000_000)\n"
            "schema_report = validate_schema(RAW_DIR, tables, activity_daily)\n"
            "pd.DataFrame(schema_report).T"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Cause-First Fraud Hypotheses\n\n"
            "BGK có xu hướng đánh giá cao việc phân tích nguyên nhân trước. Framework gom tín hiệu thành ba nhánh:\n\n"
            "1. **Account takeover / identity compromise**: thiết bị mới, IP mới, hoạt động đêm, người thụ hưởng mới, late-stage activity.\n"
            "2. **Unauthorized transfer / capital outflow**: chuyển khoản ra ngoài, số tiền lệch baseline, daily burst, dòng tiền ra lớn so với CASA.\n"
            "3. **AML network / mule-account pattern**: IP/device dùng chung, beneficiary nhận tiền từ nhiều khách hàng, giao dịch số tròn giá trị cao.\n\n"
            "`ACTIVITY_NO` được dùng như thứ tự hành động: số lớn hơn là bước sau hơn. Pipeline lấy top 10% `ACTIVITY_NO` làm late-stage activity dựa trên phân phối thật."
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame({\n"
            "    'root_cause_branch': [\n"
            "        'Account takeover / identity compromise',\n"
            "        'Unauthorized transfer / capital outflow',\n"
            "        'AML network / mule-account pattern',\n"
            "    ],\n"
            "    'signals': [\n"
            "        'New device/IP, night access, new beneficiary, late-stage activity',\n"
            "        'Outside-bank transfer, high amount vs customer baseline, daily burst, cash-out vs CASA',\n"
            "        'Shared device/IP/beneficiary, round high-value transfers, repeated external transfers',\n"
            "    ],\n"
            "    'business_action': [\n"
            "        'Step-up authentication and account verification',\n"
            "        'Manual review / temporary hold for high-risk transfer',\n"
            "        'AML escalation and network investigation',\n"
            "    ],\n"
            "})"
        ),
        nbf.v4.new_markdown_cell("## 5. Run Full Pipeline"),
        nbf.v4.new_code_cell(
            "config = PipelineConfig(\n"
            "    raw_dir=RAW_DIR,\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    figures_dir=OUTPUT_DIR / 'figures',\n"
            "    report_dir=PROJECT_ROOT / 'report',\n"
            ")\n"
            "metrics = run_pipeline(config)\n"
            "metrics['review_queue']"
        ),
        nbf.v4.new_markdown_cell("## 6. Risk Outputs"),
        nbf.v4.new_code_cell(
            "risk = pd.read_csv(OUTPUT_DIR / 'transaction_risk_scores.csv')\n"
            "customers = pd.read_csv(OUTPUT_DIR / 'customer_risk_summary.csv')\n"
            "root_cause = pd.read_csv(OUTPUT_DIR / 'root_cause_summary.csv')\n"
            "risk.head()"
        ),
        nbf.v4.new_code_cell(
            "risk['risk_band'].value_counts().reindex(['Low','Medium','High','Critical'])"
        ),
        nbf.v4.new_markdown_cell("## 7. Explainability Examples"),
        nbf.v4.new_code_cell(
            "risk[[\n"
            "    'transaction_row_id', 'CUSTOMER_NUMBER', 'TRANS_DATE', 'TRANS_HOUR', 'TRANS_LV1', 'TRANS_LV2',\n"
            "    'TRANS_AMOUNT', 'risk_score_0_100', 'risk_band', 'primary_cause_branch',\n"
            "    'top_reasons', 'recommended_action'\n"
            "]].head(10)"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Evaluation Without Fraud Labels\n\n"
            "Vì dữ liệu không có nhãn fraud, không báo precision/recall trên nhãn tự tạo. Evaluation hợp lệ gồm:\n\n"
            "- kiểm tra schema và data quality,\n"
            "- kiểm tra phân phối risk band để phù hợp capacity review,\n"
            "- kiểm tra các top-risk transaction có reason codes rõ ràng,\n"
            "- kiểm tra ba nhánh nguyên nhân có output riêng,\n"
            "- chuẩn bị feedback loop để khi investigator xác nhận case thì đo Precision@K/Recall@K thật."
        ),
        nbf.v4.new_code_cell(
            "with open(OUTPUT_DIR / 'model_metrics.json', encoding='utf-8') as f:\n"
            "    metrics = json.load(f)\n"
            "pd.Series(metrics['risk_band_distribution'])"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame(metrics['top_surrogate_features'].items(), columns=['feature', 'importance']).head(15)"
        ),
        nbf.v4.new_markdown_cell("## 9. Report Figures"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "for path in sorted((OUTPUT_DIR / 'figures').glob('*.png')):\n"
            "    print(path.name)\n"
            "    display(Image(filename=str(path)))"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. Live Demo\n\n"
            "Sau khi chạy pipeline, có thể demo bằng CLI:\n\n"
            "```bash\n"
            "python src/customer_risk_advisor.py --top-critical 3\n"
            "python src/customer_risk_advisor.py --customer-id <CUSTOMER_NUMBER>\n"
            "python src/customer_risk_advisor.py --transaction-id <transaction_row_id>\n"
            "```\n\n"
            "Demo trả về risk band, nguyên nhân chính, reason codes và recommended action bằng ngôn ngữ dễ hiểu cho risk officer."
        ),
    ]
    nb["cells"] = cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final real-data notebook artifact.")
    parser.add_argument("--output", type=Path, default=Path("notebooks/01_fraud_anomaly_detection.ipynb"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_notebook(args.output)
    print(f"Notebook written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
