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
            "tạo weak label từ rule nghiệp vụ, huấn luyện prevention model, xếp hạng giao dịch cần can thiệp "
            "và giải thích nguyên nhân theo nghiệp vụ ngân hàng."
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
            "Do không có confirmed fraud label, notebook tạo weak label từ root-cause rules, rồi huấn luyện supervised prevention model để quyết định Allow / Monitor / Step-up / Block."
        ),
        nbf.v4.new_markdown_cell(
            "## 2.1 Final 4-Phase Flow\n\n"
            "**Phase 1 - Data Foundation & Feature Engineering**\n\n"
            "1. Data Cleaning: chuẩn hóa schema, xử lý beneficiary/merchant, aggregate activity log, ghép product snapshots.\n"
            "2. Four Baseline Metrics: Transactional, Financial, Environmental, Behavioral.\n"
            "3. Customer 360 Feature Extraction: rolling window 30/60/90 ngày và baseline một dòng cho mỗi khách hàng.\n\n"
            "**Phase 2 - EDA, Thresholding & Auto-Labeling**\n\n"
            "4. Baseline EDA: lift chart, heatmap, network exposure, Customer 360 risk map.\n"
            "5. IQR Thresholding: `Q3 + 1.5 * IQR`.\n"
            "6. Dynamic Rule Engine: Account Takeover, Unauthorized Transfer, AML/Mule Network.\n"
            "7. Risk-Scoring & Auto-Labeling: tạo `rule_fraud_label` weak supervision.\n\n"
            "**Phase 3 - Machine Learning & Hybrid Check**\n\n"
            "8. ML Training: RandomForest học weak label, theo dõi recall, precision và false-positive rate.\n"
            "9. Hybrid Matrix: Rule+ML = Block/Hold, Rule-only = Step-up/eKYC, ML-only = Watchlist, No-alert = Allow.\n\n"
            "**Phase 4 - Deployment, Dashboard & xAI**\n\n"
            "10. Business Dashboard: prevention impact, protected amount, Customer 360, case review.\n"
            "11. xAI: reason codes + SHAP surrogate giải thích final hybrid prevention score."
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
        nbf.v4.new_markdown_cell(
            "## 5. Mentor Feedback Incorporated\n\n"
            "- `Beneficiary_CUSTOMER_NUMBER = 0/0.0/NaN` không bị coi mặc định là missing data; pipeline tách nhóm này thành merchant/non-customer beneficiary và phân tích tiếp bằng `Merchant_ID_Masked` + loại giao dịch.\n"
            "- Nhóm không có customer beneficiary không được đưa vào mạng lưới customer-to-customer money mule.\n"
            "- Nếu merchant nội bộ/tín dụng không có customer beneficiary nhưng đi cùng tín hiệu bất thường, pipeline đưa vào reason code để review.\n"
            "- Overdue lending/credit được quy về nhóm rủi ro 1-5 để làm bối cảnh khách hàng, nhưng không dùng một mình để kết luận fraud.\n"
            "- Baseline hiện được đóng gói thành `customer_360_baseline.csv` với 4 nhóm Transactional, Financial, Environmental, Behavioral.\n"
            "- Threshold chính có thêm ngưỡng IQR cá nhân hóa `Q3 + 1.5*IQR` và rolling window 30/60/90 ngày.\n"
            "- Hành động cuối dùng hybrid decision matrix: Rule+ML = Block/Hold, Rule-only = Step-up/eKYC, ML-only = Watchlist, no-alert = Allow.\n"
            "- Vì title là giảm thiểu rủi ro, mục tiêu tối ưu ưu tiên recall/prevention coverage; false positive rate vẫn được báo để kiểm soát trải nghiệm khách hàng."
        ),
        nbf.v4.new_markdown_cell("## 6. Run Full Pipeline"),
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
        nbf.v4.new_markdown_cell("## 7. Risk Outputs"),
        nbf.v4.new_code_cell(
            "risk = pd.read_csv(OUTPUT_DIR / 'transaction_risk_scores.csv')\n"
            "customers = pd.read_csv(OUTPUT_DIR / 'customer_risk_summary.csv')\n"
            "customer_360 = pd.read_csv(OUTPUT_DIR / 'customer_360_baseline.csv')\n"
            "root_cause = pd.read_csv(OUTPUT_DIR / 'root_cause_summary.csv')\n"
            "insights = pd.read_csv(OUTPUT_DIR / 'insight_summary.csv')\n"
            "risk.head()"
        ),
        nbf.v4.new_code_cell(
            "risk['risk_band'].value_counts().reindex(['Low','Medium','High','Critical'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 8. Customer 360 Baseline\n\n"
            "Mỗi khách hàng là một dòng, mô tả 'DNA tài chính' hiện tại bằng 4 nhóm baseline: Transactional, Financial, Environmental và Behavioral. "
            "Rolling window 30/60/90 ngày giúp baseline phản ánh hành vi gần đây hơn thay vì đóng băng theo toàn bộ lịch sử."
        ),
        nbf.v4.new_code_cell("customer_360.head()"),
        nbf.v4.new_code_cell(
            "customer_360[[\n"
            "    'transactional_iqr_upper_amount', 'rolling_30d_txn_count', 'rolling_90d_amount_avg',\n"
            "    'financial_worst_credit_risk_group', 'environmental_trusted_device_count',\n"
            "    'behavioral_avg_daily_activity_count'\n"
            "]].describe().T"
        ),
        nbf.v4.new_markdown_cell("## 9. Monthly / Quarterly Stability Backtest"),
        nbf.v4.new_code_cell(
            "monthly = pd.read_csv(OUTPUT_DIR / 'monthly_stability.csv')\n"
            "quarterly = pd.read_csv(OUTPUT_DIR / 'quarterly_stability.csv')\n"
            "monthly"
        ),
        nbf.v4.new_code_cell(
            "fig, ax1 = plt.subplots(figsize=(10,4))\n"
            "ax1.plot(monthly['month'], monthly['avg_risk_score'], marker='o', label='Avg risk score')\n"
            "ax1.set_ylabel('Avg risk score')\n"
            "ax1.tick_params(axis='x', rotation=45)\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(monthly['month'], monthly['high_critical_rate']*100, marker='s', color='#C46243', label='High/Critical rate')\n"
            "ax2.set_ylabel('High/Critical rate (%)')\n"
            "plt.title('Monthly stability backtest on 2019 data')\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_markdown_cell(
            "## 10. Data-Driven EDA Insights\n\n"
            "Các insight này được sinh từ scored population sau khi pipeline đọc dữ liệu thật. "
            "Mục tiêu là chứng minh framework không chỉ vẽ biểu đồ đơn giản, mà tìm được mối liên hệ giữa baseline, root cause, hybrid decision và business action."
        ),
        nbf.v4.new_code_cell("insights"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "for filename in insights['linked_figure'].dropna().unique():\n"
            "    path = OUTPUT_DIR / 'figures' / filename\n"
            "    if path.exists():\n"
            "        print(filename)\n"
            "        display(Image(filename=str(path)))"
        ),
        nbf.v4.new_markdown_cell("## 11. Explainability Examples"),
        nbf.v4.new_code_cell(
            "risk[[\n"
            "    'transaction_row_id', 'CUSTOMER_NUMBER', 'TRANS_DATE', 'TRANS_HOUR', 'TRANS_LV1', 'TRANS_LV2',\n"
            "    'TRANS_AMOUNT', 'rule_score_0_100', 'model_fraud_probability', 'risk_score_0_100',\n"
            "    'risk_band', 'hybrid_decision', 'primary_cause_branch', 'top_reasons', 'recommended_action'\n"
            "]].head(10)"
        ),
        nbf.v4.new_markdown_cell(
            "## 12. Hybrid Decision Matrix And Risk Band Policy\n\n"
            "Ma trận lai giúp biến Rule + ML thành hành động thực tế. Đây là phần quan trọng để bài không chỉ dừng ở phát hiện, mà chuyển sang prevention operation."
        ),
        nbf.v4.new_code_cell("pd.Series(metrics['risk_band_policy'])"),
        nbf.v4.new_code_cell(
            "risk['hybrid_decision'].value_counts()"
        ),
        nbf.v4.new_code_cell(
            "pd.crosstab(risk['hybrid_decision'], risk['prevention_action'])"
        ),
        nbf.v4.new_markdown_cell("## 13. SHAP xAI Surrogate"),
        nbf.v4.new_code_cell(
            "# Run once after the main pipeline if SHAP files do not exist:\n"
            "# !python src/xai_shap_engine.py\n"
            "shap_importance = pd.read_csv(OUTPUT_DIR / 'shap_feature_importance.csv')\n"
            "shap_local = pd.read_csv(OUTPUT_DIR / 'shap_local_explanations.csv')\n"
            "shap_importance.head(15)"
        ),
        nbf.v4.new_code_cell("shap_local.head(10)"),
        nbf.v4.new_markdown_cell(
            "## 14. Evaluation With Weak Labels\n\n"
            "Vì dữ liệu không có confirmed fraud label, các metric dưới đây được đo theo `rule_fraud_label`. "
            "Đây là weak label sinh từ rule nghiệp vụ, không phải nhãn investigator xác nhận. Cách trình bày đúng là: "
            "framework hiện đo khả năng model học lại và vận hành hóa rule engine, đồng thời chuẩn bị feedback loop "
            "để thay weak label bằng confirmed label sau này.\n\n"
            "Theo feedback mentor, fraud track ưu tiên bắt được nhiều case rủi ro nhất. Vì vậy cần nhìn recall/prevention coverage trước, "
            "sau đó kiểm soát precision và false-positive rate để không làm phiền khách hàng thường."
        ),
        nbf.v4.new_code_cell(
            "with open(OUTPUT_DIR / 'model_metrics.json', encoding='utf-8') as f:\n"
            "    metrics = json.load(f)\n"
            "pd.Series(metrics['risk_band_distribution'])"
        ),
        nbf.v4.new_code_cell("pd.Series(metrics['supervised_model_metrics'])"),
        nbf.v4.new_code_cell(
            "pd.DataFrame(metrics['supervised_model_metrics']['validation_confusion_matrix_at_high_threshold'], index=[0])"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame(metrics['top_surrogate_features'].items(), columns=['feature', 'importance']).head(15)"
        ),
        nbf.v4.new_markdown_cell("## 15. Report Figures"),
        nbf.v4.new_code_cell(
            "for path in sorted((OUTPUT_DIR / 'figures').glob('*.png')):\n"
            "    print(path.name)\n"
            "    display(Image(filename=str(path)))"
        ),
        nbf.v4.new_markdown_cell(
            "## 16. Live Demo And Slide Deck\n\n"
            "Sau khi chạy pipeline, có thể demo bằng CLI hoặc Streamlit:\n\n"
            "```bash\n"
            "python src/customer_risk_advisor.py --top-critical 3\n"
            "python src/customer_risk_advisor.py --customer-id <CUSTOMER_NUMBER>\n"
            "python src/customer_risk_advisor.py --transaction-id <transaction_row_id>\n"
            "streamlit run src/demo_app.py\n"
            "```\n\n"
            "Tạo slide deck PDF/PPTX:\n\n"
            "```bash\n"
            "python src/build_slide_deck.py\n"
            "```\n\n"
            "Demo trả về risk band, nguyên nhân chính, reason codes, SHAP evidence và recommended action bằng ngôn ngữ dễ hiểu cho risk officer."
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
