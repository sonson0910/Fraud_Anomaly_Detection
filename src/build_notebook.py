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
            "Notebook này sinh và phân tích bộ dữ liệu synthetic theo đúng data dictionary của đề bài, "
            "phân tích nguyên nhân trước, sau đó xây dựng framework hybrid rule-based + Isolation Forest để phát hiện giao dịch bất thường. "
            "Dữ liệu synthetic chỉ dùng để minh họa quy trình end-to-end khi chưa có dữ liệu bản ghi thật."
        ),
        nbf.v4.new_markdown_cell("## 1. Setup"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import sys\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "sys.path.append(str(PROJECT_ROOT / 'src'))\n"
            "from generate_synthetic_data import write_outputs, GenerationConfig\n"
            "from fraud_pipeline import load_data, validate_schema, build_features, build_rule_score, score_model, assign_top_reasons, evaluate\n"
            "sns.set_theme(style='whitegrid')\n"
            "RAW_DIR = PROJECT_ROOT / 'data/raw'\n"
            "OUTPUT_DIR = PROJECT_ROOT / 'outputs'"
        ),
        nbf.v4.new_markdown_cell("## 2. Generate Synthetic Data"),
        nbf.v4.new_code_cell(
            "config = GenerationConfig(n_customers=3000, target_transactions=150000, target_activities=220000)\n"
            "write_outputs(config, RAW_DIR)\n"
            "pd.read_csv(RAW_DIR / 'synthetic_metadata.csv')"
        ),
        nbf.v4.new_markdown_cell("## 3. Load Data And Validate Schema"),
        nbf.v4.new_code_cell(
            "data = load_data(RAW_DIR)\n"
            "schema_report = validate_schema(data)\n"
            "pd.DataFrame(schema_report).T"
        ),
        nbf.v4.new_markdown_cell("## 4. Data Quality Snapshot"),
        nbf.v4.new_code_cell(
            "quality_rows = []\n"
            "for name, df in data.items():\n"
            "    quality_rows.append({\n"
            "        'table': name,\n"
            "        'rows': len(df),\n"
            "        'columns': len(df.columns),\n"
            "        'null_cells': int(df.isna().sum().sum()),\n"
            "        'duplicate_rows': int(df.duplicated().sum()),\n"
            "    })\n"
            "pd.DataFrame(quality_rows)"
        ),
        nbf.v4.new_markdown_cell("## 5. EDA"),
        nbf.v4.new_code_cell(
            "trx = data['transaction'].copy()\n"
            "truth = data['truth'].copy()\n"
            "trx.insert(0, 'transaction_row_id', [f'TRX{i:07d}' for i in range(1, len(trx)+1)])\n"
            "trx = trx.merge(truth, on=['transaction_row_id', 'CUSTOMER_NUMBER'], how='left')\n"
            "trx['IS_SYNTHETIC_ANOMALY'] = trx['IS_SYNTHETIC_ANOMALY'].fillna(0).astype(int)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
            "sns.histplot(np.log1p(trx['TRANS_AMOUNT']), bins=50, ax=axes[0])\n"
            "axes[0].set_title('Log transaction amount distribution')\n"
            "hour_rate = trx.groupby('TRANS_HOUR')['IS_SYNTHETIC_ANOMALY'].mean().reset_index()\n"
            "sns.barplot(data=hour_rate, x='TRANS_HOUR', y='IS_SYNTHETIC_ANOMALY', ax=axes[1], color='#2F6B8F')\n"
            "axes[1].set_title('Anomaly rate by hour')\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Root-Cause Hypothesis Before Modeling\n\n"
            "BGK đánh giá cao việc xác định nguyên nhân trước khi đưa model. Framework này gom tín hiệu vào 3 nhánh nguyên nhân:\n\n"
            "1. Account access / identity compromise: thiết bị mới, IP mới, OTP, add beneficiary, change password.\n"
            "2. Abnormal transaction behavior: amount lệch baseline, giao dịch đêm, burst tần suất, dòng tiền ra bất thường.\n"
            "3. Network / AML linkage: IP/device dùng chung nhiều khách hàng, chuyển khoản ngoài hệ thống, giao dịch số tròn lặp lại.\n\n"
            "`ACTIVITY_NO` được đọc như thứ tự hành vi: số nhỏ là bước sớm, số lớn là bước sau/nhạy cảm hơn."
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame({\n"
            "    'cause_branch': ['Account access / identity compromise', 'Abnormal transaction behavior', 'Network / AML linkage'],\n"
            "    'signals': [\n"
            "        'New device/IP, OTP request, add beneficiary, change password',\n"
            "        'Night transaction, amount above customer P95, daily burst, cash-out vs balance',\n"
            "        'Shared device/IP, external transfer, round-value repeated transfers',\n"
            "    ],\n"
            "    'business_risk': ['Account takeover', 'Unauthorized transfer', 'Money laundering proxy'],\n"
            "})"
        ),
        nbf.v4.new_markdown_cell("## 7. Feature Engineering"),
        nbf.v4.new_code_cell(
            "features = build_features(data)\n"
            "features.shape, features[['TRANS_AMOUNT','amount_zscore_customer','daily_txn_count_ratio','device_customer_count','ip_customer_count','max_activity_no_same_day']].describe().T"
        ),
        nbf.v4.new_markdown_cell("## 8. Rule-Based Cause Score"),
        nbf.v4.new_code_cell(
            "scored = features.merge(data['truth'], on=['transaction_row_id', 'CUSTOMER_NUMBER'], how='left')\n"
            "scored['IS_SYNTHETIC_ANOMALY'] = scored['IS_SYNTHETIC_ANOMALY'].fillna(0).astype(int)\n"
            "scored['ANOMALY_TYPE'] = scored['ANOMALY_TYPE'].fillna('NORMAL')\n"
            "scored = build_rule_score(scored)\n"
            "scored[['primary_cause_branch','branch_identity_access_score','branch_transaction_behavior_score','branch_network_aml_score','rule_reasons']].head()"
        ),
        nbf.v4.new_markdown_cell("## 9. Isolation Forest And Hybrid Score"),
        nbf.v4.new_code_cell(
            "scored, feature_importance = score_model(scored)\n"
            "scored = assign_top_reasons(scored, feature_importance)\n"
            "scored[['transaction_row_id','risk_score_0_100','risk_band','primary_cause_branch','top_reasons']].sort_values('risk_score_0_100', ascending=False).head(10)"
        ),
        nbf.v4.new_markdown_cell("## 10. Evaluation Against Synthetic Ground Truth"),
        nbf.v4.new_code_cell(
            "metrics = evaluate(scored)\n"
            "metrics"
        ),
        nbf.v4.new_markdown_cell("## 11. Stability Across Time"),
        nbf.v4.new_code_cell(
            "pd.DataFrame.from_dict(metrics['stability_by_period'], orient='index')"
        ),
        nbf.v4.new_markdown_cell("## 12. Explainability"),
        nbf.v4.new_code_cell(
            "pd.Series(feature_importance).sort_values(ascending=False).head(15).plot(kind='barh', figsize=(8, 6), title='Surrogate feature importance')\n"
            "plt.gca().invert_yaxis()\n"
            "plt.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "scored.sort_values('risk_score_0_100', ascending=False)[[\n"
            "    'transaction_row_id','CUSTOMER_NUMBER','TRANS_DATE','TRANS_HOUR','TRANS_AMOUNT',\n"
            "    'risk_score_0_100','risk_band','primary_cause_branch','ANOMALY_TYPE','top_reasons','recommended_action'\n"
            "]].head(12)"
        ),
        nbf.v4.new_markdown_cell(
            "## 13. Business Recommendations And Live Demo Idea\n\n"
            "- Low/Medium risk: tiếp tục theo dõi và so sánh với baseline khách hàng.\n"
            "- High risk: áp dụng step-up authentication hoặc near-real-time review.\n"
            "- Critical risk: giữ giao dịch để manual review; nếu lặp theo IP/device/beneficiary thì chuyển AML escalation.\n"
            "- Có thể demo bằng `src/customer_risk_advisor.py`: nhập `CUSTOMER_NUMBER` hoặc `transaction_row_id`, hệ thống trả dự báo, root cause và action.\n"
            "- KPI vận hành: Precision@K, recall@K trên case đã review, false-positive rate theo phân khúc, số case/ngày."
        ),
    ]
    nb["cells"] = cells
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final notebook artifact.")
    parser.add_argument("--output", type=Path, default=Path("notebooks/01_fraud_anomaly_detection.ipynb"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_notebook(args.output)
    print(f"Notebook written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
