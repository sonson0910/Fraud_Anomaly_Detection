# G'Contest 2026 - Synthetic Fraud & Anomaly Detection

This workspace contains a submission-ready synthetic implementation for Problem 1: Fraud & Anomaly Detection.

The dataset is synthetic and generated from the official data dictionary columns in `/Users/sonson/Downloads/0.Data Guidline.xlsx`. It is meant to demonstrate the full analytical workflow when the real contest records are not available locally.

The storyline is cause-first: identify root causes, group them into three risk branches, then score transactions with a hybrid rule/model framework.

## Quick Start

Clone the repository and create a local environment:

```bash
git clone https://github.com/sonson0910/Fraud_Anomaly_Detection.git
cd Fraud_Anomaly_Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the synthetic source data, rerun the fraud pipeline, and rebuild the notebook:

```bash
python src/generate_synthetic_data.py
python src/fraud_pipeline.py
python src/build_notebook.py
```

The large CSV files are intentionally not committed. They can be recreated with the commands above:

- `data/raw/*.csv`
- `outputs/transaction_risk_scores.csv`
- `outputs/customer_risk_summary.csv`

Run the local risk-advisor demo:

```bash
python src/customer_risk_advisor.py --top-critical 3
python src/customer_risk_advisor.py --customer-id CUS000001
python src/customer_risk_advisor.py --transaction-id TRX0000001
```

## Main Artifacts

- `data/raw/`: generated synthetic source tables.
- `notebooks/01_fraud_anomaly_detection.ipynb`: final technical notebook.
- `outputs/transaction_risk_scores.csv`: transaction-level scoring.
- `outputs/customer_risk_summary.csv`: customer-level risk view.
- `outputs/root_cause_summary.csv`: cause-first summary for report narrative.
- `outputs/model_metrics.json`: evaluation metrics.
- `outputs/figures/`: report figures.
- `report/final_report_outline.md`: Vietnamese final report outline.

## Validate

After regenerating outputs, run:

```bash
python -m pytest -q
```
