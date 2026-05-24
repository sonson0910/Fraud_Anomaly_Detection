# G'Contest 2026 - Real-Data Fraud & Anomaly Detection

This project solves Problem 1: Fraud & Anomaly Detection using the real contest files in `Processed_Data/`.

The current version no longer uses synthetic data or synthetic fraud labels. The real dataset has no confirmed fraud label, so the solution is built as a practical fraud-monitoring framework: it creates customer behavioral baselines, detects abnormal transactions with rules and unsupervised ML, and explains each high-risk case with human-readable reason codes.

## What This Project Does

- Reads the real seven-module data structure from the contest: customer, transaction, activity, deposit, lending, card, and data dictionary.
- Builds a cause-first framework before modeling:
  - account takeover / identity compromise,
  - unauthorized transfer / capital outflow,
  - AML network / mule-account pattern.
- Uses `ACTIVITY_NO` as an ordered digital journey signal, where larger values represent later actions.
- Scores every transaction with:
  - rule-based branch scores for explainability,
  - Isolation Forest for unsupervised anomaly detection,
  - a hybrid final risk score.
- Produces transaction-level and customer-level review queues.
- Outputs an xAI-style explanation: `top_reasons` and `recommended_action`.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the real data locally:

```text
Processed_Data/
  0.Data Guidline.xlsx
  Data_Customer.csv
  Data_Transaction.csv
  Data_Activity.csv
  Data_Deposit.csv
  Data_Lending.csv
  Data_Card.csv
```

The real data folder and large generated CSVs are intentionally ignored by Git.

## Run The Full Demo

```bash
python src/fraud_pipeline.py
python src/xai_shap_engine.py
python src/build_notebook.py
python src/build_slide_deck.py
python -m pytest -q
```

Main outputs:

- `outputs/transaction_risk_scores.csv`: full transaction review queue, local only.
- `outputs/customer_risk_summary.csv`: customer-level risk summary, local only.
- `outputs/top_review_queue.csv`: top 1,000 transactions for demo, local only.
- `outputs/root_cause_summary.csv`: aggregate cause summary.
- `outputs/model_metrics.json`: schema checks, scoring thresholds, and review-queue metrics.
- `outputs/monthly_stability.csv`: monthly backtest for temporal stability.
- `outputs/shap_feature_importance.csv`: SHAP feature importance from the surrogate xAI model.
- `outputs/shap_local_explanations.csv`: local SHAP explanations for top-risk cases.
- `outputs/figures/*.png`: report figures.
- `report/final_report_outline.md`: final report outline.
- `report/final_slide_deck.pdf`: slide-style report for judging/demo.
- `report/final_slide_deck.pptx`: editable slide deck.
- `notebooks/01_fraud_anomaly_detection.ipynb`: technical notebook.

## Run The Risk Advisor Demo

```bash
python src/customer_risk_advisor.py --top-critical 3
python src/customer_risk_advisor.py --customer-id <CUSTOMER_NUMBER>
python src/customer_risk_advisor.py --transaction-id <transaction_row_id>
```

The advisor returns the risk band, main root-cause branch, reason codes, and recommended action. It does not claim a transaction is confirmed fraud unless a human/investigator label is later added.

Run the Streamlit live demo:

```bash
streamlit run src/demo_app.py
```

The Streamlit app includes an overview dashboard, case-review tab, chatbot-style advisor, and SHAP/xAI tab.

## xAI And Stability

The primary explanations are reason codes tied to the cause-first framework. A separate SHAP engine trains a tree surrogate to explain the final hybrid risk score:

```bash
python src/xai_shap_engine.py
```

If the local machine has the XGBoost OpenMP runtime, the script can use XGBoost. On machines without `libomp`, it automatically falls back to a scikit-learn tree surrogate and still produces real SHAP values.

The data covers 2019 only, so stability is measured as a monthly/quarterly temporal backtest rather than a crisis-period validation.

## Why There Is No Precision/Recall Yet

The provided real data does not contain confirmed fraud labels. Creating fake labels would make the evaluation look stronger than it really is. This implementation therefore reports valid no-label evaluation outputs: schema quality, risk-band distribution, top-risk review queue, branch coverage, and explanation quality.

When reviewed cases become available, the same pipeline can be extended to measure Precision@K, Recall@K, PR-AUC, and to train a supervised model.
