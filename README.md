# G'Contest 2026 - Real-Data Fraud & Anomaly Detection

This project solves Problem 1: Fraud & Anomaly Detection using the real contest files in `Processed_Data/`.

The current version no longer uses synthetic data or confirmed-fraud claims. The real dataset has no fraud label, so the solution uses root-cause banking rules to create weak labels, trains a supervised prevention model on those weak labels, and explains each Block/Hold or Step-up decision with human-readable reason codes.

## What This Project Does

- Reads the real seven-module data structure from the contest: customer, transaction, activity, deposit, lending, card, and data dictionary.
- Builds a cause-first framework before modeling:
  - account takeover / identity compromise,
  - unauthorized transfer / capital outflow,
  - AML network / mule-account pattern.
- Uses `ACTIVITY_NO` as an ordered digital journey signal, where larger values represent later actions.
- Scores every transaction with:
  - a Customer 360 baseline profile with transactional, financial, environmental, and behavioral features,
  - rolling-window features over 30/60/90 days,
  - personalized IQR thresholds using `Q3 + 1.5 * IQR`,
  - rule-based branch scores for weak-label creation,
  - a supervised prevention model trained from those weak labels,
  - a hybrid Rule + ML decision matrix,
  - prevention actions: Allow, Enhanced Monitoring, Step-up Authentication, Block/Hold.
- Produces transaction-level and customer-level prevention queues.
- Produces data-driven insight artifacts so the report is based on observed risk lift, not only simple count charts.
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

## One-Command Demo With Real Data

Clone the repo, put the real contest folder on the machine, then run one script:

```bash
git clone https://github.com/sonson0910/Fraud_Anomaly_Detection.git
cd Fraud_Anomaly_Detection
bash scripts/run_demo.sh /path/to/Processed_Data
```

If the data folder is already named `Processed_Data/` inside the repo, you can simply run:

```bash
bash scripts/run_demo.sh
```

The script creates `.venv`, installs dependencies, executes the exact Colab business notebook stored at `notebooks/colab_exact_business.ipynb`, writes regenerated outputs to `outputs/colab_exact_cleaned/`, and starts Streamlit at:

```text
http://localhost:8501
```

To force regeneration after replacing the data folder:

```bash
FORCE_REBUILD=1 bash scripts/run_demo.sh /path/to/Processed_Data
```

Raw contest data and regenerated heavy outputs are not committed to Git. They stay on the local machine or external Drive.

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
- `outputs/customer_360_baseline.csv`: Customer 360 baseline master table, local only.
- `outputs/top_review_queue.csv`: top 1,000 transactions for demo, local only.
- `outputs/root_cause_summary.csv`: aggregate cause summary.
- `outputs/insight_summary.csv`: data-driven insights with evidence and business meaning.
- `outputs/insight_summary.md`: readable insight summary for report writing.
- `outputs/model_metrics.json`: schema checks, weak-label model metrics, and prevention-impact metrics.
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

The advisor returns the risk band, prevention action, main root-cause branch, reason codes, and recommended action. It does not claim a transaction is confirmed fraud unless a human/investigator label is later added.

Run the Streamlit live demo:

```bash
streamlit run streamlit_app.py
```

The Streamlit app includes an overview dashboard, data-insights tab, Customer 360 tab, prevention-impact tab, case-review tab, chatbot-style advisor, and SHAP/xAI tab.

## Run The Colab-Based Demo Manually

The one-command script above is recommended. Manual equivalent:

```bash
python src/run_colab_exact.py   --notebook notebooks/colab_exact_business.ipynb   --raw-dir /path/to/Processed_Data   --cleaned-dir outputs/colab_exact_cleaned   --figures-dir outputs/colab_exact_figures
streamlit run streamlit_app.py
```

This executes the exact business cells from the Colab notebook source and only patches local paths, shell magics, and `plt.show()`. It creates `Customer_360_Master_Data.csv`, Colab rule flags, `Fraud` weak labels, `final_risk_score`, `Risk_Segment`, `Business_Action`, model metrics, feature importance, SHAP outputs, and the Streamlit dashboard. The generated output folder is intentionally ignored by Git because it is large.

## xAI And Stability

The primary explanations are reason codes tied to the cause-first framework. A separate SHAP engine trains a tree surrogate to explain the final hybrid prevention risk score after the Rule + ML decision matrix:

```bash
python src/xai_shap_engine.py
```

If the local machine has the XGBoost OpenMP runtime, the script can use XGBoost. On machines without `libomp`, it automatically falls back to a scikit-learn tree surrogate and still produces real SHAP values.

The data covers 2019 only, so stability is measured as a monthly/quarterly temporal backtest rather than a crisis-period validation.

## Weak Labels And Evaluation

The provided real data does not contain confirmed fraud labels. The project therefore creates rule-derived weak labels from the three root-cause branches and reports model performance against those weak labels. Dashboard prevention coverage means coverage against rule-derived labels, not confirmed real fraud outcomes.

When reviewed cases become available, the same pipeline can be recalibrated to measure real Precision@K, Recall@K, PR-AUC, false positive rate, and protected value.
