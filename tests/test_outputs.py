from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Processed_Data"
OUTPUT_DIR = ROOT / "outputs"


REQUIRED_RAW_FILES = [
    "Data_Customer.csv",
    "Data_Transaction.csv",
    "Data_Activity.csv",
    "Data_Deposit.csv",
    "Data_Lending.csv",
    "Data_Card.csv",
    "0.Data Guidline.xlsx",
]


def test_real_input_files_are_present() -> None:
    missing = [filename for filename in REQUIRED_RAW_FILES if not (RAW_DIR / filename).exists()]
    assert not missing, f"Missing real input files: {missing}"
    assert not (ROOT / "data" / "raw" / "synthetic_ground_truth.csv").exists()


def test_outputs_exist_and_are_real_data_scoring_outputs() -> None:
    risk_path = OUTPUT_DIR / "transaction_risk_scores.csv"
    customer_path = OUTPUT_DIR / "customer_risk_summary.csv"
    customer_360_path = OUTPUT_DIR / "customer_360_baseline.csv"
    metrics_path = OUTPUT_DIR / "model_metrics.json"
    root_cause_path = OUTPUT_DIR / "root_cause_summary.csv"
    top_queue_path = OUTPUT_DIR / "top_review_queue.csv"
    monthly_path = OUTPUT_DIR / "monthly_stability.csv"
    quarterly_path = OUTPUT_DIR / "quarterly_stability.csv"
    shap_path = OUTPUT_DIR / "shap_feature_importance.csv"
    shap_local_path = OUTPUT_DIR / "shap_local_explanations.csv"
    for path in [risk_path, customer_path, customer_360_path, metrics_path, root_cause_path, top_queue_path, monthly_path, quarterly_path, shap_path, shap_local_path]:
        assert path.exists(), f"Missing output: {path}"

    risk = pd.read_csv(
        risk_path,
        usecols=[
            "risk_score_0_100",
            "risk_band",
            "primary_cause_branch",
            "top_reasons",
            "prevention_action",
            "hybrid_decision",
            "recommended_action",
            "rule_fraud_label",
            "rule_alert",
            "model_fraud_probability",
            "ml_alert",
        ],
    )
    customers = pd.read_csv(customer_path)
    customer_360 = pd.read_csv(customer_360_path)
    root_cause = pd.read_csv(root_cause_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert len(risk) >= 1_000_000
    assert len(customers) >= 50_000
    assert len(customer_360) >= 50_000
    assert not root_cause.empty
    assert metrics["ground_truth_available"] is False
    assert metrics["data_source"] == "Processed_Data real contest tables"
    assert risk["risk_score_0_100"].between(0, 100).all()
    assert {"Low", "Medium", "High", "Critical"}.issubset(set(risk["risk_band"].unique()))
    assert "IS_SYNTHETIC_ANOMALY" not in risk.columns
    assert "ANOMALY_TYPE" not in risk.columns
    assert "monthly_stability" in metrics
    assert "quarterly_stability" in metrics
    assert "supervised_model_metrics" in metrics
    assert "prevention_impact" in metrics
    assert "baseline_engineering" in metrics
    assert metrics["baseline_engineering"]["rolling_windows_days"] == [30, 60, 90]
    assert risk["model_fraud_probability"].between(0, 1).all()
    assert {"Allow", "Enhanced Monitoring", "Step-up Authentication", "Block/Hold"}.issubset(set(risk["prevention_action"].unique()))
    assert set(risk["hybrid_decision"].unique()).issubset(
        {"Rule+ML alert: Block/Hold", "Rule-only alert: Step-up/eKYC", "ML-only alert: Special watchlist", "No alert: Allow"}
    )
    assert {
        "transactional_iqr_upper_amount",
        "rolling_30d_txn_count",
        "rolling_90d_amount_avg",
        "financial_worst_credit_risk_group",
        "environmental_trusted_device_count",
        "behavioral_avg_daily_activity_count",
    }.issubset(set(customer_360.columns))


def test_explainability_fields_are_populated_for_review_queue() -> None:
    risk = pd.read_csv(OUTPUT_DIR / "transaction_risk_scores.csv", nrows=1000)
    assert risk["top_reasons"].notna().all()
    assert risk["recommended_action"].notna().all()
    assert risk["prevention_action"].notna().all()
    assert risk["primary_cause_branch"].notna().all()
    assert risk["top_reasons"].str.len().median() > 20


def test_no_synthetic_generator_remains() -> None:
    assert not (ROOT / "src" / "generate_synthetic_data.py").exists()


@pytest.mark.parametrize(
    "figure",
    [
        "risk_score_distribution.png",
        "risk_band_counts.png",
        "root_cause_high_critical.png",
        "amount_by_risk_band.png",
        "activity_transaction_linkage.png",
        "monthly_stability_backtest.png",
        "hybrid_decision_matrix.png",
        "customer_360_credit_risk_group.png",
        "rolling_window_baseline.png",
        "shap_feature_importance.png",
        "shap_summary_beeswarm.png",
    ],
)
def test_report_figures_exist(figure: str) -> None:
    path = OUTPUT_DIR / "figures" / figure
    assert path.exists()
    assert path.stat().st_size > 1000


def test_slide_deck_outputs_exist() -> None:
    for path in [ROOT / "report" / "final_slide_deck.pdf", ROOT / "report" / "final_slide_deck.pptx"]:
        assert path.exists()
        assert path.stat().st_size > 10_000


def test_report_and_verification_reflect_current_flow() -> None:
    report = (ROOT / "report" / "final_report_outline.md").read_text(encoding="utf-8")
    verification = (ROOT / "docs" / "FEEDBACK_VERIFICATION.md").read_text(encoding="utf-8")
    assert "Customer 360 baseline" in report
    assert "hybrid Rule + ML matrix" in report or "Hybrid Rule + ML" in report
    assert "Low = Allow, Medium = Enhanced Monitoring, High = Step-up Authentication, Critical = Block/Hold" not in report
    assert "Không dùng flow synthetic/Isolation Forest cũ" in verification
    assert "Rule-based weak label" in verification
