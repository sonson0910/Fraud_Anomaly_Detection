from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_COLUMNS = {
    "Data_Customer.csv": [
        "CUSTOMER_NUMBER",
        "CLIENT_SEX",
        "CLIENT_CREATE_DATE",
        "DATE_OF_BIRTH",
        "STAFF",
        "IB_REGISTER_DATE",
        "EB_REGISTER_CHANNEL",
        "SMS",
        "VERIFY_METHOD",
        "OCCUPATION_GROUP",
        "EDUCATION_LEVEL",
        "MARITAL_STATUS",
    ],
    "Data_Transaction.csv": [
        "TRXN_LV1",
        "TRXN_LV2",
        "TRANS_DATE",
        "DAY_OF_WEEK",
        "TRANS_HOUR",
        "TRANS_NO",
        "TRANS_AMOUNT",
        "CUSTOMER_NUMBER",
        "IP_Address_Proxy",
        "Device_ID_Hash",
        "Device_OS",
        "Merchant_ID_Masked",
        "Beneficiary_CUSTOMER_NUMBER",
    ],
    "Data_Activity.csv": [
        "ACTIVITY_DATE",
        "DAY_OF_WEEK",
        "ACTIVITY_HOUR",
        "ACTIVITY_NO",
        "CUSTOMER_NUMBER",
        "ACTIVITY_NAME",
    ],
    "Data_Deposit.csv": ["MONTH", "COUNT_CA_ACCT", "AVG_CA_BALANCE", "COUNT_TD_ACCT", "AVG_TD_BALANCE", "CUSTOMER_NUMBER"],
    "Data_Lending.csv": ["MONTH", "COUNT_OF_LOAN", "AVG_LOAN_AMOUNT", "CUSTOMER_NUMBER", "OVERDUE_LENDING", "TERM_LENDING", "INTEREST_RATE"],
    "Data_Card.csv": ["MONTH", "COUNT_CREDITCARD", "COUNT_DEBITCARD", "CUSTOMER_NUMBER", "OVERDUE_CREDIT", "LIMIT_AMT", "OUTSTANDING_BALANCE"],
}


def test_raw_tables_match_dictionary() -> None:
    for filename, expected in EXPECTED_COLUMNS.items():
        df = pd.read_csv(ROOT / "data/raw" / filename, nrows=5)
        assert list(df.columns) == expected


def test_outputs_exist_and_have_risk_scores() -> None:
    risk = pd.read_csv(ROOT / "outputs/transaction_risk_scores.csv")
    customer = pd.read_csv(ROOT / "outputs/customer_risk_summary.csv")
    metrics = json.loads((ROOT / "outputs/model_metrics.json").read_text(encoding="utf-8"))
    root_cause = pd.read_csv(ROOT / "outputs/root_cause_summary.csv")
    assert len(risk) >= 140_000
    assert len(customer) == 3_000
    assert not root_cause.empty
    assert risk["risk_score_0_100"].between(0, 100).all()
    assert "primary_cause_branch" in risk.columns
    assert "stability_by_period" in metrics
    assert metrics["date_range"]["min"].startswith("2019")
    assert metrics["precision_at_100"] >= 0.80
    assert metrics["recall_at_1000"] >= 0.25


def test_injected_anomalies_rank_high() -> None:
    risk = pd.read_csv(ROOT / "outputs/transaction_risk_scores.csv")
    top_1000 = risk.head(1000)
    assert top_1000["IS_SYNTHETIC_ANOMALY"].mean() >= 0.25
    assert {"High", "Critical"}.intersection(set(risk["risk_band"].unique()))
