from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RAW_FILES = {
    "customer": "Data_Customer.csv",
    "transaction": "Data_Transaction.csv",
    "activity": "Data_Activity.csv",
    "deposit": "Data_Deposit.csv",
    "lending": "Data_Lending.csv",
    "card": "Data_Card.csv",
    "truth": "synthetic_ground_truth.csv",
}

EXPECTED_COLUMNS = {
    "customer": [
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
    "transaction": [
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
    "activity": [
        "ACTIVITY_DATE",
        "DAY_OF_WEEK",
        "ACTIVITY_HOUR",
        "ACTIVITY_NO",
        "CUSTOMER_NUMBER",
        "ACTIVITY_NAME",
    ],
    "deposit": ["MONTH", "COUNT_CA_ACCT", "AVG_CA_BALANCE", "COUNT_TD_ACCT", "AVG_TD_BALANCE", "CUSTOMER_NUMBER"],
    "lending": ["MONTH", "COUNT_OF_LOAN", "AVG_LOAN_AMOUNT", "CUSTOMER_NUMBER", "OVERDUE_LENDING", "TERM_LENDING", "INTEREST_RATE"],
    "card": ["MONTH", "COUNT_CREDITCARD", "COUNT_DEBITCARD", "CUSTOMER_NUMBER", "OVERDUE_CREDIT", "LIMIT_AMT", "OUTSTANDING_BALANCE"],
}


def load_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {key: pd.read_csv(raw_dir / filename) for key, filename in RAW_FILES.items()}


def validate_schema(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    report: dict[str, object] = {}
    for key, expected in EXPECTED_COLUMNS.items():
        actual = list(data[key].columns)
        report[key] = {
            "rows": int(len(data[key])),
            "missing_columns": [col for col in expected if col not in actual],
            "extra_columns": [col for col in actual if col not in expected],
            "null_cells": int(data[key].isna().sum().sum()),
            "duplicate_rows": int(data[key].duplicated().sum()),
        }
    return report


def _safe_divide(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return a / pd.Series(b).replace(0, np.nan)


def build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trx = data["transaction"].copy()
    trx.insert(0, "transaction_row_id", [f"TRX{i:07d}" for i in range(1, len(trx) + 1)])
    trx["TRANS_DATE"] = pd.to_datetime(trx["TRANS_DATE"])
    trx["TRANS_MONTH"] = trx["TRANS_DATE"].dt.to_period("M").dt.to_timestamp()
    trx["time_period"] = np.select(
        [
            trx["TRANS_DATE"].dt.year <= 2019,
            trx["TRANS_DATE"].dt.year.between(2020, 2021),
            trx["TRANS_DATE"].dt.year.between(2022, 2023),
            trx["TRANS_DATE"].dt.year >= 2024,
        ],
        ["2019 baseline", "2020-2021 crisis", "2022-2023 recovery", "2024-2026 digital acceleration"],
        default="unknown",
    )
    trx["is_weekend"] = trx["TRANS_DATE"].dt.dayofweek.isin([5, 6]).astype(int)
    trx["is_night_hour"] = trx["TRANS_HOUR"].isin(list(range(0, 6)) + [22, 23]).astype(int)
    trx["is_transfer"] = (trx["TRXN_LV1"] == "Transfer").astype(int)
    trx["is_external_transfer"] = trx["TRXN_LV2"].isin(["External Transfer", "Fast Transfer"]).astype(int)
    trx["is_round_amount"] = (trx["TRANS_AMOUNT"] % 1_000_000 == 0).astype(int)

    customer = data["customer"].copy()
    customer["CLIENT_CREATE_DATE"] = pd.to_datetime(customer["CLIENT_CREATE_DATE"])
    customer["DATE_OF_BIRTH"] = pd.to_datetime(customer["DATE_OF_BIRTH"])
    customer["IB_REGISTER_DATE"] = pd.to_datetime(customer["IB_REGISTER_DATE"])
    customer["customer_age"] = ((pd.Timestamp("2026-05-31") - customer["DATE_OF_BIRTH"]).dt.days / 365.25).round(1)
    customer["ib_tenure_days"] = (pd.Timestamp("2026-05-31") - customer["IB_REGISTER_DATE"]).dt.days.clip(lower=0)

    customer_features = customer[
        [
            "CUSTOMER_NUMBER",
            "STAFF",
            "SMS",
            "VERIFY_METHOD",
            "OCCUPATION_GROUP",
            "customer_age",
            "ib_tenure_days",
        ]
    ].copy()
    customer_features["is_staff"] = (customer_features["STAFF"] == "Y").astype(int)
    customer_features["uses_sms"] = (customer_features["SMS"] == "Y").astype(int)
    customer_features["uses_strong_auth"] = customer_features["VERIFY_METHOD"].isin(["Smart OTP", "Biometric", "Token"]).astype(int)

    trx = trx.merge(customer_features.drop(columns=["STAFF", "SMS", "VERIFY_METHOD"]), on="CUSTOMER_NUMBER", how="left")

    cust_stats = trx.groupby("CUSTOMER_NUMBER").agg(
        customer_txn_count=("TRANS_AMOUNT", "size"),
        customer_amount_mean=("TRANS_AMOUNT", "mean"),
        customer_amount_std=("TRANS_AMOUNT", "std"),
        customer_amount_p95=("TRANS_AMOUNT", lambda s: s.quantile(0.95)),
        customer_night_rate=("is_night_hour", "mean"),
        customer_unique_devices=("Device_ID_Hash", "nunique"),
        customer_unique_ips=("IP_Address_Proxy", "nunique"),
    )
    trx = trx.merge(cust_stats, on="CUSTOMER_NUMBER", how="left")
    trx["customer_amount_std"] = trx["customer_amount_std"].fillna(trx["customer_amount_std"].median()).replace(0, trx["customer_amount_std"].median())
    trx["amount_zscore_customer"] = ((trx["TRANS_AMOUNT"] - trx["customer_amount_mean"]) / trx["customer_amount_std"]).clip(-10, 20)
    trx["amount_vs_customer_p95"] = _safe_divide(trx["TRANS_AMOUNT"], trx["customer_amount_p95"]).fillna(0).clip(0, 25)

    daily = trx.groupby(["CUSTOMER_NUMBER", "TRANS_DATE"]).agg(
        daily_txn_count=("TRANS_AMOUNT", "size"),
        daily_amount_sum=("TRANS_AMOUNT", "sum"),
    ).reset_index()
    daily_baseline = daily.groupby("CUSTOMER_NUMBER").agg(
        daily_txn_count_mean=("daily_txn_count", "mean"),
        daily_amount_sum_mean=("daily_amount_sum", "mean"),
    ).reset_index()
    trx = trx.merge(daily, on=["CUSTOMER_NUMBER", "TRANS_DATE"], how="left").merge(daily_baseline, on="CUSTOMER_NUMBER", how="left")
    trx["daily_txn_count_ratio"] = _safe_divide(trx["daily_txn_count"], trx["daily_txn_count_mean"]).fillna(1).clip(0, 30)
    trx["daily_amount_ratio"] = _safe_divide(trx["daily_amount_sum"], trx["daily_amount_sum_mean"]).fillna(1).clip(0, 40)

    device_customer_count = trx.groupby("Device_ID_Hash")["CUSTOMER_NUMBER"].nunique().rename("device_customer_count")
    ip_customer_count = trx.groupby("IP_Address_Proxy")["CUSTOMER_NUMBER"].nunique().rename("ip_customer_count")
    device_txn_count = trx.groupby("Device_ID_Hash")["TRANS_AMOUNT"].size().rename("device_txn_count")
    ip_txn_count = trx.groupby("IP_Address_Proxy")["TRANS_AMOUNT"].size().rename("ip_txn_count")
    trx = trx.merge(device_customer_count, on="Device_ID_Hash", how="left")
    trx = trx.merge(ip_customer_count, on="IP_Address_Proxy", how="left")
    trx = trx.merge(device_txn_count, on="Device_ID_Hash", how="left")
    trx = trx.merge(ip_txn_count, on="IP_Address_Proxy", how="left")

    first_device_seen = trx.groupby(["CUSTOMER_NUMBER", "Device_ID_Hash"])["TRANS_DATE"].transform("min")
    first_ip_seen = trx.groupby(["CUSTOMER_NUMBER", "IP_Address_Proxy"])["TRANS_DATE"].transform("min")
    trx["is_new_device_for_customer"] = (trx["TRANS_DATE"] == first_device_seen).astype(int)
    trx["is_new_ip_for_customer"] = (trx["TRANS_DATE"] == first_ip_seen).astype(int)

    activity = data["activity"].copy()
    activity["ACTIVITY_DATE"] = pd.to_datetime(activity["ACTIVITY_DATE"])
    activity_daily = activity.groupby(["CUSTOMER_NUMBER", "ACTIVITY_DATE"]).agg(
        daily_activity_count=("ACTIVITY_NO", "size"),
        sensitive_activity_count=("ACTIVITY_NAME", lambda s: s.isin(["Add Beneficiary", "Change Password", "OTP Request"]).sum()),
        night_activity_count=("ACTIVITY_HOUR", lambda s: s.isin(list(range(0, 6)) + [22, 23]).sum()),
        max_activity_no_same_day=("ACTIVITY_NO", "max"),
    ).reset_index().rename(columns={"ACTIVITY_DATE": "TRANS_DATE"})
    trx = trx.merge(activity_daily, on=["CUSTOMER_NUMBER", "TRANS_DATE"], how="left")
    for col in ["daily_activity_count", "sensitive_activity_count", "night_activity_count", "max_activity_no_same_day"]:
        trx[col] = trx[col].fillna(0)

    deposit = data["deposit"].copy()
    deposit["MONTH"] = pd.to_datetime(deposit["MONTH"])
    lending = data["lending"].copy()
    lending["MONTH"] = pd.to_datetime(lending["MONTH"])
    card = data["card"].copy()
    card["MONTH"] = pd.to_datetime(card["MONTH"])
    product = deposit.merge(lending, on=["CUSTOMER_NUMBER", "MONTH"], how="outer").merge(card, on=["CUSTOMER_NUMBER", "MONTH"], how="outer")
    trx = trx.merge(product, left_on=["CUSTOMER_NUMBER", "TRANS_MONTH"], right_on=["CUSTOMER_NUMBER", "MONTH"], how="left")
    for col in ["AVG_CA_BALANCE", "AVG_TD_BALANCE", "AVG_LOAN_AMOUNT", "LIMIT_AMT", "OUTSTANDING_BALANCE"]:
        trx[col] = trx[col].fillna(0)
    for col in ["COUNT_CA_ACCT", "COUNT_TD_ACCT", "COUNT_OF_LOAN", "OVERDUE_LENDING", "TERM_LENDING", "COUNT_CREDITCARD", "COUNT_DEBITCARD", "OVERDUE_CREDIT"]:
        trx[col] = trx[col].fillna(0)
    trx["credit_utilization"] = _safe_divide(trx["OUTSTANDING_BALANCE"], trx["LIMIT_AMT"]).fillna(0).clip(0, 5)
    trx["amount_vs_ca_balance"] = _safe_divide(trx["TRANS_AMOUNT"], trx["AVG_CA_BALANCE"]).fillna(10).clip(0, 50)

    return trx


def build_rule_score(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    rules = {
        "Giao dịch ngoài khung giờ thông thường": df["is_night_hour"].eq(1),
        "Số tiền cao bất thường so với lịch sử khách hàng": (df["amount_zscore_customer"] > 3.5) | (df["amount_vs_customer_p95"] > 1.8),
        "Tần suất giao dịch tăng đột biến trong ngày": df["daily_txn_count_ratio"] > 3.0,
        "Dòng tiền ra bất thường so với số dư CASA": df["amount_vs_ca_balance"] > 2.5,
        "Thiết bị mới của khách hàng": df["is_new_device_for_customer"].eq(1) & (df["TRANS_AMOUNT"] > df["customer_amount_p95"]),
        "IP mới của khách hàng": df["is_new_ip_for_customer"].eq(1) & (df["TRANS_AMOUNT"] > df["customer_amount_p95"]),
        "Thiết bị/IP dùng bởi nhiều khách hàng": (df["device_customer_count"] >= 4) | (df["ip_customer_count"] >= 4),
        "Hoạt động nhạy cảm cùng ngày giao dịch": df["sensitive_activity_count"] >= 3,
        "Hành vi số ở giai đoạn sau trước/cùng ngày giao dịch": df["max_activity_no_same_day"] >= 7,
        "Giao dịch chuyển khoản ngoài hệ thống rủi ro cao": df["is_external_transfer"].eq(1) & (df["TRANS_AMOUNT"] > df["customer_amount_p95"]),
        "Giao dịch số tròn lặp lại": df["is_round_amount"].eq(1) & (df["daily_txn_count"] >= 3),
    }
    weights = {
        "Giao dịch ngoài khung giờ thông thường": 8,
        "Số tiền cao bất thường so với lịch sử khách hàng": 18,
        "Tần suất giao dịch tăng đột biến trong ngày": 12,
        "Dòng tiền ra bất thường so với số dư CASA": 12,
        "Thiết bị mới của khách hàng": 12,
        "IP mới của khách hàng": 10,
        "Thiết bị/IP dùng bởi nhiều khách hàng": 13,
        "Hoạt động nhạy cảm cùng ngày giao dịch": 8,
        "Hành vi số ở giai đoạn sau trước/cùng ngày giao dịch": 8,
        "Giao dịch chuyển khoản ngoài hệ thống rủi ro cao": 12,
        "Giao dịch số tròn lặp lại": 7,
    }
    rule_score = np.zeros(len(df), dtype=float)
    reason_lists: list[list[str]] = [[] for _ in range(len(df))]
    for reason, mask in rules.items():
        values = mask.fillna(False).to_numpy()
        rule_score += values.astype(float) * weights[reason]
        indices = np.flatnonzero(values)
        for idx in indices:
            reason_lists[idx].append(reason)
    df["rule_score_0_100"] = np.clip(rule_score, 0, 100)
    df["rule_reasons"] = ["; ".join(reasons[:4]) if reasons else "Không có rule cảnh báo mạnh" for reasons in reason_lists]
    df["branch_identity_access_score"] = np.clip(
        25 * df["is_new_device_for_customer"].eq(1).astype(float)
        + 20 * df["is_new_ip_for_customer"].eq(1).astype(float)
        + 18 * (df["sensitive_activity_count"] >= 3).astype(float)
        + 12 * (df["max_activity_no_same_day"] >= 7).astype(float)
        + 10 * (df["night_activity_count"] > 0).astype(float),
        0,
        100,
    )
    df["branch_transaction_behavior_score"] = np.clip(
        25 * ((df["amount_zscore_customer"] > 3.5) | (df["amount_vs_customer_p95"] > 1.8)).astype(float)
        + 20 * (df["daily_txn_count_ratio"] > 3.0).astype(float)
        + 15 * (df["daily_amount_ratio"] > 3.0).astype(float)
        + 12 * df["is_night_hour"].eq(1).astype(float)
        + 15 * (df["amount_vs_ca_balance"] > 2.5).astype(float),
        0,
        100,
    )
    df["branch_network_aml_score"] = np.clip(
        24 * (df["device_customer_count"] >= 4).astype(float)
        + 24 * (df["ip_customer_count"] >= 4).astype(float)
        + 20 * (df["is_external_transfer"].eq(1) & (df["TRANS_AMOUNT"] > df["customer_amount_p95"])).astype(float)
        + 12 * (df["is_round_amount"].eq(1) & (df["daily_txn_count"] >= 3)).astype(float),
        0,
        100,
    )
    branch_cols = [
        "branch_identity_access_score",
        "branch_transaction_behavior_score",
        "branch_network_aml_score",
    ]
    branch_labels = {
        "branch_identity_access_score": "Account access / identity compromise",
        "branch_transaction_behavior_score": "Abnormal transaction behavior",
        "branch_network_aml_score": "Network / AML linkage",
    }
    max_branch = df[branch_cols].idxmax(axis=1).map(branch_labels)
    df["primary_cause_branch"] = np.where(df[branch_cols].max(axis=1) > 0, max_branch, "No strong root cause")
    return df


MODEL_FEATURES = [
    "TRANS_AMOUNT",
    "TRANS_HOUR",
    "TRANS_NO",
    "is_weekend",
    "is_night_hour",
    "is_transfer",
    "is_external_transfer",
    "is_round_amount",
    "customer_age",
    "ib_tenure_days",
    "is_staff",
    "uses_sms",
    "uses_strong_auth",
    "customer_txn_count",
    "customer_amount_mean",
    "customer_amount_p95",
    "customer_night_rate",
    "customer_unique_devices",
    "customer_unique_ips",
    "amount_zscore_customer",
    "amount_vs_customer_p95",
    "daily_txn_count",
    "daily_amount_sum",
    "daily_txn_count_ratio",
    "daily_amount_ratio",
    "device_customer_count",
    "ip_customer_count",
    "device_txn_count",
    "ip_txn_count",
    "is_new_device_for_customer",
    "is_new_ip_for_customer",
    "daily_activity_count",
    "sensitive_activity_count",
    "night_activity_count",
    "max_activity_no_same_day",
    "AVG_CA_BALANCE",
    "AVG_TD_BALANCE",
    "COUNT_OF_LOAN",
    "AVG_LOAN_AMOUNT",
    "OVERDUE_LENDING",
    "COUNT_CREDITCARD",
    "OVERDUE_CREDIT",
    "credit_utilization",
    "amount_vs_ca_balance",
]


def score_model(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    df = features.copy()
    X = df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=220,
        contamination=0.025,
        max_samples=12_000,
        random_state=20260524,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    raw = -model.decision_function(X_scaled)
    model_score = MinMaxScaler(feature_range=(0, 100)).fit_transform(raw.reshape(-1, 1)).ravel()
    df["model_score_0_100"] = model_score
    df["risk_score_0_100"] = np.clip(0.60 * df["model_score_0_100"] + 0.40 * df["rule_score_0_100"], 0, 100).round(2)
    df["risk_band"] = pd.cut(
        df["risk_score_0_100"],
        bins=[-0.01, 35, 60, 80, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    df["recommended_action"] = df["risk_band"].map(
        {
            "Low": "Monitor normally",
            "Medium": "Soft monitoring and customer-history comparison",
            "High": "Step-up authentication or near-real-time review",
            "Critical": "Block/hold transaction for manual review and AML escalation if repeated",
        }
    )

    feature_importance = pd.Series(0.0, index=MODEL_FEATURES)
    if "IS_SYNTHETIC_ANOMALY" in df.columns and df["IS_SYNTHETIC_ANOMALY"].nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            df["IS_SYNTHETIC_ANOMALY"],
            test_size=0.30,
            stratify=df["IS_SYNTHETIC_ANOMALY"],
            random_state=20260524,
        )
        rf = RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=30, random_state=20260524, n_jobs=-1)
        rf.fit(X_train, y_train)
        feature_importance = pd.Series(rf.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
        df.attrs["surrogate_feature_importance"] = feature_importance.to_dict()

    return df, feature_importance.to_dict()


def assign_top_reasons(scored: pd.DataFrame, feature_importance: dict[str, float]) -> pd.DataFrame:
    df = scored.copy()
    feature_reason_map = {
        "model_score_0_100": "Mô hình anomaly đánh giá tổng thể cao",
        "amount_zscore_customer": "Giá trị giao dịch lệch mạnh khỏi baseline khách hàng",
        "daily_txn_count_ratio": "Tần suất giao dịch trong ngày tăng bất thường",
        "device_customer_count": "Thiết bị có dấu hiệu liên quan nhiều khách hàng",
        "ip_customer_count": "IP/proxy có dấu hiệu liên quan nhiều khách hàng",
        "sensitive_activity_count": "Có nhiều hoạt động nhạy cảm trước/cùng ngày giao dịch",
        "amount_vs_ca_balance": "Số tiền giao dịch lớn so với số dư bình quân",
    }

    def row_reasons(row: pd.Series) -> str:
        reasons = []
        if row["rule_reasons"] != "Không có rule cảnh báo mạnh":
            reasons.extend(row["rule_reasons"].split("; "))
        if row["model_score_0_100"] >= 75:
            reasons.append(feature_reason_map["model_score_0_100"])
        if row["amount_zscore_customer"] > 3.5:
            reasons.append(feature_reason_map["amount_zscore_customer"])
        if row["daily_txn_count_ratio"] > 3:
            reasons.append(feature_reason_map["daily_txn_count_ratio"])
        if row["device_customer_count"] >= 4:
            reasons.append(feature_reason_map["device_customer_count"])
        if row["ip_customer_count"] >= 4:
            reasons.append(feature_reason_map["ip_customer_count"])
        if row["sensitive_activity_count"] >= 3:
            reasons.append(feature_reason_map["sensitive_activity_count"])
        if row["amount_vs_ca_balance"] > 2.5:
            reasons.append(feature_reason_map["amount_vs_ca_balance"])
        deduped = list(dict.fromkeys(reasons))
        return "; ".join(deduped[:5]) if deduped else "Rủi ro thấp, chưa có tín hiệu bất thường rõ"

    df["top_reasons"] = df.apply(row_reasons, axis=1)
    df.attrs["top_model_drivers"] = dict(sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:12])
    return df


def evaluate(scored: pd.DataFrame) -> dict[str, object]:
    y = scored["IS_SYNTHETIC_ANOMALY"].astype(int)
    scores = scored["risk_score_0_100"].astype(float)
    operating_threshold = 60
    y_pred = (scores >= operating_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()
    metrics: dict[str, object] = {
        "rows_scored": int(len(scored)),
        "synthetic_anomaly_count": int(y.sum()),
        "threshold_for_positive": operating_threshold,
        "precision_at_threshold": round(float(precision), 4),
        "recall_at_threshold": round(float(recall), 4),
        "f1_at_threshold": round(float(f1), 4),
        "confusion_matrix": cm,
        "average_precision_pr_auc": round(float(average_precision_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
    }
    for k in [100, 500, 1000, 2500]:
        top = scored.nlargest(k, "risk_score_0_100")
        metrics[f"precision_at_{k}"] = round(float(top["IS_SYNTHETIC_ANOMALY"].mean()), 4)
        metrics[f"recall_at_{k}"] = round(float(top["IS_SYNTHETIC_ANOMALY"].sum() / max(1, y.sum())), 4)
    metrics["risk_band_distribution"] = scored["risk_band"].value_counts().to_dict()
    metrics["top_anomaly_types_in_critical"] = (
        scored.loc[scored["risk_band"].eq("Critical"), "ANOMALY_TYPE"].value_counts().head(10).to_dict()
    )
    stability: dict[str, object] = {}
    for period, group in scored.groupby("time_period", sort=False):
        y_period = group["IS_SYNTHETIC_ANOMALY"].astype(int)
        top_k = max(25, int(len(group) * 0.01))
        top_period = group.nlargest(top_k, "risk_score_0_100")
        period_metrics: dict[str, object] = {
            "rows": int(len(group)),
            "anomaly_rate": round(float(y_period.mean()), 4),
            "avg_risk_score": round(float(group["risk_score_0_100"].mean()), 4),
            "high_or_critical_count": int(group["risk_band"].isin(["High", "Critical"]).sum()),
            "precision_at_top_1pct": round(float(top_period["IS_SYNTHETIC_ANOMALY"].mean()), 4),
        }
        if y_period.nunique() > 1:
            period_metrics["roc_auc"] = round(float(roc_auc_score(y_period, group["risk_score_0_100"])), 4)
            period_metrics["pr_auc"] = round(float(average_precision_score(y_period, group["risk_score_0_100"])), 4)
        stability[str(period)] = period_metrics
    metrics["stability_by_period"] = stability
    return metrics


def make_figures(scored: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.histplot(data=scored, x="risk_score_0_100", hue="IS_SYNTHETIC_ANOMALY", bins=40, stat="density", common_norm=False)
    plt.title("Risk score distribution by synthetic anomaly label")
    plt.xlabel("Risk score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_score_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    hour_rate = scored.groupby("TRANS_HOUR")["IS_SYNTHETIC_ANOMALY"].mean().reset_index()
    sns.barplot(data=hour_rate, x="TRANS_HOUR", y="IS_SYNTHETIC_ANOMALY", color="#2F6B8F")
    plt.title("Synthetic anomaly rate by transaction hour")
    plt.xlabel("Transaction hour")
    plt.ylabel("Anomaly rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "anomaly_rate_by_hour.png", dpi=180)
    plt.close()

    amount_sample = scored.sample(min(len(scored), 20_000), random_state=20260524).copy()
    amount_sample["log_amount"] = np.log1p(amount_sample["TRANS_AMOUNT"])
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=amount_sample, x="IS_SYNTHETIC_ANOMALY", y="log_amount")
    plt.title("Transaction amount distribution")
    plt.xlabel("Synthetic anomaly")
    plt.ylabel("log(1 + amount)")
    plt.tight_layout()
    plt.savefig(figures_dir / "amount_distribution_by_label.png", dpi=180)
    plt.close()

    band_counts = scored["risk_band"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=band_counts.index,
        y=band_counts.values,
        hue=band_counts.index,
        palette=["#6C9A8B", "#E6B655", "#D97757", "#9C3D54"],
        legend=False,
    )
    plt.title("Transactions by risk band")
    plt.xlabel("Risk band")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_band_counts.png", dpi=180)
    plt.close()

    period_summary = scored.groupby("time_period", sort=False).agg(
        avg_risk_score=("risk_score_0_100", "mean"),
        anomaly_rate=("IS_SYNTHETIC_ANOMALY", "mean"),
    ).reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=period_summary, x="time_period", y="avg_risk_score", color="#6C9A8B")
    plt.title("Average risk score by time period")
    plt.xlabel("Time period")
    plt.ylabel("Average risk score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_stability_by_period.png", dpi=180)
    plt.close()


def write_outputs(scored: pd.DataFrame, metrics: dict[str, object], feature_importance: dict[str, float], output_dir: Path, report_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "transaction_row_id",
        "CUSTOMER_NUMBER",
        "TRANS_DATE",
        "TRANS_HOUR",
        "TRXN_LV1",
        "TRXN_LV2",
        "TRANS_AMOUNT",
        "risk_score_0_100",
        "risk_band",
        "primary_cause_branch",
        "branch_identity_access_score",
        "branch_transaction_behavior_score",
        "branch_network_aml_score",
        "rule_score_0_100",
        "model_score_0_100",
        "top_reasons",
        "recommended_action",
        "IS_SYNTHETIC_ANOMALY",
        "ANOMALY_TYPE",
    ]
    scored[columns].sort_values("risk_score_0_100", ascending=False).to_csv(output_dir / "transaction_risk_scores.csv", index=False)

    customer = scored.groupby("CUSTOMER_NUMBER").agg(
        max_risk_score=("risk_score_0_100", "max"),
        avg_risk_score=("risk_score_0_100", "mean"),
        transaction_count=("transaction_row_id", "count"),
        high_or_critical_count=("risk_band", lambda s: s.isin(["High", "Critical"]).sum()),
        synthetic_anomaly_count=("IS_SYNTHETIC_ANOMALY", "sum"),
        total_amount=("TRANS_AMOUNT", "sum"),
    ).reset_index()
    customer["customer_risk_band"] = pd.cut(
        customer["max_risk_score"],
        bins=[-0.01, 35, 60, 80, 100],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)
    sample_reasons = (
        scored.sort_values("risk_score_0_100", ascending=False)
        .groupby("CUSTOMER_NUMBER")["top_reasons"]
        .first()
        .rename("main_reason")
    )
    sample_branches = (
        scored.sort_values("risk_score_0_100", ascending=False)
        .groupby("CUSTOMER_NUMBER")["primary_cause_branch"]
        .first()
        .rename("main_cause_branch")
    )
    customer = customer.merge(sample_reasons, on="CUSTOMER_NUMBER", how="left")
    customer = customer.merge(sample_branches, on="CUSTOMER_NUMBER", how="left")
    customer.sort_values("max_risk_score", ascending=False).to_csv(output_dir / "customer_risk_summary.csv", index=False)

    scored_for_cause = scored.assign(
        is_high_or_critical=scored["risk_band"].isin(["High", "Critical"]).astype(int),
        high_critical_anomaly=scored["risk_band"].isin(["High", "Critical"]).astype(int) * scored["IS_SYNTHETIC_ANOMALY"].astype(int),
    )
    root_cause = scored_for_cause.groupby("primary_cause_branch").agg(
        transaction_count=("transaction_row_id", "count"),
        high_or_critical_count=("is_high_or_critical", "sum"),
        high_critical_anomaly_count=("high_critical_anomaly", "sum"),
        synthetic_anomaly_count=("IS_SYNTHETIC_ANOMALY", "sum"),
        avg_risk_score=("risk_score_0_100", "mean"),
    ).reset_index().sort_values(["high_or_critical_count", "avg_risk_score"], ascending=False)
    root_cause["high_critical_precision_proxy"] = (
        root_cause["high_critical_anomaly_count"] / root_cause["high_or_critical_count"].replace(0, np.nan)
    ).fillna(0)
    root_cause.to_csv(output_dir / "root_cause_summary.csv", index=False)
    root_cause_display = root_cause.copy()
    root_cause_display["avg_risk_score"] = root_cause_display["avg_risk_score"].round(2)
    root_cause_display["high_critical_precision_proxy"] = root_cause_display["high_critical_precision_proxy"].round(4)

    stability_display = pd.DataFrame.from_dict(metrics["stability_by_period"], orient="index").reset_index(names="time_period")
    for col in ["anomaly_rate", "avg_risk_score", "precision_at_top_1pct", "roc_auc", "pr_auc"]:
        if col in stability_display.columns:
            stability_display[col] = stability_display[col].round(4)

    def markdown_table(df: pd.DataFrame) -> str:
        table = df.astype(str)
        header_line = "| " + " | ".join(table.columns) + " |"
        separator_line = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        body_lines = ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
        return "\n".join([header_line, separator_line, *body_lines])

    root_cause_markdown = markdown_table(root_cause_display)
    stability_markdown = markdown_table(stability_display)

    metrics_with_importance = dict(metrics)
    metrics_with_importance["top_model_drivers"] = dict(sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:12])
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics_with_importance, ensure_ascii=False, indent=2), encoding="utf-8")

    top_examples = scored.sort_values("risk_score_0_100", ascending=False).head(8)
    example_cols = ["transaction_row_id", "CUSTOMER_NUMBER", "risk_score_0_100", "risk_band", "primary_cause_branch", "ANOMALY_TYPE", "top_reasons"]
    example_table = top_examples[example_cols].astype(str)
    header = "| " + " | ".join(example_table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(example_table.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in example_table.to_numpy()]
    example_markdown = "\n".join([header, separator, *rows])
    report = f"""# G'Contest 2026 - Bài toán 1: Fraud & Anomaly Detection

## 1. Chọn nhánh bài toán và mục tiêu

Booklet gợi ý 3 nhánh chính: Fraud & Anomaly Detection, Next Best Financial Offer và Persona-Based Digital Personalization. Nhóm chọn nhánh 1, nhưng thiết kế feature vẫn tận dụng dữ liệu chân dung khách hàng, hành vi số và sản phẩm tài chính để có thể mở rộng sang 2 nhánh còn lại.

Đề án không bắt đầu từ model. Luồng trình bày là: phân tích nguyên nhân có thể gây rủi ro -> gom thành root-cause branches -> xây framework phát hiện -> dùng model và xAI để lượng hóa/giải thích.

## 2. Dữ liệu synthetic

- Khách hàng: {metrics['customer_count']:,}
- Giao dịch: {metrics['rows_scored']:,}
- Giao dịch anomaly được cài nhãn kiểm chứng: {metrics['synthetic_anomaly_count']:,}
- Tỷ lệ anomaly: {metrics['synthetic_anomaly_rate']:.2%}
- Khoảng thời gian: {metrics['date_range']['min']} đến {metrics['date_range']['max']}

Dữ liệu gồm 6 nhóm: thông tin khách hàng, giao dịch e-banking, hoạt động số, tiền gửi, tín dụng và thẻ. `CUSTOMER_NUMBER` là key để nối các bảng. `ACTIVITY_NO` được hiểu theo thứ tự hành vi: số nhỏ là bước sớm, số lớn là bước sau/nhạy cảm hơn như add beneficiary hoặc change password. Nhãn chỉ nằm ở file `synthetic_ground_truth.csv`, không trộn vào schema gốc.

## 3. Phân tích nguyên nhân trước khi xây model

Ba nhóm nguyên nhân chính:

1. Account access / identity compromise: thiết bị mới, IP mới, OTP request, add beneficiary, change password.
2. Abnormal transaction behavior: giao dịch ngoài giờ quen thuộc, amount vượt baseline, burst tần suất, dòng tiền ra lớn so với số dư.
3. Network / AML linkage: IP/device dùng bởi nhiều khách hàng, chuyển khoản ngoài hệ thống, giao dịch số tròn lặp lại.

Tóm tắt root-cause trên tập synthetic:

{root_cause_markdown}

## 4. Framework phát hiện bất thường

Framework dùng hai lớp:

1. Root-cause rule score: dễ giải thích, bám các nguyên nhân nghiệp vụ ở trên.
2. Isolation Forest: học cấu trúc hành vi tổng thể để bắt các điểm lệch đa chiều.
3. xAI layer: trả `primary_cause_branch`, reason codes, feature importance và khuyến nghị xử lý.

Risk score cuối cùng = 60% model score + 40% rule score.

## 5. Kết quả mô hình

- PR-AUC: {metrics['average_precision_pr_auc']}
- ROC-AUC: {metrics['roc_auc']}
- Precision@100: {metrics['precision_at_100']}
- Recall@1000: {metrics['recall_at_1000']}
- Precision tại ngưỡng High/Critical >= 60: {metrics['precision_at_threshold']}
- Recall tại ngưỡng High/Critical >= 60: {metrics['recall_at_threshold']}

Stability theo thời gian:

{stability_markdown}

## 6. Ví dụ explainability

{example_markdown}

## 7. Đề xuất triển khai

- Medium risk: theo dõi mềm và so sánh thêm với lịch sử khách hàng.
- High risk: yêu cầu step-up authentication, ưu tiên giao dịch chuyển khoản ngoài hệ thống.
- Critical risk: tạm giữ hoặc đưa vào hàng đợi manual review; nếu có pattern lặp theo IP/device/beneficiary thì chuyển AML escalation.
- Demo/AI assistant: có thể dùng `src/customer_risk_advisor.py` như live demo để nhập `CUSTOMER_NUMBER` hoặc `transaction_row_id`, sau đó trả risk band, nguyên nhân, bằng chứng và hành động đề xuất. Nếu có API LLM, phần này có thể chuyển thành chatbot giải thích cho risk officer.
- KPI nên theo dõi: Precision@K, số case review/ngày, false-positive rate theo phân khúc khách hàng, thời gian xử lý manual review.

## 8. Hạn chế

Kết quả hiện dựa trên synthetic data để minh họa. Khi có dữ liệu thật, cần hiệu chỉnh ngưỡng risk band, contamination rate, rule weight và kiểm định với nhãn fraud/chargeback/manual review thực tế.
"""
    (report_dir / "final_report_outline.md").write_text(report, encoding="utf-8")


def run_pipeline(raw_dir: Path, output_dir: Path, figures_dir: Path, report_dir: Path) -> dict[str, object]:
    data = load_data(raw_dir)
    schema_report = validate_schema(data)
    features = build_features(data)
    truth = data["truth"]
    scored = features.merge(truth, on=["transaction_row_id", "CUSTOMER_NUMBER"], how="left")
    scored["IS_SYNTHETIC_ANOMALY"] = scored["IS_SYNTHETIC_ANOMALY"].fillna(0).astype(int)
    scored["ANOMALY_TYPE"] = scored["ANOMALY_TYPE"].fillna("NORMAL")
    scored = build_rule_score(scored)
    scored, feature_importance = score_model(scored)
    scored = assign_top_reasons(scored, feature_importance)
    metrics = evaluate(scored)
    metrics["customer_count"] = int(data["customer"]["CUSTOMER_NUMBER"].nunique())
    metrics["synthetic_anomaly_rate"] = float(scored["IS_SYNTHETIC_ANOMALY"].mean())
    metrics["date_range"] = {
        "min": str(scored["TRANS_DATE"].min().date()),
        "max": str(scored["TRANS_DATE"].max().date()),
    }
    metrics["schema_validation"] = schema_report
    make_figures(scored, figures_dir)
    write_outputs(scored, metrics, feature_importance, output_dir, report_dir)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the G'Contest synthetic fraud detection pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(args.raw_dir, args.output_dir, args.figures_dir, args.report_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
