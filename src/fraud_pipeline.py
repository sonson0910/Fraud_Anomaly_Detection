from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


REAL_DATA_FILES = {
    "customer": "Data_Customer.csv",
    "transaction": "Data_Transaction.csv",
    "activity": "Data_Activity.csv",
    "deposit": "Data_Deposit.csv",
    "lending": "Data_Lending.csv",
    "card": "Data_Card.csv",
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
        "AGE",
        "OCCUPATION_GROUP",
        "EDUCATION_LEVEL",
        "MARITAL_STATUS",
    ],
    "transaction": [
        "TRANS_LV1",
        "TRANS_LV2",
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

MODEL_FEATURES = [
    "log_trans_amount",
    "TRANS_HOUR",
    "TRANS_NO",
    "is_weekend",
    "is_night_hour",
    "is_transfer",
    "is_external_transfer",
    "is_cashout_flow",
    "is_round_high_amount",
    "amount_zscore_customer",
    "amount_vs_customer_p95",
    "amount_global_percentile",
    "daily_txn_count",
    "daily_txn_count_ratio",
    "daily_amount_ratio",
    "daily_external_transfer_count",
    "daily_activity_count",
    "night_activity_count",
    "late_stage_activity_count",
    "max_activity_no_same_day",
    "device_customer_count",
    "ip_customer_count",
    "beneficiary_customer_count",
    "is_new_device_after_history",
    "is_new_ip_after_history",
    "is_new_beneficiary_after_history",
    "COUNT_CA_ACCT",
    "AVG_CA_BALANCE",
    "COUNT_TD_ACCT",
    "AVG_TD_BALANCE",
    "COUNT_OF_LOAN",
    "AVG_LOAN_AMOUNT",
    "OVERDUE_LENDING",
    "TERM_LENDING",
    "INTEREST_RATE",
    "COUNT_CREDITCARD",
    "COUNT_DEBITCARD",
    "OVERDUE_CREDIT",
    "LIMIT_AMT",
    "OUTSTANDING_BALANCE",
    "credit_utilization",
    "amount_vs_ca_balance",
    "customer_age",
    "ib_tenure_days",
    "uses_sms",
    "uses_strong_auth",
]


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path = Path("Processed_Data")
    output_dir: Path = Path("outputs")
    figures_dir: Path = Path("outputs/figures")
    report_dir: Path = Path("report")
    chunksize: int = 1_000_000
    random_state: int = 20260524
    iforest_fit_sample: int = 200_000
    iforest_max_samples: int = 50_000
    iforest_contamination: float = 0.02
    surrogate_sample: int = 120_000
    xai_sample: int = 80_000


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()


def read_csv_clean(path: Path, **kwargs: Any) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    return drop_unnamed_columns(df)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "TRXN_LV1": "TRANS_LV1",
        "TRXN_LV2": "TRANS_LV2",
        "Occupation_Group": "OCCUPATION_GROUP",
        "Education_Level": "EDUCATION_LEVEL",
        "Marital_Status": "MARITAL_STATUS",
        "LIMIT_AMT_CREDIT": "LIMIT_AMT",
        "OUTSTANDING_BAL_CREDIT": "OUTSTANDING_BALANCE",
    }
    return df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})


def load_reference_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REAL_DATA_FILES.items():
        if key == "activity":
            continue
        tables[key] = normalise_columns(read_csv_clean(raw_dir / filename))
    return tables


def validate_schema(raw_dir: Path, tables: dict[str, pd.DataFrame], activity_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key, expected in EXPECTED_COLUMNS.items():
        if key == "activity":
            activity_path = raw_dir / REAL_DATA_FILES[key]
            actual = normalise_columns(read_csv_clean(activity_path, nrows=5)).columns.tolist()
            rows = count_csv_rows(activity_path)
            null_cells = None
            duplicate_rows = None
        else:
            actual = tables[key].columns.tolist()
            rows = int(len(tables[key]))
            null_cells = int(tables[key].isna().sum().sum())
            duplicate_rows = int(tables[key].duplicated().sum())
        report[key] = {
            "rows": rows,
            "missing_columns": [col for col in expected if col not in actual],
            "extra_columns": [col for col in actual if col not in expected],
            "null_cells": null_cells,
            "duplicate_rows": duplicate_rows,
        }
    if activity_daily is not None:
        report["activity_daily_aggregated"] = {"rows": int(len(activity_daily)), "columns": activity_daily.columns.tolist()}
    return report


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as file:
        return max(sum(1 for _ in file) - 1, 0)


def weighted_quantile_from_counts(counts: dict[int, int], q: float) -> int:
    total = sum(counts.values())
    if total == 0:
        return 0
    target = total * q
    cumulative = 0
    for value in sorted(counts):
        cumulative += counts[value]
        if cumulative >= target:
            return int(value)
    return int(max(counts))


def estimate_activity_no_threshold(raw_dir: Path, chunksize: int) -> int:
    path = raw_dir / REAL_DATA_FILES["activity"]
    counts: dict[int, int] = {}
    for chunk in pd.read_csv(path, usecols=["CUSTOMER_NUMBER", "ACTIVITY_NO"], chunksize=chunksize):
        vc = pd.to_numeric(chunk["ACTIVITY_NO"], errors="coerce").dropna().astype(int).value_counts()
        for value, count in vc.items():
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return weighted_quantile_from_counts(counts, 0.90)


def aggregate_activity(raw_dir: Path, chunksize: int) -> tuple[pd.DataFrame, int]:
    threshold = estimate_activity_no_threshold(raw_dir, chunksize)
    path = raw_dir / REAL_DATA_FILES["activity"]
    aggregations: list[pd.DataFrame] = []
    usecols = ["ACTIVITY_DATE", "ACTIVITY_HOUR", "ACTIVITY_NO", "CUSTOMER_NUMBER", "ACTIVITY_NAME"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = normalise_columns(drop_unnamed_columns(chunk))
        chunk["ACTIVITY_DATE"] = parse_date(chunk["ACTIVITY_DATE"]).dt.normalize()
        chunk["ACTIVITY_HOUR"] = pd.to_numeric(chunk["ACTIVITY_HOUR"], errors="coerce").fillna(-1).astype(int)
        chunk["ACTIVITY_NO"] = pd.to_numeric(chunk["ACTIVITY_NO"], errors="coerce").fillna(0).astype(int)
        activity_name = chunk["ACTIVITY_NAME"].astype(str).str.upper()
        chunk["is_night_activity"] = chunk["ACTIVITY_HOUR"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
        chunk["is_late_stage_activity"] = (chunk["ACTIVITY_NO"] >= threshold).astype(int)
        chunk["is_auth_or_account_activity"] = activity_name.str.contains(
            "OTP|AUTH|LOGIN|PASSWORD|PASS|BENEFICIARY|LIMIT|PROFILE|PIN|DEVICE", regex=True, na=False
        ).astype(int)
        grouped = (
            chunk.groupby(["CUSTOMER_NUMBER", "ACTIVITY_DATE"], dropna=False)
            .agg(
                daily_activity_count=("ACTIVITY_NO", "size"),
                night_activity_count=("is_night_activity", "sum"),
                late_stage_activity_count=("is_late_stage_activity", "sum"),
                auth_or_account_activity_count=("is_auth_or_account_activity", "sum"),
                max_activity_no_same_day=("ACTIVITY_NO", "max"),
                unique_activity_name_count=("ACTIVITY_NAME", "nunique"),
            )
            .reset_index()
            .rename(columns={"ACTIVITY_DATE": "TRANS_DATE"})
        )
        aggregations.append(grouped)
    activity_daily = pd.concat(aggregations, ignore_index=True)
    activity_daily = (
        activity_daily.groupby(["CUSTOMER_NUMBER", "TRANS_DATE"], dropna=False)
        .agg(
            daily_activity_count=("daily_activity_count", "sum"),
            night_activity_count=("night_activity_count", "sum"),
            late_stage_activity_count=("late_stage_activity_count", "sum"),
            auth_or_account_activity_count=("auth_or_account_activity_count", "sum"),
            max_activity_no_same_day=("max_activity_no_same_day", "max"),
            unique_activity_name_count=("unique_activity_name_count", "max"),
        )
        .reset_index()
    )
    return activity_daily, threshold


def prepare_monthly_products(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    deposit = tables["deposit"].copy()
    lending = tables["lending"].copy()
    card = tables["card"].copy()
    for frame in [deposit, lending, card]:
        frame["MONTH"] = parse_date(frame["MONTH"])
        frame["MONTH_KEY"] = frame["MONTH"].dt.to_period("M").astype(str)

    product = deposit.merge(lending, on=["CUSTOMER_NUMBER", "MONTH_KEY"], how="outer", suffixes=("", "_LENDING"))
    product = product.merge(card, on=["CUSTOMER_NUMBER", "MONTH_KEY"], how="outer", suffixes=("", "_CARD"))
    product = product.loc[:, ~product.columns.str.match(r"MONTH(_LENDING|_CARD)?$")]

    numeric_cols = [col for col in product.columns if col not in ["CUSTOMER_NUMBER", "MONTH_KEY"]]
    for col in numeric_cols:
        product[col] = pd.to_numeric(product[col], errors="coerce")
    product = product.groupby(["CUSTOMER_NUMBER", "MONTH_KEY"], as_index=False)[numeric_cols].max()
    return product


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float, default: float = 0.0) -> pd.Series:
    if isinstance(denominator, pd.Series):
        denom = denominator.replace(0, np.nan)
    else:
        denom = np.nan if denominator == 0 else denominator
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def build_features(tables: dict[str, pd.DataFrame], activity_daily: pd.DataFrame) -> pd.DataFrame:
    trx = tables["transaction"].copy()
    trx.insert(0, "transaction_row_id", np.arange(1, len(trx) + 1, dtype=np.int64))
    trx["TRANS_DATE"] = parse_date(trx["TRANS_DATE"]).dt.normalize()
    trx["MONTH_KEY"] = trx["TRANS_DATE"].dt.to_period("M").astype(str)
    trx["TRANS_HOUR"] = pd.to_numeric(trx["TRANS_HOUR"], errors="coerce").fillna(-1).astype(int)
    trx["TRANS_NO"] = pd.to_numeric(trx["TRANS_NO"], errors="coerce").fillna(0).astype(int)
    trx["TRANS_AMOUNT"] = pd.to_numeric(trx["TRANS_AMOUNT"], errors="coerce").fillna(0.0).clip(lower=0)

    for col in ["TRANS_LV1", "TRANS_LV2", "DAY_OF_WEEK", "IP_Address_Proxy", "Device_ID_Hash", "Device_OS", "Merchant_ID_Masked"]:
        trx[col] = trx[col].fillna("UNKNOWN").astype(str)
    trx["Beneficiary_CUSTOMER_NUMBER"] = trx["Beneficiary_CUSTOMER_NUMBER"].fillna("NO_BENEFICIARY").astype(str)

    lv1 = trx["TRANS_LV1"].str.lower()
    lv2 = trx["TRANS_LV2"].str.lower()
    trx["log_trans_amount"] = np.log1p(trx["TRANS_AMOUNT"])
    trx["is_weekend"] = trx["DAY_OF_WEEK"].isin(["Sat", "Sun"]).astype(int)
    trx["is_night_hour"] = trx["TRANS_HOUR"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    trx["is_transfer"] = lv1.eq("transfer").astype(int)
    trx["is_external_transfer"] = (lv1.eq("transfer") & lv2.isin(["outside_bank", "external_transfer", "fast_transfer", "outside bank"])).astype(int)
    trx["is_cashout_flow"] = (lv2.isin(["outside_bank", "ewallet", "mobile", "vndirect"]) | lv1.eq("transfer")).astype(int)
    amount_p75 = trx["TRANS_AMOUNT"].quantile(0.75)
    trx["is_round_high_amount"] = ((trx["TRANS_AMOUNT"] >= amount_p75) & np.isclose(trx["TRANS_AMOUNT"] % 1_000_000, 0)).astype(int)
    trx["amount_global_percentile"] = trx["TRANS_AMOUNT"].rank(pct=True, method="average")

    customer = tables["customer"].copy()
    customer["DATE_OF_BIRTH"] = parse_date(customer["DATE_OF_BIRTH"])
    customer["IB_REGISTER_DATE"] = parse_date(customer["IB_REGISTER_DATE"])
    as_of_date = trx["TRANS_DATE"].max()
    customer["customer_age"] = ((as_of_date - customer["DATE_OF_BIRTH"]).dt.days / 365.25).clip(0, 110)
    customer["ib_tenure_days"] = (as_of_date - customer["IB_REGISTER_DATE"]).dt.days.clip(lower=0)
    customer["uses_sms"] = customer["SMS"].astype(str).str.upper().eq("Y").astype(int)
    verify = customer["VERIFY_METHOD"].astype(str).str.upper()
    customer["uses_strong_auth"] = verify.str.contains("SMART|OTP|TOKEN|BIOMETRIC", regex=True, na=False).astype(int)
    customer_cols = ["CUSTOMER_NUMBER", "customer_age", "ib_tenure_days", "uses_sms", "uses_strong_auth"]
    trx = trx.merge(customer[customer_cols], on="CUSTOMER_NUMBER", how="left")

    cust_stats = (
        trx.groupby("CUSTOMER_NUMBER")
        .agg(
            customer_txn_count=("TRANS_AMOUNT", "size"),
            customer_amount_mean=("TRANS_AMOUNT", "mean"),
            customer_amount_std=("TRANS_AMOUNT", "std"),
            customer_amount_p95=("TRANS_AMOUNT", lambda s: float(s.quantile(0.95))),
            customer_night_rate=("is_night_hour", "mean"),
            customer_first_txn_date=("TRANS_DATE", "min"),
            customer_unique_devices=("Device_ID_Hash", "nunique"),
            customer_unique_ips=("IP_Address_Proxy", "nunique"),
            customer_unique_beneficiaries=("Beneficiary_CUSTOMER_NUMBER", "nunique"),
        )
        .reset_index()
    )
    trx = trx.merge(cust_stats, on="CUSTOMER_NUMBER", how="left")
    amount_std_fallback = float(trx["customer_amount_std"].median(skipna=True) or 1.0)
    trx["customer_amount_std"] = trx["customer_amount_std"].replace(0, np.nan).fillna(amount_std_fallback)
    trx["customer_amount_p95"] = trx["customer_amount_p95"].replace(0, np.nan).fillna(trx["TRANS_AMOUNT"].quantile(0.95))
    trx["amount_zscore_customer"] = ((trx["TRANS_AMOUNT"] - trx["customer_amount_mean"]) / trx["customer_amount_std"]).clip(-10, 50)
    trx["amount_vs_customer_p95"] = _safe_divide(trx["TRANS_AMOUNT"], trx["customer_amount_p95"], default=0).clip(0, 50)

    daily = (
        trx.groupby(["CUSTOMER_NUMBER", "TRANS_DATE"], as_index=False)
        .agg(
            daily_txn_count=("TRANS_AMOUNT", "size"),
            daily_amount_sum=("TRANS_AMOUNT", "sum"),
            daily_external_transfer_count=("is_external_transfer", "sum"),
            daily_cashout_flow_count=("is_cashout_flow", "sum"),
        )
    )
    daily_baseline = (
        daily.groupby("CUSTOMER_NUMBER", as_index=False)
        .agg(
            daily_txn_count_mean=("daily_txn_count", "mean"),
            daily_amount_sum_mean=("daily_amount_sum", "mean"),
            daily_external_transfer_mean=("daily_external_transfer_count", "mean"),
        )
    )
    trx = trx.merge(daily, on=["CUSTOMER_NUMBER", "TRANS_DATE"], how="left")
    trx = trx.merge(daily_baseline, on="CUSTOMER_NUMBER", how="left")
    trx["daily_txn_count_ratio"] = _safe_divide(trx["daily_txn_count"], trx["daily_txn_count_mean"], default=1).clip(0, 50)
    trx["daily_amount_ratio"] = _safe_divide(trx["daily_amount_sum"], trx["daily_amount_sum_mean"], default=1).clip(0, 100)
    trx["daily_external_transfer_ratio"] = _safe_divide(
        trx["daily_external_transfer_count"], trx["daily_external_transfer_mean"], default=0
    ).clip(0, 50)

    device_customer_count = trx.groupby("Device_ID_Hash")["CUSTOMER_NUMBER"].nunique().rename("device_customer_count")
    ip_customer_count = trx.groupby("IP_Address_Proxy")["CUSTOMER_NUMBER"].nunique().rename("ip_customer_count")
    beneficiary_customer_count = trx.groupby("Beneficiary_CUSTOMER_NUMBER")["CUSTOMER_NUMBER"].nunique().rename("beneficiary_customer_count")
    trx = trx.merge(device_customer_count, on="Device_ID_Hash", how="left")
    trx = trx.merge(ip_customer_count, on="IP_Address_Proxy", how="left")
    trx = trx.merge(beneficiary_customer_count, on="Beneficiary_CUSTOMER_NUMBER", how="left")
    trx.loc[trx["Beneficiary_CUSTOMER_NUMBER"].eq("NO_BENEFICIARY"), "beneficiary_customer_count"] = 0

    first_device_seen = trx.groupby(["CUSTOMER_NUMBER", "Device_ID_Hash"])["TRANS_DATE"].transform("min")
    first_ip_seen = trx.groupby(["CUSTOMER_NUMBER", "IP_Address_Proxy"])["TRANS_DATE"].transform("min")
    first_beneficiary_seen = trx.groupby(["CUSTOMER_NUMBER", "Beneficiary_CUSTOMER_NUMBER"])["TRANS_DATE"].transform("min")
    has_history = trx["TRANS_DATE"].gt(trx["customer_first_txn_date"])
    trx["is_new_device_after_history"] = (has_history & trx["TRANS_DATE"].eq(first_device_seen)).astype(int)
    trx["is_new_ip_after_history"] = (has_history & trx["TRANS_DATE"].eq(first_ip_seen)).astype(int)
    trx["is_new_beneficiary_after_history"] = (
        has_history & trx["TRANS_DATE"].eq(first_beneficiary_seen) & ~trx["Beneficiary_CUSTOMER_NUMBER"].eq("NO_BENEFICIARY")
    ).astype(int)

    trx = trx.merge(activity_daily, on=["CUSTOMER_NUMBER", "TRANS_DATE"], how="left")
    activity_cols = [
        "daily_activity_count",
        "night_activity_count",
        "late_stage_activity_count",
        "auth_or_account_activity_count",
        "max_activity_no_same_day",
        "unique_activity_name_count",
    ]
    for col in activity_cols:
        trx[col] = pd.to_numeric(trx[col], errors="coerce").fillna(0)

    product = prepare_monthly_products(tables)
    trx = trx.merge(product, on=["CUSTOMER_NUMBER", "MONTH_KEY"], how="left")
    product_cols = [col for col in product.columns if col not in ["CUSTOMER_NUMBER", "MONTH_KEY"]]
    for col in product_cols:
        trx[col] = pd.to_numeric(trx[col], errors="coerce").fillna(0)
    for col in MODEL_FEATURES:
        if col not in trx.columns:
            trx[col] = 0

    trx["credit_utilization"] = _safe_divide(trx["OUTSTANDING_BALANCE"], trx["LIMIT_AMT"], default=0).clip(0, 5)
    trx["amount_vs_ca_balance"] = _safe_divide(trx["TRANS_AMOUNT"], trx["AVG_CA_BALANCE"], default=10).clip(0, 100)
    trx[MODEL_FEATURES] = trx[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return trx


def build_thresholds(df: pd.DataFrame) -> dict[str, float]:
    unique_device_counts = df.drop_duplicates("Device_ID_Hash")["device_customer_count"]
    unique_ip_counts = df.drop_duplicates("IP_Address_Proxy")["ip_customer_count"]
    unique_beneficiary_counts = df.drop_duplicates("Beneficiary_CUSTOMER_NUMBER")["beneficiary_customer_count"]
    return {
        "high_amount_zscore": 3.0,
        "high_amount_vs_p95": 1.5,
        "top_global_amount_percentile": 0.995,
        "daily_burst_ratio": 3.0,
        "daily_amount_burst_ratio": 4.0,
        "cashout_vs_balance_ratio": 2.0,
        "shared_device_customer_count": float(max(3, unique_device_counts.quantile(0.99))),
        "shared_ip_customer_count": float(max(3, unique_ip_counts.quantile(0.99))),
        "shared_beneficiary_customer_count": float(max(3, unique_beneficiary_counts.quantile(0.99))),
    }


def build_rule_score(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    result = df.copy()
    high_amount = (
        result["amount_zscore_customer"].ge(thresholds["high_amount_zscore"])
        | result["amount_vs_customer_p95"].ge(thresholds["high_amount_vs_p95"])
        | result["amount_global_percentile"].ge(thresholds["top_global_amount_percentile"])
    )
    daily_burst = result["daily_txn_count_ratio"].ge(thresholds["daily_burst_ratio"])
    daily_amount_burst = result["daily_amount_ratio"].ge(thresholds["daily_amount_burst_ratio"])
    cashout_vs_balance = result["amount_vs_ca_balance"].ge(thresholds["cashout_vs_balance_ratio"]) & result["is_cashout_flow"].eq(1)
    shared_device = result["device_customer_count"].ge(thresholds["shared_device_customer_count"])
    shared_ip = result["ip_customer_count"].ge(thresholds["shared_ip_customer_count"])
    shared_beneficiary = result["beneficiary_customer_count"].ge(thresholds["shared_beneficiary_customer_count"])
    late_activity = result["late_stage_activity_count"].gt(0) | result["auth_or_account_activity_count"].gt(0)
    nighttime_access = result["is_night_hour"].eq(1) | result["night_activity_count"].gt(0)

    result["branch_account_takeover_score"] = np.clip(
        22 * result["is_new_device_after_history"]
        + 20 * result["is_new_ip_after_history"]
        + 18 * (result["is_new_beneficiary_after_history"].eq(1) & result["is_external_transfer"].eq(1)).astype(int)
        + 14 * nighttime_access.astype(int)
        + 14 * late_activity.astype(int)
        + 12 * high_amount.astype(int)
        + 8 * result["uses_sms"].eq(1).astype(int),
        0,
        100,
    )
    result["branch_unauthorized_transfer_score"] = np.clip(
        22 * result["is_external_transfer"]
        + 20 * high_amount.astype(int)
        + 16 * result["is_new_beneficiary_after_history"]
        + 16 * daily_burst.astype(int)
        + 12 * daily_amount_burst.astype(int)
        + 10 * cashout_vs_balance.astype(int)
        + 8 * nighttime_access.astype(int),
        0,
        100,
    )
    result["branch_aml_network_score"] = np.clip(
        20 * shared_device.astype(int)
        + 20 * shared_ip.astype(int)
        + 20 * shared_beneficiary.astype(int)
        + 14 * result["is_round_high_amount"]
        + 14 * result["is_external_transfer"]
        + 8 * result["daily_external_transfer_ratio"].ge(3).astype(int)
        + 8 * high_amount.astype(int),
        0,
        100,
    )

    branch_cols = ["branch_account_takeover_score", "branch_unauthorized_transfer_score", "branch_aml_network_score"]
    result["max_branch_score"] = result[branch_cols].max(axis=1)
    result["active_branch_count"] = result[branch_cols].ge(35).sum(axis=1)
    result["rule_score_0_100"] = np.clip(result["max_branch_score"] + 7 * result["active_branch_count"].clip(0, 2), 0, 100)

    branch_names = np.array(
        [
            "Account takeover / identity compromise",
            "Unauthorized transfer / capital outflow",
            "AML network / mule-account pattern",
        ],
        dtype=object,
    )
    result["primary_cause_branch"] = branch_names[result[branch_cols].to_numpy().argmax(axis=1)]

    reason_masks: list[tuple[str, pd.Series]] = [
        ("Thiết bị mới sau khi khách hàng đã có lịch sử giao dịch", result["is_new_device_after_history"].eq(1)),
        ("IP mới sau khi khách hàng đã có lịch sử giao dịch", result["is_new_ip_after_history"].eq(1)),
        ("Người thụ hưởng mới trong giao dịch chuyển tiền", result["is_new_beneficiary_after_history"].eq(1)),
        ("Giao dịch ngoài khung giờ thông thường hoặc có hoạt động đêm", nighttime_access),
        ("Số tiền cao bất thường so với baseline của khách hàng", high_amount),
        ("Tần suất giao dịch trong ngày tăng đột biến", daily_burst),
        ("Tổng dòng tiền ngày đó tăng mạnh so với baseline", daily_amount_burst),
        ("Dòng tiền ra lớn so với số dư CASA", cashout_vs_balance),
        ("Thiết bị dùng chung bởi nhiều khách hàng", shared_device),
        ("IP dùng chung bởi nhiều khách hàng", shared_ip),
        ("Người thụ hưởng nhận tiền từ nhiều khách hàng", shared_beneficiary),
        ("Giao dịch chuyển khoản ra ngoài ngân hàng", result["is_external_transfer"].eq(1)),
        ("Giao dịch số tròn giá trị cao", result["is_round_high_amount"].eq(1)),
        ("Hoạt động digital ở giai đoạn muộn trong cùng ngày", late_activity),
    ]
    reasons = np.full(len(result), "", dtype=object)
    for reason, mask in reason_masks:
        values = mask.fillna(False).to_numpy()
        if not values.any():
            continue
        existing = reasons[values]
        reasons[values] = np.where(existing == "", reason, existing + "; " + reason)
    result["rule_reasons"] = np.where(reasons == "", "Không có rule đơn lẻ vượt ngưỡng mạnh", reasons)
    result["review_label_from_rules"] = result["rule_score_0_100"].ge(60).astype(int)
    return result


def score_isolation_forest(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, IsolationForest]:
    result = df.copy()
    feature_matrix = result[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    rng = np.random.default_rng(config.random_state)
    sample_size = min(config.iforest_fit_sample, len(feature_matrix))
    sample_idx = rng.choice(len(feature_matrix), size=sample_size, replace=False)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(5, 95))
    x_sample = scaler.fit_transform(imputer.fit_transform(feature_matrix.iloc[sample_idx]))
    x_all = scaler.transform(imputer.transform(feature_matrix))

    model = IsolationForest(
        n_estimators=220,
        max_samples=min(config.iforest_max_samples, sample_size),
        contamination=config.iforest_contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_sample)
    raw_scores = -model.score_samples(x_all)
    lo, hi = np.quantile(raw_scores, [0.01, 0.995])
    if hi <= lo:
        model_score = np.zeros_like(raw_scores)
    else:
        model_score = np.clip((raw_scores - lo) / (hi - lo) * 100, 0, 100)
    result["model_anomaly_score_0_100"] = model_score
    result["risk_score_0_100"] = np.clip(0.60 * result["rule_score_0_100"] + 0.40 * result["model_anomaly_score_0_100"], 0, 100)
    return result, model


def assign_risk_bands(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    result = df.copy()
    thresholds = {
        "medium_min_score": float(result["risk_score_0_100"].quantile(0.90)),
        "high_min_score": float(result["risk_score_0_100"].quantile(0.975)),
        "critical_min_score": float(result["risk_score_0_100"].quantile(0.995)),
    }
    result["risk_band"] = np.select(
        [
            result["risk_score_0_100"].ge(thresholds["critical_min_score"]),
            result["risk_score_0_100"].ge(thresholds["high_min_score"]),
            result["risk_score_0_100"].ge(thresholds["medium_min_score"]),
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    return result, thresholds


def build_surrogate_importance(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    sample_size = min(config.surrogate_sample, len(df))
    sample = df.sample(sample_size, random_state=config.random_state)
    x = sample[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = sample["risk_score_0_100"]
    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=50,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x, y)
    return pd.Series(model.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)


def recommended_action(row: pd.Series) -> str:
    band = row["risk_band"]
    branch = row["primary_cause_branch"]
    if band == "Critical" and "AML" in branch:
        return "Đưa vào hàng đợi AML escalation, kiểm tra mạng lưới IP/device/beneficiary trước khi xử lý tiếp."
    if band == "Critical":
        return "Tạm giữ để manual review gần real-time, gọi xác minh khách hàng và áp dụng step-up authentication."
    if band == "High" and "Account takeover" in branch:
        return "Yêu cầu xác thực tăng cường, kiểm tra thiết bị/IP mới và lịch sử đổi người thụ hưởng."
    if band == "High":
        return "Đẩy vào manual review queue theo SLA trong ngày và theo dõi giao dịch tiếp theo của khách hàng."
    if band == "Medium":
        return "Theo dõi tăng cường; nếu lặp lại trong 7 ngày thì nâng lên review thủ công."
    return "Không cần can thiệp ngay; tiếp tục cập nhật baseline hành vi khách hàng."


def finalise_explanations(df: pd.DataFrame, feature_importance: pd.Series) -> pd.DataFrame:
    result = df.copy()
    model_hint = "Model bất thường cao theo các biến: " + ", ".join(feature_importance.head(3).index.tolist())
    high_model = result["model_anomaly_score_0_100"].ge(result["model_anomaly_score_0_100"].quantile(0.975))
    result["top_reasons"] = result["rule_reasons"]
    result.loc[high_model, "top_reasons"] = result.loc[high_model, "top_reasons"] + "; " + model_hint
    result["recommended_action"] = result.apply(recommended_action, axis=1)
    return result


def build_customer_summary(scores: pd.DataFrame) -> pd.DataFrame:
    ordered = scores.sort_values(["CUSTOMER_NUMBER", "risk_score_0_100"], ascending=[True, False])
    top_by_customer = ordered.groupby("CUSTOMER_NUMBER", as_index=False).head(1)[
        ["CUSTOMER_NUMBER", "transaction_row_id", "primary_cause_branch", "top_reasons", "recommended_action"]
    ].rename(
        columns={
            "transaction_row_id": "top_risk_transaction_row_id",
            "primary_cause_branch": "main_cause_branch",
            "top_reasons": "main_reason",
            "recommended_action": "customer_recommended_action",
        }
    )
    summary = (
        scores.groupby("CUSTOMER_NUMBER", as_index=False)
        .agg(
            transaction_count=("transaction_row_id", "size"),
            max_risk_score=("risk_score_0_100", "max"),
            avg_risk_score=("risk_score_0_100", "mean"),
            high_or_critical_count=("risk_band", lambda s: int(s.isin(["High", "Critical"]).sum())),
            critical_count=("risk_band", lambda s: int(s.eq("Critical").sum())),
            total_amount=("TRANS_AMOUNT", "sum"),
            max_amount=("TRANS_AMOUNT", "max"),
        )
        .merge(top_by_customer, on="CUSTOMER_NUMBER", how="left")
    )
    summary["customer_risk_band"] = np.select(
        [summary["critical_count"].gt(0), summary["high_or_critical_count"].gt(0), summary["max_risk_score"].ge(50)],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    summary = summary.sort_values(["critical_count", "high_or_critical_count", "max_risk_score"], ascending=False)
    return summary


def build_root_cause_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("primary_cause_branch", as_index=False)
        .agg(
            total_transactions=("transaction_row_id", "size"),
            medium_or_above_transactions=("risk_band", lambda s: int(s.isin(["Medium", "High", "Critical"]).sum())),
            high_or_critical_transactions=("risk_band", lambda s: int(s.isin(["High", "Critical"]).sum())),
            critical_transactions=("risk_band", lambda s: int(s.eq("Critical").sum())),
            avg_risk_score=("risk_score_0_100", "mean"),
            median_amount=("TRANS_AMOUNT", "median"),
            p95_amount=("TRANS_AMOUNT", lambda s: float(s.quantile(0.95))),
        )
        .sort_values("high_or_critical_transactions", ascending=False)
    )


def build_time_stability(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = scores.copy()
    stability["month"] = pd.to_datetime(stability["TRANS_DATE"]).dt.to_period("M").astype(str)
    stability["quarter"] = pd.to_datetime(stability["TRANS_DATE"]).dt.to_period("Q").astype(str)
    stability["is_medium_plus"] = stability["risk_band"].isin(["Medium", "High", "Critical"]).astype(int)
    stability["is_high_critical"] = stability["risk_band"].isin(["High", "Critical"]).astype(int)
    monthly = (
        stability.groupby("month", as_index=False)
        .agg(
            transaction_count=("transaction_row_id", "size"),
            avg_risk_score=("risk_score_0_100", "mean"),
            p95_risk_score=("risk_score_0_100", lambda s: float(s.quantile(0.95))),
            medium_plus_rate=("is_medium_plus", "mean"),
            high_critical_rate=("is_high_critical", "mean"),
            avg_amount=("TRANS_AMOUNT", "mean"),
        )
    )
    quarterly = (
        stability.groupby("quarter", as_index=False)
        .agg(
            transaction_count=("transaction_row_id", "size"),
            avg_risk_score=("risk_score_0_100", "mean"),
            p95_risk_score=("risk_score_0_100", lambda s: float(s.quantile(0.95))),
            medium_plus_rate=("is_medium_plus", "mean"),
            high_critical_rate=("is_high_critical", "mean"),
        )
    )
    return monthly, quarterly


def write_figures(scores: pd.DataFrame, root_cause: pd.DataFrame, monthly_stability: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.histplot(scores["risk_score_0_100"], bins=60, color="#2F6B8F")
    plt.title("Risk score distribution on real 2019 transactions")
    plt.xlabel("Risk score (0-100)")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_score_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    order = ["Low", "Medium", "High", "Critical"]
    sns.countplot(data=scores, x="risk_band", order=order, color="#2F6B8F")
    plt.title("Operational review bands")
    plt.xlabel("Risk band")
    plt.ylabel("Transaction count")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_band_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=root_cause, y="primary_cause_branch", x="high_or_critical_transactions", color="#C46243")
    plt.title("High/Critical transactions by root-cause branch")
    plt.xlabel("High/Critical transaction count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "root_cause_high_critical.png", dpi=180)
    plt.close()

    amount_plot = scores.copy()
    amount_plot["log_amount"] = np.log1p(amount_plot["TRANS_AMOUNT"])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=amount_plot, x="risk_band", y="log_amount", order=order, color="#88A868")
    plt.title("Transaction amount by risk band")
    plt.xlabel("Risk band")
    plt.ylabel("log(1 + amount)")
    plt.tight_layout()
    plt.savefig(figures_dir / "amount_by_risk_band.png", dpi=180)
    plt.close()

    linkage = scores.groupby("risk_band", as_index=False)[["daily_activity_count", "late_stage_activity_count"]].mean()
    linkage["risk_band"] = pd.Categorical(linkage["risk_band"], categories=order, ordered=True)
    linkage = linkage.sort_values("risk_band")
    plt.figure(figsize=(8, 5))
    x = np.arange(len(linkage))
    width = 0.35
    plt.bar(x - width / 2, linkage["daily_activity_count"], width, label="Daily activity count", color="#2F6B8F")
    plt.bar(x + width / 2, linkage["late_stage_activity_count"], width, label="Late-stage activity count", color="#C46243")
    plt.xticks(x, linkage["risk_band"].astype(str))
    plt.title("Digital activity linkage by risk band")
    plt.xlabel("Risk band")
    plt.ylabel("Average count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "activity_transaction_linkage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(monthly_stability["month"], monthly_stability["avg_risk_score"], marker="o", color="#2F6B8F", label="Avg risk score")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Average risk score")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(monthly_stability["month"], monthly_stability["high_critical_rate"] * 100, marker="s", color="#C46243", label="High/Critical rate")
    ax2.set_ylabel("High/Critical rate (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title("Monthly stability backtest on 2019 data")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_stability_backtest.png", dpi=180)
    plt.close()


def build_metrics(
    scores: pd.DataFrame,
    customer_summary: pd.DataFrame,
    root_cause: pd.DataFrame,
    schema_report: dict[str, Any],
    thresholds: dict[str, float],
    band_thresholds: dict[str, float],
    activity_no_threshold: int,
    feature_importance: pd.Series,
    monthly_stability: pd.DataFrame,
    quarterly_stability: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    return {
        "data_source": "Processed_Data real contest tables",
        "assignment_track": "Problem 1 - Fraud & Anomaly Detection",
        "ground_truth_available": False,
        "evaluation_note": (
            "The provided real data has no confirmed fraud labels, so this package reports monitoring coverage, "
            "risk-band distribution, schema checks, and review-queue outputs instead of precision/recall against fake labels."
        ),
        "pipeline_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "row_counts": {
            "transactions_scored": int(len(scores)),
            "customers_scored": int(customer_summary["CUSTOMER_NUMBER"].nunique()),
            "activity_customer_days": int(schema_report.get("activity_daily_aggregated", {}).get("rows", 0)),
        },
        "date_range": {
            "min": str(scores["TRANS_DATE"].min().date()),
            "max": str(scores["TRANS_DATE"].max().date()),
        },
        "schema_report": schema_report,
        "rule_thresholds": thresholds,
        "risk_band_thresholds": band_thresholds,
        "activity_no_late_stage_threshold_p90": int(activity_no_threshold),
        "risk_band_distribution": scores["risk_band"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0).astype(int).to_dict(),
        "root_cause_distribution": root_cause.set_index("primary_cause_branch")["high_or_critical_transactions"].astype(int).to_dict(),
        "monthly_stability": monthly_stability.round(6).to_dict(orient="records"),
        "quarterly_stability": quarterly_stability.round(6).to_dict(orient="records"),
        "stability_note": (
            "The 2019 holdout-style stability check tracks monthly and quarterly review rates. "
            "Because only one calendar year is provided, this is a temporal robustness check, not a crisis-period backtest."
        ),
        "review_queue": {
            "medium_or_above_transactions": int(scores["risk_band"].isin(["Medium", "High", "Critical"]).sum()),
            "high_or_critical_transactions": int(scores["risk_band"].isin(["High", "Critical"]).sum()),
            "critical_transactions": int(scores["risk_band"].eq("Critical").sum()),
            "customers_with_high_or_critical": int(customer_summary["high_or_critical_count"].gt(0).sum()),
        },
        "top_surrogate_features": feature_importance.head(15).round(6).to_dict(),
    }


def write_report_outline(metrics: dict[str, Any], root_cause: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    high_critical = metrics["review_queue"]["high_or_critical_transactions"]
    critical = metrics["review_queue"]["critical_transactions"]
    rows = metrics["row_counts"]["transactions_scored"]
    root_lines = "\n".join(
        f"- {row.primary_cause_branch}: {int(row.high_or_critical_transactions):,} High/Critical giao dịch, "
        f"risk trung bình {row.avg_risk_score:.1f}/100."
        for row in root_cause.itertuples()
    )
    content = f"""# Final Report Outline - Fraud & Anomaly Detection

## 1. Bối cảnh và mục tiêu

Bài toán 1 của G'Contest yêu cầu phát hiện gian lận và bất thường từ dữ liệu ngân hàng 360 độ: chân dung khách hàng, giao dịch, digital activity và sản phẩm. Dữ liệu thật không có nhãn fraud đã xác minh, nên giải pháp không tự bịa nhãn. Framework được thiết kế như một hệ thống xếp hạng rủi ro để đưa giao dịch vào hàng đợi review.

## 2. Cách tiếp cận nguyên nhân trước model

Thay vì đưa model trước, nhóm xác định ba nhánh nguyên nhân theo đúng assignment:

1. Account takeover / identity compromise: thiết bị mới, IP mới, hoạt động đêm, hoạt động digital ở giai đoạn muộn, người thụ hưởng mới.
2. Unauthorized transfer / capital outflow: chuyển khoản ra ngoài ngân hàng, số tiền lệch baseline, tần suất giao dịch tăng, dòng tiền ra lớn so với CASA.
3. AML network / mule-account pattern: IP/device dùng chung nhiều khách hàng, người thụ hưởng nhận tiền từ nhiều khách hàng, giao dịch số tròn giá trị cao.

`ACTIVITY_NO` được dùng đúng ý nghĩa trong note mentor: số lớn hơn là hành động sau hơn, nên top 10% `ACTIVITY_NO` trong dữ liệu được xem là late-stage digital activity.

## 3. Dữ liệu sử dụng

- Nguồn: `Processed_Data/` và `G_Contest 26'_3rd round assignment.docx`.
- Số giao dịch scored: {rows:,}.
- Số khách hàng scored: {metrics["row_counts"]["customers_scored"]:,}.
- Thời gian dữ liệu: {metrics["date_range"]["min"]} đến {metrics["date_range"]["max"]}.
- Không dùng synthetic data, không dùng synthetic ground truth.

## 4. Framework kỹ thuật

1. Chuẩn hóa schema theo data dictionary, đồng thời xử lý khác biệt tên cột trong file thật như `TRANS_LV1`/`TRXN_LV1` và `LIMIT_AMT_CREDIT`/`LIMIT_AMT`.
2. Tạo baseline hành vi theo từng khách hàng: số tiền trung bình, P95, tần suất ngày, thiết bị/IP/người thụ hưởng đã từng thấy.
3. Liên kết digital activity cùng ngày với giao dịch: activity đêm, late-stage activity, activity liên quan account/authentication.
4. Chấm điểm rule-based theo ba nhánh nguyên nhân.
5. Huấn luyện Isolation Forest không giám sát để bắt giao dịch lệch khỏi phân bố hành vi chung.
6. Điểm cuối = 60% rule score + 40% model anomaly score, sau đó chia band theo capacity review: Low, Medium, High, Critical.
7. xAI engine xuất `top_reasons` và `recommended_action` cho từng giao dịch.

## 5. Kết quả chính

- High/Critical transactions: {high_critical:,}.
- Critical transactions: {critical:,}.
- Khách hàng có High/Critical transaction: {metrics["review_queue"]["customers_with_high_or_critical"]:,}.

Root-cause summary:

{root_lines}

Monthly stability backtest:

- Framework được kiểm tra theo từng tháng trong năm 2019.
- `model_metrics.json` có bảng monthly/quarterly stability gồm transaction count, average risk, P95 risk và High/Critical rate.
- Vì dữ liệu chỉ có năm 2019, đây là temporal robustness check, không phải crisis-period validation.

## 6. Vì sao không báo Precision/Recall như bài có nhãn?

Vì file thật không có confirmed fraud label. Báo precision/recall dựa trên nhãn tự bịa sẽ làm sai bản chất bài thi. Notebook vẫn có phần evaluation, nhưng evaluation ở đây là:

- Schema/data quality checks.
- Coverage của review queue.
- Distribution theo risk band.
- Kiểm tra top-risk có reason codes rõ ràng.
- Chuẩn bị cơ chế nhận feedback từ investigator để hiệu chỉnh threshold/model sau này.

## 7. Gợi ý vận hành thực tế

- Critical: near-real-time hold/manual review, gọi xác minh khách hàng, kiểm tra device/IP/beneficiary.
- High: step-up authentication hoặc manual review trong ngày.
- AML branch: escalation theo mạng lưới IP/device/beneficiary, không nhìn từng giao dịch riêng lẻ.
- Medium: theo dõi tăng cường và nâng cấp nếu lặp lại trong 7 ngày.
- KPI sau khi triển khai: hit rate trong top-K, false positive rate theo phân khúc, review capacity, số case AML escalation, time-to-review.

## 8. Liên hệ với chuẩn nghiệp vụ quốc tế

- FATF Risk-Based Approach for Banking Sector: ngân hàng nên hiểu mức độ rủi ro, ưu tiên nguồn lực vào nơi rủi ro cao và áp dụng biện pháp giảm thiểu tương ứng. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- FFIEC Authentication and Access Guidance: với digital banking, kiểm soát nên theo hướng layered security, MFA/step-up authentication và tăng kiểm soát khi giao dịch hoặc truy cập có rủi ro cao. Link: https://www.ffiec.gov/news/press-releases/2021/pr-08-11
- FATF Guidance on Risk-Based Supervision: tránh cách làm tick-box, tập trung vào full spectrum of risks và nơi rủi ro cao hơn. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-supervision.html

Framework của dự án bám đúng tinh thần này: risk-based, layered controls, review queue theo capacity, và xAI reason codes để investigator kiểm tra được.

## 9. Hạn chế

Đây là framework chuẩn cho dữ liệu không nhãn, không phải model xác nhận fraud. Khi ngân hàng có kết quả review thật, cần đưa label đó quay lại pipeline để hiệu chỉnh threshold, huấn luyện supervised model, và đo Precision@K/Recall@K chính thức.
"""
    (report_dir / "final_report_outline.md").write_text(content, encoding="utf-8")


def write_outputs(
    scores: pd.DataFrame,
    customer_summary: pd.DataFrame,
    root_cause: pd.DataFrame,
    monthly_stability: pd.DataFrame,
    quarterly_stability: pd.DataFrame,
    metrics: dict[str, Any],
    config: PipelineConfig,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "transaction_row_id",
        "CUSTOMER_NUMBER",
        "TRANS_DATE",
        "DAY_OF_WEEK",
        "TRANS_HOUR",
        "TRANS_NO",
        "TRANS_LV1",
        "TRANS_LV2",
        "TRANS_AMOUNT",
        "IP_Address_Proxy",
        "Device_ID_Hash",
        "Device_OS",
        "Beneficiary_CUSTOMER_NUMBER",
        "rule_score_0_100",
        "model_anomaly_score_0_100",
        "risk_score_0_100",
        "risk_band",
        "primary_cause_branch",
        "top_reasons",
        "recommended_action",
    ]
    scores.sort_values("risk_score_0_100", ascending=False)[output_cols].to_csv(config.output_dir / "transaction_risk_scores.csv", index=False)
    customer_summary.to_csv(config.output_dir / "customer_risk_summary.csv", index=False)
    root_cause.to_csv(config.output_dir / "root_cause_summary.csv", index=False)
    monthly_stability.to_csv(config.output_dir / "monthly_stability.csv", index=False)
    quarterly_stability.to_csv(config.output_dir / "quarterly_stability.csv", index=False)
    scores.sort_values("risk_score_0_100", ascending=False)[output_cols].head(1000).to_csv(config.output_dir / "top_review_queue.csv", index=False)
    xai_cols = ["transaction_row_id", "CUSTOMER_NUMBER", "risk_score_0_100", "risk_band", "primary_cause_branch", *MODEL_FEATURES]
    top_n = min(10_000, len(scores), config.xai_sample // 4)
    random_n = min(config.xai_sample - top_n, len(scores))
    xai_sample = pd.concat(
        [
            scores.sort_values("risk_score_0_100", ascending=False)[xai_cols].head(top_n),
            scores[xai_cols].sample(random_n, random_state=config.random_state),
        ],
        ignore_index=True,
    ).drop_duplicates("transaction_row_id")
    xai_sample.to_csv(config.output_dir / "xai_feature_sample.csv", index=False)
    (config.output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    tables = load_reference_tables(config.raw_dir)
    activity_daily, activity_no_threshold = aggregate_activity(config.raw_dir, config.chunksize)
    schema_report = validate_schema(config.raw_dir, tables, activity_daily)
    features = build_features(tables, activity_daily)
    thresholds = build_thresholds(features)
    rule_scored = build_rule_score(features, thresholds)
    model_scored, _ = score_isolation_forest(rule_scored, config)
    banded, band_thresholds = assign_risk_bands(model_scored)
    feature_importance = build_surrogate_importance(banded, config)
    final_scores = finalise_explanations(banded, feature_importance)
    customer_summary = build_customer_summary(final_scores)
    root_cause = build_root_cause_summary(final_scores)
    monthly_stability, quarterly_stability = build_time_stability(final_scores)
    write_figures(final_scores, root_cause, monthly_stability, config.figures_dir)
    metrics = build_metrics(
        final_scores,
        customer_summary,
        root_cause,
        schema_report,
        thresholds,
        band_thresholds,
        activity_no_threshold,
        feature_importance,
        monthly_stability,
        quarterly_stability,
        config,
    )
    write_report_outline(metrics, root_cause, config.report_dir)
    write_outputs(final_scores, customer_summary, root_cause, monthly_stability, quarterly_stability, metrics, config)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data fraud and anomaly detection pipeline for G'Contest.")
    parser.add_argument("--raw-dir", type=Path, default=Path("Processed_Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--iforest-fit-sample", type=int, default=200_000)
    parser.add_argument("--iforest-max-samples", type=int, default=50_000)
    parser.add_argument("--surrogate-sample", type=int, default=120_000)
    parser.add_argument("--xai-sample", type=int, default=80_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        report_dir=args.report_dir,
        chunksize=args.chunksize,
        iforest_fit_sample=args.iforest_fit_sample,
        iforest_max_samples=args.iforest_max_samples,
        surrogate_sample=args.surrogate_sample,
        xai_sample=args.xai_sample,
    )
    metrics = run_pipeline(config)
    print(json.dumps({"outputs": str(config.output_dir.resolve()), "metrics": metrics["review_queue"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
