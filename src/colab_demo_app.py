from __future__ import annotations

import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBClassifier


CLEANED_DIR = Path(os.environ.get("COLAB_CLEANED_DIR", "outputs/vong3_2_cleaned"))
FIGURES_DIR = Path(os.environ.get("COLAB_FIGURES_DIR", "outputs/vong3_2_figures"))


MASTER_COLS = [
    "CUSTOMER_NUMBER",
    "txn_count",
    "total_trans_amount",
    "avg_trans_amount",
    "max_trans_amount",
    "unique_devices",
    "unique_ips",
    "beneficiary_count",
    "night_txn_ratio",
    "outside_bank_ratio",
    "max_inactive_gap",
    "burst_max",
    "rolling_30d_txn_count",
    "rolling_60d_txn_count",
    "rolling_90d_txn_count",
    "rolling_30d_amount_sum",
    "rolling_60d_amount_sum",
    "rolling_90d_amount_sum",
    "total_app_activities",
    "night_activity_ratio",
    "password_change_count",
    "max_daily_activity",
    "avg_balance_ca",
    "balance_volatility",
    "max_cic_overdue_days",
    "card_utilization_ratio",
    "rule_behavior_device",
    "rule_ato",
    "rule_money_mule",
    "rule_dormant_active",
    "rule_night_anomaly",
    "score_fraud_rule",
    "score_behavioral_instability",
    "score_aml_risk",
    "final_risk_score",
    "Risk_Segment",
    "Fraud",
    "Reason_Code_Details",
    "ML_Pred",
    "Rule_Detected",
    "Business_Action",
]

MODEL_FEATURE_COLS = [
    "txn_count",
    "total_trans_amount",
    "avg_trans_amount",
    "max_trans_amount",
    "std_trans_amount",
    "unique_devices",
    "unique_ips",
    "beneficiary_count",
    "night_txn_ratio",
    "outside_bank_ratio",
    "max_inactive_gap",
    "burst_max",
    "total_app_activities",
    "avg_activity_hour",
    "night_activity_ratio",
    "password_change_count",
    "max_daily_activity",
    "avg_balance_ca",
    "balance_volatility",
    "max_cic_overdue_days",
    "card_utilization_ratio",
]


@st.cache_data(show_spinner=False)
def load_colab_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    master_path = CLEANED_DIR / "Customer_360_Demo_Light.csv"
    if not master_path.exists():
        master_path = CLEANED_DIR / "Customer_360_Master_Data.csv"
    metrics_path = CLEANED_DIR / "colab_metrics.json"
    if not master_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Missing Vòng 3 notebook outputs. Run `scripts/run_demo.sh` or `scripts/run_demo.ps1` first.")
    master = pd.read_csv(master_path, usecols=lambda col: col in MASTER_COLS, low_memory=False)
    master["CUSTOMER_NUMBER"] = master["CUSTOMER_NUMBER"].astype(str)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_importance = pd.read_csv(CLEANED_DIR / "colab_feature_importance.csv")
    shap_importance = pd.read_csv(CLEANED_DIR / "colab_shap_feature_importance.csv")
    shap_local_path = CLEANED_DIR / "colab_shap_local_explanations.csv"
    shap_local = pd.read_csv(shap_local_path) if shap_local_path.exists() else pd.DataFrame()
    return master, feature_importance, shap_importance, shap_local, metrics


def advisor_text(row: pd.Series) -> str:
    return (
        f"Customer {row['CUSTOMER_NUMBER']} is {row['Risk_Segment']} with Vòng 3 risk score "
        f"{float(row['final_risk_score']):.0f}/100. Business action: {row['Business_Action']}. "
        f"Reasons: {row['Reason_Code_Details']}"
    )


def find_customer(query: str, master: pd.DataFrame) -> pd.DataFrame:
    ids = re.findall(r"\d+", query)
    if not ids:
        return pd.DataFrame()
    return master.loc[master["CUSTOMER_NUMBER"].eq(ids[0])]


def show_case(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk segment", row["Risk_Segment"])
    c2.metric("Vòng 3 risk score", f"{float(row['final_risk_score']):.0f}")
    c3.metric("Max amount", f"{float(row['max_trans_amount']):,.0f}")
    c4.metric("Action", row["Business_Action"])
    st.write("**Reason code from Vòng 3 notebook logic**")
    st.write(row["Reason_Code_Details"])
    st.write("**Rule flags**")
    rule_cols = ["rule_behavior_device", "rule_ato", "rule_money_mule", "rule_dormant_active", "rule_night_anomaly"]
    st.dataframe(row[rule_cols].to_frame("value"), use_container_width=True)
    st.write("**Customer 360 baseline**")
    profile_cols = [
        "txn_count",
        "avg_trans_amount",
        "max_trans_amount",
        "unique_devices",
        "unique_ips",
        "beneficiary_count",
        "night_txn_ratio",
        "outside_bank_ratio",
        "max_inactive_gap",
        "burst_max",
        "rolling_30d_txn_count",
        "rolling_90d_amount_sum",
        "total_app_activities",
        "password_change_count",
        "avg_balance_ca",
        "max_cic_overdue_days",
        "card_utilization_ratio",
    ]
    profile_cols = [col for col in profile_cols if col in row.index]
    st.dataframe(row[profile_cols].to_frame("value"), use_container_width=True)


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Không tìm thấy figure: {path.name}")


@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: lambda _: "colab_exact_customer_360_demo"})
def train_runtime_transaction_model(master: pd.DataFrame) -> tuple[XGBClassifier, list[str]]:
    feature_cols = [col for col in MODEL_FEATURE_COLS if col in master.columns]
    train_df = master[feature_cols + ["Fraud"]].copy()
    train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(train_df["Fraud"], errors="coerce").fillna(0).astype(int)
    negative = max(1, int((y == 0).sum()))
    positive = max(1, int((y == 1).sum()))
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=negative / positive,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(train_df[feature_cols], y)
    return model, feature_cols


def risk_segment(score: float) -> str:
    if score == 0:
        return "Low"
    if score <= 30:
        return "Medium"
    if score <= 50:
        return "High"
    return "Critical"


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _truthy(value: object, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "co", "có", "outside", "external"}:
        return True
    if text in {"0", "false", "no", "n", "khong", "không", "internal"}:
        return False
    return default


def _first_present(row: pd.Series, names: list[str], default: object = None) -> object:
    lower_map = {str(col).lower(): col for col in row.index}
    for name in names:
        actual = lower_map.get(name.lower())
        if actual is not None and not pd.isna(row[actual]):
            return row[actual]
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _to_int(value: object, default: int = 0) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return int(parsed)


def _normalize_choice(value: object, valid_values: list[str], default: str) -> str:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    for option in valid_values:
        if text == option.lower():
            return option
    if "shared" in text or "farm" in text or "trại" in text:
        return "Shared/farm device"
    if "new" in text or "lạ" in text or "la" in text:
        if "ip" in text or "proxy" in text:
            return "New IP/proxy"
        if "beneficiary" in text or "external" in text or "thụ hưởng" in text:
            return "New external beneficiary"
        return "New device"
    if "merchant" in text or text in {"0", "0.0"}:
        return "Merchant/zero beneficiary"
    return default


def _choice_from_boolean_flag(row: pd.Series, names: list[str], true_value: str, false_value: str) -> str | None:
    lower_map = {str(col).lower(): col for col in row.index}
    for name in names:
        actual = lower_map.get(name.lower())
        if actual is not None and not pd.isna(row[actual]):
            return true_value if _truthy(row[actual]) else false_value
    return None


def _portfolio_default_profile(master: pd.DataFrame, customer_number: str) -> pd.Series:
    values: dict[str, object] = {"CUSTOMER_NUMBER": str(customer_number)}
    for col in sorted(set(MASTER_COLS + MODEL_FEATURE_COLS)):
        if col == "CUSTOMER_NUMBER":
            continue
        if col in master.columns and pd.api.types.is_numeric_dtype(master[col]):
            values[col] = float(pd.to_numeric(master[col], errors="coerce").median())
        else:
            values[col] = 0
    values.update(
        {
            "rule_behavior_device": 0,
            "rule_ato": 0,
            "rule_money_mule": 0,
            "rule_dormant_active": 0,
            "rule_night_anomaly": 0,
            "score_fraud_rule": 0.0,
            "score_behavioral_instability": 0.0,
            "score_aml_risk": 0.0,
            "final_risk_score": 0.0,
            "Risk_Segment": "Low",
            "Fraud": 0,
            "Reason_Code_Details": "",
            "ML_Pred": 0,
            "Rule_Detected": 0,
            "Business_Action": "PASS: ALLOW TRANSACTION",
        }
    )
    return pd.Series(values)


def build_manual_customer_profile(master: pd.DataFrame, customer_number: str, overrides: dict[str, object]) -> pd.Series:
    profile = _portfolio_default_profile(master, customer_number)
    for key, value in overrides.items():
        if key in profile.index and value is not None and not pd.isna(value):
            profile[key] = value
    txn_count = max(0.0, float(profile.get("txn_count", 0)))
    avg_amount = max(0.0, float(profile.get("avg_trans_amount", 0)))
    max_amount = max(avg_amount, float(profile.get("max_trans_amount", 0)))
    profile["txn_count"] = txn_count
    profile["avg_trans_amount"] = avg_amount
    profile["max_trans_amount"] = max_amount
    profile["total_trans_amount"] = max(float(profile.get("total_trans_amount", 0)), avg_amount * txn_count)
    profile["std_trans_amount"] = max(float(profile.get("std_trans_amount", 0)), avg_amount * 0.5)
    profile["night_txn_ratio"] = _clip(float(profile.get("night_txn_ratio", 0)), 0, 1)
    profile["outside_bank_ratio"] = _clip(float(profile.get("outside_bank_ratio", 0)), 0, 1)
    profile["night_activity_ratio"] = _clip(float(profile.get("night_activity_ratio", 0)), 0, 1)
    profile["card_utilization_ratio"] = _clip(float(profile.get("card_utilization_ratio", 0)), 0, 1)
    return profile


MANUAL_BASELINE_ALIASES: dict[str, list[str]] = {
    "txn_count": ["txn_count", "historical_txn_count", "baseline_txn_count"],
    "avg_trans_amount": ["avg_trans_amount", "historical_avg_amount", "baseline_avg_amount"],
    "max_trans_amount": ["max_trans_amount", "historical_max_amount", "baseline_max_amount"],
    "unique_devices": ["unique_devices", "trusted_device_count", "device_count"],
    "unique_ips": ["unique_ips", "trusted_ip_count", "ip_count"],
    "beneficiary_count": ["beneficiary_count", "known_beneficiary_count"],
    "night_txn_ratio": ["night_txn_ratio", "baseline_night_ratio"],
    "outside_bank_ratio": ["outside_bank_ratio", "baseline_outside_bank_ratio"],
    "max_inactive_gap": ["max_inactive_gap", "inactive_days"],
    "burst_max": ["burst_max", "historical_burst_max"],
    "total_app_activities": ["total_app_activities", "app_activity_count"],
    "night_activity_ratio": ["night_activity_ratio"],
    "password_change_count": ["password_change_count", "security_change_count"],
    "max_daily_activity": ["max_daily_activity"],
    "avg_balance_ca": ["avg_balance_ca", "avg_balance", "ca_balance"],
    "balance_volatility": ["balance_volatility"],
    "max_cic_overdue_days": ["max_cic_overdue_days", "overdue_days", "cic_overdue_days"],
    "card_utilization_ratio": ["card_utilization_ratio", "card_utilization"],
}


def row_has_manual_baseline(row: pd.Series) -> bool:
    lower_cols = {str(col).lower() for col in row.index}
    return any(alias.lower() in lower_cols for aliases in MANUAL_BASELINE_ALIASES.values() for alias in aliases)


def manual_profile_from_batch_row(row: pd.Series, master: pd.DataFrame, fallback_customer_number: str) -> pd.Series:
    overrides: dict[str, object] = {}
    for target_col, aliases in MANUAL_BASELINE_ALIASES.items():
        value = _first_present(row, aliases, None)
        if value is not None:
            overrides[target_col] = _to_float(value, 0.0)
    return build_manual_customer_profile(master, fallback_customer_number, overrides)


def business_action(rule_detected: int, ml_pred: int) -> str:
    if rule_detected and ml_pred:
        return "CRITICAL: BLOCK IMMEDIATELY"
    if rule_detected and not ml_pred:
        return "WARNING: REQUIRE STEP-UP EKYC/OTP"
    if not rule_detected and ml_pred:
        return "MONITOR: ADD TO SPECIAL WATCHLIST"
    return "PASS: ALLOW TRANSACTION"


def build_transaction_reasons(candidate: pd.Series, amount: float, threshold_amount: float, device_mode: str) -> str:
    if candidate["final_risk_score"] == 0:
        return "Giao dịch chưa vi phạm rule trọng yếu; cho phép đi tiếp hoặc giám sát bình thường."
    reasons: list[str] = []
    if candidate["rule_behavior_device"]:
        reasons.append(f"Device nhập vào thuộc nhóm thiết bị chia sẻ/rủi ro ({device_mode}).")
    if candidate["rule_ato"]:
        reasons.append(
            f"ATO: thiết bị sau giao dịch = {candidate['unique_devices']:.0f}, "
            f"password/security change = {candidate['password_change_count']:.0f}."
        )
    if candidate["rule_money_mule"]:
        reasons.append(
            f"AML/Mule: avg transaction sau cập nhật = {candidate['avg_trans_amount']:,.0f} VND "
            f"> IQR threshold {threshold_amount:,.0f} VND và overdue = {candidate['max_cic_overdue_days']:.0f}."
        )
    if candidate["rule_dormant_active"]:
        reasons.append(
            f"Behavior: tài khoản từng inactive {candidate['max_inactive_gap']:.0f} ngày, "
            f"burst hôm nay = {candidate['burst_max']:.0f} giao dịch/ngày."
        )
    if candidate["rule_night_anomaly"]:
        reasons.append(
            f"Night anomaly: tỷ lệ giao dịch đêm sau cập nhật = {candidate['night_txn_ratio']:.1%}, "
            f"amount hiện tại = {amount:,.0f} VND."
        )
    if not reasons:
        reasons.append(
            f"Điểm runtime phản ánh tín hiệu nhẹ: amount/threshold={candidate.get('runtime_amount_ratio', 0):.2f}, "
            f"device={device_mode}, ML probability={candidate.get('model_fraud_probability', 0):.1%}."
        )
    return " | ".join(reasons)


def score_runtime_transaction(
    candidate: pd.Series,
    amount: float,
    hour: int,
    device_mode: str,
    ip_mode: str,
    beneficiary_mode: str,
    outside_bank: bool,
    recent_security_change: bool,
    txns_today: int,
    threshold_amount: float,
) -> pd.Series:
    amount_ratio = amount / max(threshold_amount, 1.0)
    avg_ratio = float(candidate.get("avg_trans_amount", 0)) / max(threshold_amount, 1.0)
    night_hour = hour in [23, 0, 1, 2, 3, 4]
    new_device = device_mode in {"New device", "Shared/farm device"}
    new_ip = ip_mode == "New IP/proxy"
    risky_beneficiary = beneficiary_mode in {"New external beneficiary", "Merchant/zero beneficiary"}

    ato_score = 0.0
    if device_mode == "New device":
        ato_score += 16
    elif device_mode == "Shared/farm device":
        ato_score += 30
    if new_ip:
        ato_score += 10
    if recent_security_change:
        ato_score += 10
    if new_device and recent_security_change:
        ato_score += 8
    candidate["score_fraud_rule"] = round(_clip(ato_score, 0, 50), 2)

    behavior_score = 0.0
    if float(candidate.get("max_inactive_gap", 0)) > 60 and txns_today > 5:
        behavior_score += 8 + min(12, (txns_today - 5) * 1.5)
    if night_hour:
        behavior_score += 5 + min(8, amount_ratio * 2)
    if float(candidate.get("night_txn_ratio", 0)) > 0.6:
        behavior_score += 7
    candidate["score_behavioral_instability"] = round(_clip(behavior_score, 0, 30), 2)

    aml_score = 0.0
    if outside_bank:
        aml_score += 4
    if risky_beneficiary:
        aml_score += 4
    if amount_ratio > 1:
        aml_score += min(8, 2 + amount_ratio * 1.5)
    if avg_ratio > 1:
        aml_score += min(4, avg_ratio)
    if float(candidate.get("max_cic_overdue_days", 0)) > 0:
        aml_score += 4
    candidate["score_aml_risk"] = round(_clip(aml_score, 0, 20), 2)

    candidate["runtime_amount_ratio"] = round(float(amount_ratio), 4)
    candidate["runtime_avg_amount_ratio"] = round(float(avg_ratio), 4)
    candidate["final_risk_score"] = round(
        _clip(
            float(candidate["score_fraud_rule"])
            + float(candidate["score_behavioral_instability"])
            + float(candidate["score_aml_risk"]),
            0,
            100,
        ),
        2,
    )
    return candidate


def simulate_transaction(
    base_row: pd.Series,
    amount: float,
    hour: int,
    device_mode: str,
    ip_mode: str,
    beneficiary_mode: str,
    outside_bank: bool,
    recent_security_change: bool,
    txns_today: int,
    threshold_amount: float,
    model: XGBClassifier,
    feature_cols: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    candidate = base_row.copy()
    txn_count = max(0.0, float(candidate.get("txn_count", 0)))
    total_amount = max(0.0, float(candidate.get("total_trans_amount", 0)))
    night_count = float(candidate.get("night_txn_ratio", 0)) * txn_count
    outside_count = float(candidate.get("outside_bank_ratio", 0)) * txn_count

    candidate["txn_count"] = txn_count + 1
    candidate["total_trans_amount"] = total_amount + amount
    candidate["avg_trans_amount"] = candidate["total_trans_amount"] / max(1, candidate["txn_count"])
    candidate["max_trans_amount"] = max(float(candidate.get("max_trans_amount", 0)), amount)
    candidate["unique_devices"] = float(candidate.get("unique_devices", 0)) + (1 if device_mode != "Known device" else 0)
    candidate["unique_ips"] = float(candidate.get("unique_ips", 0)) + (1 if ip_mode != "Known IP" else 0)
    candidate["beneficiary_count"] = float(candidate.get("beneficiary_count", 0)) + (
        1 if beneficiary_mode != "Known beneficiary" else 0
    )
    candidate["night_txn_ratio"] = (night_count + int(hour in [23, 0, 1, 2, 3, 4])) / max(1, candidate["txn_count"])
    candidate["outside_bank_ratio"] = (outside_count + int(outside_bank)) / max(1, candidate["txn_count"])
    candidate["password_change_count"] = float(candidate.get("password_change_count", 0)) + int(recent_security_change)
    candidate["burst_max"] = max(float(candidate.get("burst_max", 0)), float(txns_today))

    candidate["rule_behavior_device"] = int(device_mode == "Shared/farm device")
    candidate["rule_ato"] = int(candidate["unique_devices"] > 1.0 and candidate["password_change_count"] > 0)
    candidate["rule_money_mule"] = int(candidate["avg_trans_amount"] > threshold_amount and candidate["max_cic_overdue_days"] > 0)
    candidate["rule_dormant_active"] = int(candidate["max_inactive_gap"] > 60 and candidate["burst_max"] > 5)
    candidate["rule_night_anomaly"] = int(
        candidate["night_txn_ratio"] > 0.6 and candidate["avg_trans_amount"] > threshold_amount * 0.5
    )

    candidate = score_runtime_transaction(
        candidate,
        amount,
        hour,
        device_mode,
        ip_mode,
        beneficiary_mode,
        outside_bank,
        recent_security_change,
        txns_today,
        threshold_amount,
    )
    candidate["Risk_Segment"] = risk_segment(float(candidate["final_risk_score"]))
    candidate["Fraud"] = int(candidate["final_risk_score"] > 0)

    model_input = pd.DataFrame([{col: candidate.get(col, 0) for col in feature_cols}])
    model_input = model_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    model_proba = float(model.predict_proba(model_input)[0, 1])
    candidate["model_fraud_probability"] = round(model_proba, 6)
    candidate["ML_Pred"] = int(model_proba >= 0.5)
    candidate["Rule_Detected"] = int(
        candidate["final_risk_score"] >= 30
        or candidate["rule_ato"]
        or candidate["rule_money_mule"]
        or candidate["rule_behavior_device"]
    )
    candidate["Business_Action"] = business_action(int(candidate["Rule_Detected"]), int(candidate["ML_Pred"]))
    candidate["Reason_Code_Details"] = build_transaction_reasons(candidate, amount, threshold_amount, device_mode)

    audit = pd.DataFrame(
        [
            {"step": "1. Customer 360 baseline", "output": f"CIF {candidate['CUSTOMER_NUMBER']} loaded from exact Colab master."},
            {"step": "2. Transaction update", "output": f"Amount {amount:,.0f} VND at hour {hour}; device={device_mode}; IP={ip_mode}."},
            {
                "step": "3. Rule engine",
                "output": (
                    f"ATO={candidate['rule_ato']}, AML={candidate['rule_money_mule']}, "
                    f"Behavior={candidate['rule_dormant_active'] or candidate['rule_night_anomaly']}, "
                    f"DeviceFarm={candidate['rule_behavior_device']}"
                ),
            },
            {
                "step": "4. Weak-label ML model",
                "output": f"ML_Pred={candidate['ML_Pred']} with probability={candidate['model_fraud_probability']:.1%}.",
            },
            {"step": "5. Hybrid action matrix", "output": str(candidate["Business_Action"])},
        ]
    )
    return candidate, audit


def weak_label_action_rates(master: pd.DataFrame, impact: dict) -> dict[str, float]:
    if {"Fraud", "Business_Action"}.issubset(master.columns):
        weak = master["Fraud"].eq(1)
        weak_count = int(weak.sum())
        if weak_count:
            action = master["Business_Action"].astype(str)
            blocked = action.eq("CRITICAL: BLOCK IMMEDIATELY")
            step_up = action.eq("WARNING: REQUIRE STEP-UP EKYC/OTP")
            watchlist = action.eq("MONITOR: ADD TO SPECIAL WATCHLIST")
            return {
                "blocked": float((weak & blocked).sum() / weak_count),
                "step_up": float((weak & step_up).sum() / weak_count),
                "challenge": float((weak & (blocked | step_up)).sum() / weak_count),
                "review": float((weak & (blocked | step_up | watchlist)).sum() / weak_count),
            }
    return {
        "blocked": float(impact.get("blocked_coverage_against_rule_label", 0.0)),
        "step_up": float(impact.get("step_up_coverage_against_rule_label", 0.0)),
        "challenge": float(impact.get("challenge_coverage_against_rule_label", 0.0)),
        "review": float(impact.get("review_coverage_against_rule_label", impact.get("challenge_coverage_against_rule_label", 0.0))),
    }


def batch_row_to_simulation_args(row: pd.Series) -> dict[str, object]:
    amount = _to_float(_first_present(row, ["TRANS_AMOUNT", "amount", "transaction_amount"], 0), 0.0)
    hour = _to_int(_first_present(row, ["TRANS_HOUR", "hour", "transaction_hour"], 12), 12)
    hour = int(_clip(hour, 0, 23))

    device_mode = _normalize_choice(
        _first_present(row, ["Device status", "DEVICE_STATUS", "device_mode", "device_status", "is_new_device"], "Known device"),
        ["Known device", "New device", "Shared/farm device"],
        "Known device",
    )
    device_mode = _choice_from_boolean_flag(row, ["is_new_device"], "New device", "Known device") or device_mode
    ip_mode = _normalize_choice(
        _first_present(row, ["IP status", "IP_STATUS", "ip_mode", "ip_status", "is_new_ip"], "Known IP"),
        ["Known IP", "New IP/proxy"],
        "Known IP",
    )
    ip_mode = _choice_from_boolean_flag(row, ["is_new_ip"], "New IP/proxy", "Known IP") or ip_mode
    beneficiary_mode = _normalize_choice(
        _first_present(
            row,
            ["Beneficiary status", "BENEFICIARY_STATUS", "beneficiary_mode", "beneficiary_status", "Beneficiary_CUSTOMER_NUMBER"],
            "Known beneficiary",
        ),
        ["Known beneficiary", "New external beneficiary", "Merchant/zero beneficiary"],
        "Known beneficiary",
    )
    beneficiary_mode = (
        _choice_from_boolean_flag(row, ["is_new_beneficiary"], "New external beneficiary", "Known beneficiary")
        or beneficiary_mode
    )
    outside_bank = _truthy(_first_present(row, ["outside_bank", "Outside-bank transfer", "is_outside_bank"], False))
    recent_security_change = _truthy(
        _first_present(row, ["recent_security_change", "Recent password/security change", "password_change", "is_security_change"], False)
    )
    txns_today = _to_int(_first_present(row, ["txns_today", "Transactions today after this txn", "burst_max"], 1), 1)
    return {
        "amount": amount,
        "hour": hour,
        "device_mode": device_mode,
        "ip_mode": ip_mode,
        "beneficiary_mode": beneficiary_mode,
        "outside_bank": outside_bank,
        "recent_security_change": recent_security_change,
        "txns_today": max(1, txns_today),
    }


def score_batch_transactions(
    upload_df: pd.DataFrame,
    master: pd.DataFrame,
    threshold_amount: float,
    model: XGBClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    if "CUSTOMER_NUMBER" not in upload_df.columns and not any(row_has_manual_baseline(row) for _, row in upload_df.iterrows()):
        raise ValueError("CSV phải có `CUSTOMER_NUMBER` hoặc các cột baseline tự nhập như `txn_count`, `avg_trans_amount`.")

    master_by_customer = master.assign(CUSTOMER_NUMBER=master["CUSTOMER_NUMBER"].astype(str)).set_index("CUSTOMER_NUMBER", drop=False)
    results: list[dict[str, object]] = []
    for idx, row in upload_df.iterrows():
        customer_number = str(row["CUSTOMER_NUMBER"]).strip() if "CUSTOMER_NUMBER" in upload_df.columns and not pd.isna(row.get("CUSTOMER_NUMBER")) else f"NEW_ROW_{idx + 1}"
        baseline_source = "DATA_CUSTOMER_360"
        if customer_number in master_by_customer.index:
            base_profile = master_by_customer.loc[customer_number]
        elif row_has_manual_baseline(row):
            baseline_source = "CSV_MANUAL_BASELINE"
            base_profile = manual_profile_from_batch_row(row, master, customer_number)
        else:
            results.append(
                {
                    "row_number": idx + 1,
                    "CUSTOMER_NUMBER": customer_number,
                    "status": "CUSTOMER_NOT_FOUND",
                    "final_risk_score": np.nan,
                    "Risk_Segment": "",
                    "ML_Pred": "",
                    "Business_Action": "",
                    "Reason_Code_Details": "Không tìm thấy CUSTOMER_NUMBER trong Customer 360 baseline.",
                }
            )
            continue

        args = batch_row_to_simulation_args(row)
        scored, _ = simulate_transaction(
            base_profile,
            float(args["amount"]),
            int(args["hour"]),
            str(args["device_mode"]),
            str(args["ip_mode"]),
            str(args["beneficiary_mode"]),
            bool(args["outside_bank"]),
            bool(args["recent_security_change"]),
            int(args["txns_today"]),
            threshold_amount,
            model,
            feature_cols,
        )
        results.append(
            {
                "row_number": idx + 1,
                "CUSTOMER_NUMBER": customer_number,
                "status": "SCORED",
                "baseline_source": baseline_source,
                "TRANS_AMOUNT": float(args["amount"]),
                "TRANS_HOUR": int(args["hour"]),
                "final_risk_score": float(scored["final_risk_score"]),
                "Risk_Segment": scored["Risk_Segment"],
                "model_fraud_probability": float(scored["model_fraud_probability"]),
                "ML_Pred": int(scored["ML_Pred"]),
                "Rule_Detected": int(scored["Rule_Detected"]),
                "Business_Action": scored["Business_Action"],
                "score_fraud_rule": float(scored["score_fraud_rule"]),
                "score_behavioral_instability": float(scored["score_behavioral_instability"]),
                "score_aml_risk": float(scored["score_aml_risk"]),
                "Reason_Code_Details": scored["Reason_Code_Details"],
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    st.set_page_config(page_title="Vòng 3 Fraud Demo", layout="wide")
    st.title("Vòng 3 Fraud Prevention Demo")
    st.caption("Demo này đọc output sinh trực tiếp từ `Vong_3_EAZII_2.ipynb`, chạy local bằng dữ liệu thật trong `Processed_Data/`.")

    try:
        master, feature_importance, shap_importance, shap_local, metrics = load_colab_demo_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("macOS/Linux: bash scripts/run_demo.sh ./Processed_Data")
        st.code("Windows: powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -DataDir .\\Processed_Data")
        return

    impact = metrics["business_impact"]
    model_metrics = metrics.get("model_metrics", {})
    test_fraction = model_metrics.get("test_fraction")
    if test_fraction is None and model_metrics.get("test_rows") is not None and model_metrics.get("training_rows") is not None:
        total_model_rows = model_metrics["test_rows"] + model_metrics["training_rows"]
        test_fraction = model_metrics["test_rows"] / total_model_rows if total_model_rows else 0.0
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Customer 360 rows", f"{metrics['row_counts']['customer_360_rows']:,}")
    r2.metric("Weak-fraud customers", f"{metrics['row_counts']['weak_fraud_customers']:,}")
    r3.metric("Critical", f"{metrics['row_counts']['critical_customers']:,}")
    protected_amount = impact.get("test_total_saved_amount", impact.get("protected_avg_transaction_amount", 0))
    r4.metric("Protected amount", f"{protected_amount:,.0f}")

    tab_transaction, tab_overview, tab_actions, tab_case, tab_xai, tab_colab_flow = st.tabs(
        ["Transaction test", "Overview", "Business actions", "Case advisor", "xAI", "Vòng 3 flow"]
    )

    with tab_overview:
        left, right = st.columns([1, 1])
        with left:
            risk_dist = pd.Series(metrics["risk_segment_distribution"]).reset_index()
            risk_dist.columns = ["Risk_Segment", "customers"]
            fig = px.bar(
                risk_dist,
                x="Risk_Segment",
                y="customers",
                color="Risk_Segment",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"},
                title="Vòng 3 risk segment distribution",
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.scatter(
                master.sample(min(8000, len(master)), random_state=42),
                x="txn_count",
                y="total_trans_amount",
                color="Risk_Segment",
                hover_data=["CUSTOMER_NUMBER", "Business_Action"],
                title="Customer flow scale: frequency vs total amount",
                color_discrete_map={"Low": "#9CA3AF", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"},
            )
            fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
        st.write("**Top Vòng 3 review queue**")
        queue_cols = ["CUSTOMER_NUMBER", "final_risk_score", "Risk_Segment", "Business_Action", "Reason_Code_Details"]
        st.dataframe(master.sort_values("final_risk_score", ascending=False)[queue_cols].head(50), use_container_width=True, hide_index=True)

    with tab_colab_flow:
        st.subheader("Vòng 3 notebook logic mapped into local web demo")
        st.markdown(
            """
1. Execute the business cells directly from `Vong_3_EAZII_2.ipynb`.
2. Clean real CSV files from `Processed_Data/` into `outputs/vong3_2_cleaned/`.
3. Build `df_baseline_trans_env.csv`, `df_baseline_behavior.csv`, and `df_baseline_financial.csv`.
4. Merge them into `Customer_360_Master_Data.csv`.
5. Compute Vòng 3 IQR amount threshold and five rule flags.
6. Create `Fraud`, `Risk_Segment`, `final_risk_score`, and `Reason_Code_Details`.
7. Train the Vòng 3 XGBoost layer exactly as written in the notebook with an 80/20 train/test split (test set = 20%).
8. Apply the hybrid matrix to create `Business_Action`.
9. Measure business impact on the held-out 20% test set, matching the newest notebook KPI logic.
10. Export figures, feature importance, SHAP evidence, and business-impact metrics.
"""
        )
        if test_fraction is not None:
            st.metric("Model test set", f"{test_fraction:.0%}")
        fig_cols = st.columns(2)
        figure_files = [(f"Exact Colab figure {i:02d}", f"colab_exact_figure_{i:02d}.png") for i in range(1, 9)]
        for idx, (label, filename) in enumerate(figure_files):
            with fig_cols[idx % 2]:
                st.write(f"**{label}**")
                st.image(str(FIGURES_DIR / filename))

    with tab_actions:
        st.subheader("Business action matrix")
        test_action_dist = impact.get("test_action_distribution", {})
        action_source = test_action_dist if test_action_dist else metrics["business_action_distribution"]
        action_dist = pd.Series(action_source).reset_index()
        action_dist.columns = ["Business_Action", "customers"]
        fig = px.bar(
            action_dist,
            x="customers",
            y="Business_Action",
            orientation="h",
            color="Business_Action",
            title="Hybrid action output from newest notebook" + (" (20% test set)" if test_action_dist else ""),
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        if test_action_dist:
            c1.metric("Test BLOCK", f"{impact.get('test_blocked_accounts', 0):,}")
            c2.metric("Test eKYC", f"{impact.get('test_ekyc_accounts', 0):,}")
            c3.metric("Test Watchlist", f"{impact.get('test_watchlist_accounts', 0):,}")
            c4.metric("BLOCK share of test", f"{impact.get('test_block_rate_of_all_cases', 0):.1%}")
            st.caption("Newest notebook measures operational impact on the held-out 20% test set.")
            st.write(
                f"Test challenge coverage vs weak fraud: **{impact.get('test_challenge_coverage_against_rule_label', 0):.1%}** | "
                f"Test saved amount: **{impact.get('test_total_saved_amount', 0):,.0f} VND** | "
                f"Test review/eKYC rate: **{impact.get('test_ekyc_rate_of_all_cases', 0):.1%}**"
            )
        else:
            c1.metric("Blocked", f"{impact['blocked_accounts']:,}")
            c2.metric("Step-up", f"{impact['step_up_accounts']:,}")
            c3.metric("Watchlist", f"{impact['watchlist_accounts']:,}")
            rates = weak_label_action_rates(master, impact)
            c4.metric("Block rate vs weak fraud", f"{rates['blocked']:.1%}")
            st.caption("Block rate is hard prevention. Challenge coverage = Block + Step-up; review coverage also includes Watchlist.")
            st.write(
                f"Challenge coverage: **{rates['challenge']:.1%}** | "
                f"Step-up only: **{rates['step_up']:.1%}** | "
                f"Review coverage incl. watchlist: **{rates['review']:.1%}**"
            )
        st.write("**Model validation against Vòng 3 weak labels**")
        st.json(metrics["model_metrics"])

    with tab_transaction:
        st.subheader("Real-time transaction test")
        st.caption("Chấm một giao dịch mới theo 2 mode: lấy baseline Customer 360 từ data, hoặc tự nhập baseline cho khách/giao dịch thực tế mới.")
        demo_customer = "639362"
        if not master["CUSTOMER_NUMBER"].eq(demo_customer).any():
            low_customers = master.loc[master["Risk_Segment"].eq("Low"), "CUSTOMER_NUMBER"]
            demo_customer = str(low_customers.iloc[0] if not low_customers.empty else master.iloc[0]["CUSTOMER_NUMBER"])
        test_mode = st.radio(
            "Baseline mode",
            ["Use existing Customer 360 from data", "Manually enter new customer baseline"],
            horizontal=True,
            help="Mode 1 dùng CUSTOMER_NUMBER đã có trong dữ liệu. Mode 2 tự nhập hồ sơ hành vi/tài chính tối giản, không cần khách hàng tồn tại trong data.",
        )
        with st.form("transaction_test_form"):
            c1, c2, c3 = st.columns(3)
            if test_mode == "Use existing Customer 360 from data":
                customer_id = c1.text_input("CUSTOMER_NUMBER", value=demo_customer)
            else:
                customer_id = c1.text_input("New customer/reference ID", value="NEW_CUSTOMER_001")
            amount = c2.number_input("Transaction amount (VND)", min_value=0.0, value=150_000_000.0, step=1_000_000.0)
            hour = c3.number_input("Transaction hour", min_value=0, max_value=23, value=1, step=1)

            manual_overrides: dict[str, object] = {}
            if test_mode == "Manually enter new customer baseline":
                st.write("**Manual Customer 360 baseline**")
                st.caption("Nhập thói quen trước giao dịch hiện tại. Nếu không chắc, để gần giá trị mặc định/0 để hệ thống coi là khách ít lịch sử.")
                b1, b2, b3, b4 = st.columns(4)
                manual_overrides["txn_count"] = b1.number_input("Historical txn count", min_value=0.0, value=20.0, step=1.0)
                manual_overrides["avg_trans_amount"] = b2.number_input("Avg txn amount", min_value=0.0, value=2_000_000.0, step=500_000.0)
                manual_overrides["max_trans_amount"] = b3.number_input("Historical max amount", min_value=0.0, value=20_000_000.0, step=1_000_000.0)
                manual_overrides["avg_balance_ca"] = b4.number_input("Avg CA balance", min_value=0.0, value=30_000_000.0, step=1_000_000.0)
                b5, b6, b7, b8 = st.columns(4)
                manual_overrides["unique_devices"] = b5.number_input("Trusted devices", min_value=0.0, value=1.0, step=1.0)
                manual_overrides["unique_ips"] = b6.number_input("Trusted IPs", min_value=0.0, value=1.0, step=1.0)
                manual_overrides["beneficiary_count"] = b7.number_input("Known beneficiaries", min_value=0.0, value=2.0, step=1.0)
                manual_overrides["max_inactive_gap"] = b8.number_input("Max inactive days", min_value=0.0, value=10.0, step=1.0)
                b9, b10, b11, b12 = st.columns(4)
                manual_overrides["night_txn_ratio"] = b9.number_input("Night txn ratio", min_value=0.0, max_value=1.0, value=0.05, step=0.05)
                manual_overrides["outside_bank_ratio"] = b10.number_input("Outside-bank ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
                manual_overrides["password_change_count"] = b11.number_input("Past security changes", min_value=0.0, value=0.0, step=1.0)
                manual_overrides["max_cic_overdue_days"] = b12.number_input("Max overdue days", min_value=0.0, value=0.0, step=1.0)
                b13, b14, b15 = st.columns(3)
                manual_overrides["total_app_activities"] = b13.number_input("Historical app activities", min_value=0.0, value=80.0, step=5.0)
                manual_overrides["max_daily_activity"] = b14.number_input("Max daily app activity", min_value=0.0, value=8.0, step=1.0)
                manual_overrides["card_utilization_ratio"] = b15.number_input("Card utilization ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

            c4, c5, c6 = st.columns(3)
            device_mode = c4.selectbox("Device status", ["Known device", "New device", "Shared/farm device"], index=1)
            ip_mode = c5.selectbox("IP status", ["Known IP", "New IP/proxy"], index=1)
            beneficiary_mode = c6.selectbox(
                "Beneficiary status",
                ["Known beneficiary", "New external beneficiary", "Merchant/zero beneficiary"],
                index=1,
            )

            c7, c8, c9 = st.columns(3)
            outside_bank = c7.checkbox("Outside-bank transfer", value=True)
            recent_security_change = c8.checkbox("Recent password/security change", value=True)
            txns_today = c9.number_input("Transactions today after this txn", min_value=1, max_value=200, value=8, step=1)
            submitted = st.form_submit_button("Run prevention flow")

        if submitted:
            if test_mode == "Use existing Customer 360 from data":
                found = find_customer(customer_id, master)
                if found.empty:
                    st.error("Không tìm thấy CUSTOMER_NUMBER trong Customer 360 exact output.")
                    found_profile = None
                else:
                    found_profile = found.iloc[0]
            else:
                found_profile = build_manual_customer_profile(master, customer_id, manual_overrides)

            if found_profile is not None:
                with st.spinner("Training/loading runtime XGBoost model and scoring transaction..."):
                    model, feature_cols = train_runtime_transaction_model(master)
                    threshold_amount = float(metrics.get("thresholds", {}).get("THRESHOLD_AMOUNT") or 0.0)
                    scored, audit = simulate_transaction(
                        found_profile,
                        float(amount),
                        int(hour),
                        device_mode,
                        ip_mode,
                        beneficiary_mode,
                        bool(outside_bank),
                        bool(recent_security_change),
                        int(txns_today),
                        threshold_amount,
                        model,
                        feature_cols,
                    )

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Final risk score", f"{float(scored['final_risk_score']):.0f}")
                s2.metric("Risk segment", scored["Risk_Segment"])
                s3.metric("ML probability", f"{float(scored['model_fraud_probability']):.1%}")
                s4.metric("Action", scored["Business_Action"])
                st.caption(
                    "Runtime risk score is severity-based inside the same 3 cause branches, so it can move between 0-100 instead of only 0/20/30/50."
                )
                if test_mode == "Manually enter new customer baseline":
                    st.info("Mode tự nhập: baseline không lấy từ CUSTOMER_NUMBER có sẵn; ML probability vẫn dùng model học từ dữ liệu contest để so mẫu hành vi.")

                st.write("**End-to-end execution trace**")
                st.dataframe(audit, use_container_width=True, hide_index=True)
                st.write("**Reason code**")
                st.warning(scored["Reason_Code_Details"])
                st.write("**Updated rule flags and scores**")
                display_cols = [
                    "rule_behavior_device",
                    "rule_ato",
                    "rule_money_mule",
                    "rule_dormant_active",
                    "rule_night_anomaly",
                    "score_fraud_rule",
                    "score_behavioral_instability",
                    "score_aml_risk",
                    "model_fraud_probability",
                    "runtime_amount_ratio",
                    "avg_trans_amount",
                    "unique_devices",
                    "unique_ips",
                    "night_txn_ratio",
                    "burst_max",
                ]
                st.dataframe(scored[[col for col in display_cols if col in scored.index]].to_frame("value"), use_container_width=True)

        st.divider()
        st.subheader("Batch CSV transaction test")
        st.caption(
            "Upload một CSV nhiều giao dịch để chấm cùng lúc. Có thể dùng `CUSTOMER_NUMBER` để lấy baseline từ data, "
            "hoặc tự cung cấp baseline bằng các cột như `txn_count`, `avg_trans_amount`, `unique_devices`. "
            "Các cột giao dịch có thể dùng tên như "
            "`TRANS_AMOUNT`, `TRANS_HOUR`, `device_status`, `ip_status`, `beneficiary_status`, `outside_bank`, "
            "`recent_security_change`, `txns_today`."
        )
        sample_batch = pd.DataFrame(
            [
                {
                    "CUSTOMER_NUMBER": demo_customer,
                    "TRANS_AMOUNT": 150_000_000,
                    "TRANS_HOUR": 1,
                    "device_status": "New device",
                    "ip_status": "New IP/proxy",
                    "beneficiary_status": "New external beneficiary",
                    "outside_bank": 1,
                    "recent_security_change": 1,
                    "txns_today": 8,
                },
                {
                    "CUSTOMER_NUMBER": demo_customer,
                    "TRANS_AMOUNT": 500_000,
                    "TRANS_HOUR": 14,
                    "device_status": "Known device",
                    "ip_status": "Known IP",
                    "beneficiary_status": "Known beneficiary",
                    "outside_bank": 0,
                    "recent_security_change": 0,
                    "txns_today": 1,
                },
                {
                    "CUSTOMER_NUMBER": "NEW_CUSTOMER_001",
                    "TRANS_AMOUNT": 250_000_000,
                    "TRANS_HOUR": 2,
                    "is_new_device": 1,
                    "is_new_ip": 1,
                    "is_new_beneficiary": 1,
                    "outside_bank": 1,
                    "recent_security_change": 1,
                    "txns_today": 10,
                    "txn_count": 12,
                    "avg_trans_amount": 2_000_000,
                    "max_trans_amount": 20_000_000,
                    "unique_devices": 1,
                    "unique_ips": 1,
                    "beneficiary_count": 2,
                    "night_txn_ratio": 0.05,
                    "outside_bank_ratio": 0.2,
                    "max_inactive_gap": 45,
                    "password_change_count": 0,
                    "max_cic_overdue_days": 0,
                },
            ]
        )
        st.download_button(
            "Download sample CSV template",
            sample_batch.to_csv(index=False).encode("utf-8-sig"),
            file_name="sample_batch_transactions.csv",
            mime="text/csv",
        )
        uploaded_csv = st.file_uploader("Upload transaction CSV", type=["csv"])
        if uploaded_csv is not None:
            try:
                upload_df = pd.read_csv(uploaded_csv)
                with st.spinner("Scoring uploaded transactions..."):
                    model, feature_cols = train_runtime_transaction_model(master)
                    threshold_amount = float(metrics.get("thresholds", {}).get("THRESHOLD_AMOUNT") or 0.0)
                    batch_result = score_batch_transactions(upload_df, master, threshold_amount, model, feature_cols)
                st.success(f"Scored {len(batch_result):,} rows.")
                b1, b2, b3, b4 = st.columns(4)
                valid_rows = batch_result["status"].eq("SCORED")
                b1.metric("Rows scored", f"{int(valid_rows.sum()):,}")
                b2.metric("Block/Hold", f"{int(batch_result['Business_Action'].eq('CRITICAL: BLOCK IMMEDIATELY').sum()):,}")
                b3.metric("Step-up/eKYC", f"{int(batch_result['Business_Action'].eq('WARNING: REQUIRE STEP-UP EKYC/OTP').sum()):,}")
                b4.metric("Avg risk score", f"{batch_result.loc[valid_rows, 'final_risk_score'].mean():.1f}" if valid_rows.any() else "N/A")
                st.dataframe(
                    batch_result.sort_values("final_risk_score", ascending=False, na_position="last"),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download scored results CSV",
                    batch_result.to_csv(index=False).encode("utf-8-sig"),
                    file_name="scored_batch_transactions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Không đọc/chấm được CSV: {exc}")

    with tab_case:
        st.subheader("Customer risk advisor")
        default_customer = str(master.sort_values("final_risk_score", ascending=False).iloc[0]["CUSTOMER_NUMBER"])
        query = st.text_input("Enter CUSTOMER_NUMBER", value=default_customer)
        found = find_customer(query, master)
        if found.empty:
            st.warning("Không tìm thấy CUSTOMER_NUMBER trong Vòng 3 Customer 360.")
        else:
            row = found.iloc[0]
            st.info(advisor_text(row))
            show_case(row)

    with tab_xai:
        st.subheader("xAI from Vòng 3 model")
        st.caption("SHAP được sinh từ model học weak label của notebook Vòng 3. Đây không phải confirmed fraud label.")
        c1, c2 = st.columns(2)
        with c1:
            show_image_if_exists(FIGURES_DIR / "colab_exact_figure_13.png", "Feature importance từ cell 19")
        with c2:
            show_image_if_exists(FIGURES_DIR / "colab_exact_figure_15.png", "SHAP waterfall từ cell 21")
        st.write("**Top model features**")
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)
        st.write("**Top SHAP features**")
        st.dataframe(shap_importance.head(20), use_container_width=True, hide_index=True)
        if not shap_local.empty:
            st.write("**Local SHAP explanations**")
            st.dataframe(shap_local.head(50), use_container_width=True, hide_index=True)
        waterfall = FIGURES_DIR / "colab_shap_waterfall.png"
        if waterfall.exists():
            show_image_if_exists(waterfall, "SHAP waterfall")


if __name__ == "__main__":
    main()
