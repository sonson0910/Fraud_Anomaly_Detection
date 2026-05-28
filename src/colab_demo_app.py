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


CLEANED_DIR = Path(os.environ.get("COLAB_CLEANED_DIR", "outputs/colab_exact_cleaned"))
FIGURES_DIR = Path(os.environ.get("COLAB_FIGURES_DIR", "outputs/colab_exact_figures"))


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
        raise FileNotFoundError("Missing Colab outputs. Run `scripts/run_demo.sh` or `scripts/run_demo.ps1` first.")
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
        f"Customer {row['CUSTOMER_NUMBER']} is {row['Risk_Segment']} with Colab risk score "
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
    c2.metric("Colab risk score", f"{float(row['final_risk_score']):.0f}")
    c3.metric("Max amount", f"{float(row['max_trans_amount']):,.0f}")
    c4.metric("Action", row["Business_Action"])
    st.write("**Reason code from Colab notebook logic**")
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
    return " | ".join(reasons)


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

    candidate["score_fraud_rule"] = float((candidate["rule_ato"] == 1) or (candidate["rule_behavior_device"] == 1)) * 50.0
    candidate["score_behavioral_instability"] = float(
        (candidate["rule_dormant_active"] == 1) or (candidate["rule_night_anomaly"] == 1)
    ) * 30.0
    candidate["score_aml_risk"] = float(candidate["rule_money_mule"] == 1) * 20.0
    candidate["final_risk_score"] = (
        candidate["score_fraud_rule"] + candidate["score_behavioral_instability"] + candidate["score_aml_risk"]
    )
    candidate["Risk_Segment"] = risk_segment(float(candidate["final_risk_score"]))
    candidate["Fraud"] = int(candidate["final_risk_score"] > 0)

    model_input = pd.DataFrame([{col: candidate.get(col, 0) for col in feature_cols}])
    model_input = model_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    candidate["ML_Pred"] = int(model.predict(model_input)[0])
    candidate["Rule_Detected"] = int(candidate["Fraud"] == 1)
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
            {"step": "4. Weak-label ML model", "output": f"ML_Pred={candidate['ML_Pred']} from XGBoost runtime model."},
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


def main() -> None:
    st.set_page_config(page_title="Colab Fraud Demo", layout="wide")
    st.title("Colab Fraud Prevention Demo")
    st.caption("Demo này đọc output sinh từ logic trong `Another copy of Welcome To Colab`, chạy local bằng dữ liệu cũ trong `Processed_Data/`.")

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
        ["Transaction test", "Overview", "Business actions", "Case advisor", "xAI", "Colab flow"]
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
                title="Colab risk segment distribution",
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
        st.write("**Top Colab review queue**")
        queue_cols = ["CUSTOMER_NUMBER", "final_risk_score", "Risk_Segment", "Business_Action", "Reason_Code_Details"]
        st.dataframe(master.sort_values("final_risk_score", ascending=False)[queue_cols].head(50), use_container_width=True, hide_index=True)

    with tab_colab_flow:
        st.subheader("Notebook Colab logic mapped into local web demo")
        st.markdown(
            """
1. Execute the business cells directly from `Another copy of Welcome To Colab`.
2. Clean real CSV files from `Processed_Data/` into `outputs/colab_exact_cleaned/`.
3. Build `df_baseline_trans_env.csv`, `df_baseline_behavior.csv`, and `df_baseline_financial.csv`.
4. Merge them into `Customer_360_Master_Data.csv`.
5. Compute Colab IQR amount threshold and five rule flags.
6. Create `Fraud`, `Risk_Segment`, `final_risk_score`, and `Reason_Code_Details`.
7. Train the Colab XGBoost layer exactly as written in the notebook with an 80/20 train/test split (test set = 20%).
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
        st.write("**Model validation against Colab weak labels**")
        st.json(metrics["model_metrics"])

    with tab_transaction:
        st.subheader("Real-time transaction test")
        st.caption("Nhập một giao dịch mới; hệ thống cập nhật tạm Customer 360 rồi chạy rule engine, ML và action matrix.")
        demo_customer = "639362"
        if not master["CUSTOMER_NUMBER"].eq(demo_customer).any():
            low_customers = master.loc[master["Risk_Segment"].eq("Low"), "CUSTOMER_NUMBER"]
            demo_customer = str(low_customers.iloc[0] if not low_customers.empty else master.iloc[0]["CUSTOMER_NUMBER"])
        st.info(
            "Demo nhanh: giữ CIF mặc định, bấm `Run prevention flow` để thấy một giao dịch mới đi qua "
            "baseline 360 -> rule engine -> XGBoost -> action matrix."
        )
        with st.form("transaction_test_form"):
            c1, c2, c3 = st.columns(3)
            customer_id = c1.text_input("CUSTOMER_NUMBER", value=demo_customer)
            amount = c2.number_input("Transaction amount (VND)", min_value=0.0, value=150_000_000.0, step=1_000_000.0)
            hour = c3.number_input("Transaction hour", min_value=0, max_value=23, value=1, step=1)

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
            found = find_customer(customer_id, master)
            if found.empty:
                st.error("Không tìm thấy CUSTOMER_NUMBER trong Customer 360 exact output.")
            else:
                with st.spinner("Training/loading runtime XGBoost model and scoring transaction..."):
                    model, feature_cols = train_runtime_transaction_model(master)
                    threshold_amount = float(metrics.get("thresholds", {}).get("THRESHOLD_AMOUNT") or 0.0)
                    scored, audit = simulate_transaction(
                        found.iloc[0],
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
                s3.metric("ML prediction", "Fraud-like" if int(scored["ML_Pred"]) else "Normal-like")
                s4.metric("Action", scored["Business_Action"])

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
                    "avg_trans_amount",
                    "unique_devices",
                    "unique_ips",
                    "night_txn_ratio",
                    "burst_max",
                ]
                st.dataframe(scored[[col for col in display_cols if col in scored.index]].to_frame("value"), use_container_width=True)

    with tab_case:
        st.subheader("Customer risk advisor")
        default_customer = str(master.sort_values("final_risk_score", ascending=False).iloc[0]["CUSTOMER_NUMBER"])
        query = st.text_input("Enter CUSTOMER_NUMBER", value=default_customer)
        found = find_customer(query, master)
        if found.empty:
            st.warning("Không tìm thấy CUSTOMER_NUMBER trong Colab Customer 360.")
        else:
            row = found.iloc[0]
            st.info(advisor_text(row))
            show_case(row)

    with tab_xai:
        st.subheader("xAI from Colab model")
        st.caption("SHAP được sinh từ model học weak label của notebook Colab. Đây không phải confirmed fraud label.")
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
