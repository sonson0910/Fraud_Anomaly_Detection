from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    top_queue = pd.read_csv(OUTPUT_DIR / "top_review_queue.csv")
    customers = pd.read_csv(OUTPUT_DIR / "customer_risk_summary.csv")
    customer_360 = pd.read_csv(OUTPUT_DIR / "customer_360_baseline.csv")
    root_cause = pd.read_csv(OUTPUT_DIR / "root_cause_summary.csv")
    insights = pd.read_csv(OUTPUT_DIR / "insight_summary.csv")
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    top_queue["transaction_row_id"] = top_queue["transaction_row_id"].astype(str)
    top_queue["CUSTOMER_NUMBER"] = top_queue["CUSTOMER_NUMBER"].astype(str)
    customers["CUSTOMER_NUMBER"] = customers["CUSTOMER_NUMBER"].astype(str)
    customer_360["CUSTOMER_NUMBER"] = customer_360["CUSTOMER_NUMBER"].astype(str)
    return top_queue, customers, customer_360, root_cause, insights, metrics


def advisor_text(row: pd.Series) -> str:
    return (
        f"Transaction {row['transaction_row_id']} is {row['risk_band']} "
        f"({float(row['risk_score_0_100']):.2f}/100). Main branch: {row['primary_cause_branch']}. "
        f"Prevention action: {row.get('prevention_action', 'N/A')}. "
        f"Reasons: {row['top_reasons']}. Recommended action: {row['recommended_action']}"
    )


def show_transaction(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk band", row["risk_band"])
    c2.metric("Risk score", f"{float(row['risk_score_0_100']):.2f}")
    c3.metric("Amount", f"{float(row['TRANS_AMOUNT']):,.0f}")
    c4.metric("Action", row.get("prevention_action", "N/A"))
    st.write("**Primary cause branch**")
    st.write(row["primary_cause_branch"])
    st.write("**Why flagged**")
    st.write(row["top_reasons"])
    st.write("**Recommended action**")
    st.write(row["recommended_action"])


def answer_query(query: str, top_queue: pd.DataFrame, customers: pd.DataFrame) -> str:
    ids = re.findall(r"\d+", query)
    if not ids:
        return "Nhập `CUSTOMER_NUMBER` hoặc `transaction_row_id` để advisor giải thích case cụ thể."
    candidate = ids[0]
    trx = top_queue.loc[top_queue["transaction_row_id"].eq(candidate)]
    if not trx.empty:
        return advisor_text(trx.iloc[0])
    cust = customers.loc[customers["CUSTOMER_NUMBER"].eq(candidate)]
    if not cust.empty:
        row = cust.iloc[0]
        return (
            f"Customer {candidate} is {row['customer_risk_band']}. "
            f"Max risk score {float(row['max_risk_score']):.2f}/100, "
            f"High/Critical transactions {int(row['high_or_critical_count'])}. "
            f"Main branch: {row['main_cause_branch']}. Main reason: {row['main_reason']}. "
            f"Recommended action: {row['customer_recommended_action']}"
        )
    return f"Không tìm thấy `{candidate}` trong demo queue. Hãy thử một ID trong top review queue."


def main() -> None:
    st.set_page_config(page_title="G'Contest Fraud Risk Advisor", layout="wide")
    top_queue, customers, customer_360, root_cause, insights, metrics = load_demo_data()

    st.title("Fraud Risk Advisor")
    st.caption("Real-data demo for G'Contest 2026. Rule-derived weak labels train a supervised fraud prevention model.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions scored", f"{metrics['row_counts']['transactions_scored']:,}")
    m2.metric("High/Critical", f"{metrics['review_queue']['high_or_critical_transactions']:,}")
    m3.metric("Critical", f"{metrics['review_queue']['critical_transactions']:,}")
    m4.metric("Prevention coverage", f"{metrics['prevention_impact']['prevention_coverage_against_rule_labels']:.1%}")

    tab_overview, tab_insights, tab_customer360, tab_prevention, tab_case, tab_chat, tab_xai = st.tabs(
        ["Overview", "Data insights", "Customer 360", "Prevention impact", "Case review", "Advisor chat", "xAI"]
    )

    with tab_overview:
        left, right = st.columns([1.2, 1])
        with left:
            fig = px.bar(
                root_cause,
                x="high_or_critical_transactions",
                y="primary_cause_branch",
                orientation="h",
                title="High/Critical transactions by root-cause branch",
                color_discrete_sequence=["#2F6B8F"],
            )
            fig.update_layout(yaxis_title="", xaxis_title="High/Critical transactions", height=360)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            band_counts = pd.Series(metrics["risk_band_distribution"]).reset_index()
            band_counts.columns = ["risk_band", "count"]
            fig = px.bar(
                band_counts,
                x="risk_band",
                y="count",
                title="Risk band distribution",
                color="risk_band",
                color_discrete_map={"Low": "#88A868", "Medium": "#E0A72E", "High": "#C46243", "Critical": "#8E2D2D"},
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_queue.head(30), use_container_width=True, hide_index=True)

    with tab_insights:
        st.subheader("Data-driven insights")
        st.caption("Các insight này được tính trực tiếp từ output scored population, không phải nhận xét thủ công.")
        for row in insights.itertuples():
            st.markdown(f"**{row.title}**")
            st.write(row.evidence)
            st.caption(row.business_meaning)
        st.divider()
        fig_cols = st.columns(2)
        insight_figures = [
            ("Root cause x hybrid matrix", "root_cause_hybrid_heatmap.png"),
            ("IQR breach lift", "iqr_breach_lift.png"),
            ("Time risk heatmap", "time_risk_heatmap.png"),
            ("Network exposure bubble", "network_exposure_bubble.png"),
            ("Customer 360 risk heatmap", "customer360_risk_heatmap.png"),
        ]
        for idx, (label, filename) in enumerate(insight_figures):
            with fig_cols[idx % 2]:
                st.write(f"**{label}**")
                st.image(str(OUTPUT_DIR / "figures" / filename))

    with tab_customer360:
        st.subheader("Customer 360 baseline")
        st.caption("Một khách hàng một dòng: Transactional, Financial, Environmental và Behavioral baseline với rolling window 30/60/90 ngày.")
        selected_customer = st.selectbox("CUSTOMER_NUMBER", customer_360["CUSTOMER_NUMBER"].head(5000), index=0)
        profile = customer_360.loc[customer_360["CUSTOMER_NUMBER"].eq(str(selected_customer))]
        if not profile.empty:
            row = profile.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("30d txns", f"{float(row['rolling_30d_txn_count']):,.0f}")
            c2.metric("90d avg amount", f"{float(row['rolling_90d_amount_avg']):,.0f}")
            c3.metric("Trusted devices", f"{int(row['environmental_trusted_device_count']):,}")
            c4.metric("Worst credit group", f"{int(row['financial_worst_credit_risk_group'])}")
            st.dataframe(profile, use_container_width=True, hide_index=True)
        st.write("**Sample profiles**")
        st.dataframe(customer_360.head(50), use_container_width=True, hide_index=True)

    with tab_prevention:
        st.subheader("Prevention impact")
        impact = metrics["prevention_impact"]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Block/Hold", f"{impact['blocked_transactions']:,}")
        p2.metric("Step-up", f"{impact['step_up_transactions']:,}")
        p3.metric("Protected amount", f"{impact['protected_amount_block_or_step_up']:,.0f}")
        p4.metric("Coverage vs weak labels", f"{impact['prevention_coverage_against_rule_labels']:.1%}")
        model_metrics = metrics["supervised_model_metrics"]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Recall vs weak labels", f"{model_metrics['validation_recall_at_high_threshold']:.1%}")
        q2.metric("Precision vs weak labels", f"{model_metrics['validation_precision_at_high_threshold']:.1%}")
        q3.metric("False positive rate", f"{model_metrics['validation_false_positive_rate_at_high_threshold']:.2%}")
        q4.metric("PR-AUC vs weak labels", f"{model_metrics['validation_pr_auc']:.3f}")
        st.write("**Validation confusion matrix against weak labels**")
        st.json(model_metrics["validation_confusion_matrix_at_high_threshold"])
        action_counts = top_queue["prevention_action"].value_counts().reset_index()
        action_counts.columns = ["prevention_action", "count"]
        fig = px.bar(
            action_counts,
            x="prevention_action",
            y="count",
            title="Top queue by prevention action",
            color="prevention_action",
            color_discrete_sequence=["#8E2D2D", "#C46243", "#E0A72E", "#88A868"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Hybrid decision matrix - full scored population**")
        hybrid_counts = pd.DataFrame(
            [
                {"hybrid_decision": "Rule+ML alert: Block/Hold", "count": impact["rule_and_ml_alert_transactions"]},
                {"hybrid_decision": "Rule-only alert: Step-up/eKYC", "count": impact["rule_only_alert_transactions"]},
                {"hybrid_decision": "ML-only alert: Special watchlist", "count": impact["ml_only_alert_transactions"]},
                {"hybrid_decision": "No alert: Allow", "count": impact["no_alert_transactions"]},
            ]
        )
        st.dataframe(hybrid_counts, use_container_width=True, hide_index=True)
        st.caption("Coverage is measured against rule-derived weak labels, not confirmed fraud outcomes.")

    with tab_case:
        st.subheader("Review a transaction or customer")
        col1, col2 = st.columns([1, 1])
        transaction_id = col1.text_input("transaction_row_id", value=str(top_queue.iloc[0]["transaction_row_id"]))
        customer_id = col2.text_input("CUSTOMER_NUMBER", value=str(top_queue.iloc[0]["CUSTOMER_NUMBER"]))
        match = top_queue.loc[top_queue["transaction_row_id"].eq(str(transaction_id))]
        if not match.empty:
            show_transaction(match.iloc[0])
        else:
            st.warning("Transaction is not in top demo queue.")
        cust = customers.loc[customers["CUSTOMER_NUMBER"].eq(str(customer_id))]
        if not cust.empty:
            st.write("**Customer profile**")
            st.dataframe(cust.head(1), use_container_width=True, hide_index=True)

    with tab_chat:
        st.subheader("Chatbot-style advisor")
        query = st.text_input("Ask about a customer or transaction", value=f"Explain transaction {top_queue.iloc[0]['transaction_row_id']}")
        st.info(answer_query(query, top_queue, customers))

    with tab_xai:
        st.subheader("SHAP explainability")
        shap_importance = OUTPUT_DIR / "shap_feature_importance.csv"
        shap_local = OUTPUT_DIR / "shap_local_explanations.csv"
        if shap_importance.exists() and shap_local.exists():
            st.image(str(OUTPUT_DIR / "figures" / "shap_feature_importance.png"))
            st.dataframe(pd.read_csv(shap_importance).head(20), use_container_width=True, hide_index=True)
            st.write("**Local SHAP explanations for top cases**")
            st.dataframe(pd.read_csv(shap_local).head(20), use_container_width=True, hide_index=True)
        else:
            st.warning("Run `python src/xai_shap_engine.py` to create SHAP outputs.")


if __name__ == "__main__":
    main()
