from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")


@st.cache_data(show_spinner=False)
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    top_queue = pd.read_csv(OUTPUT_DIR / "top_review_queue.csv")
    customers = pd.read_csv(OUTPUT_DIR / "customer_risk_summary.csv")
    root_cause = pd.read_csv(OUTPUT_DIR / "root_cause_summary.csv")
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    top_queue["transaction_row_id"] = top_queue["transaction_row_id"].astype(str)
    top_queue["CUSTOMER_NUMBER"] = top_queue["CUSTOMER_NUMBER"].astype(str)
    customers["CUSTOMER_NUMBER"] = customers["CUSTOMER_NUMBER"].astype(str)
    return top_queue, customers, root_cause, metrics


def advisor_text(row: pd.Series) -> str:
    return (
        f"Transaction {row['transaction_row_id']} is {row['risk_band']} "
        f"({float(row['risk_score_0_100']):.2f}/100). Main branch: {row['primary_cause_branch']}. "
        f"Reasons: {row['top_reasons']}. Recommended action: {row['recommended_action']}"
    )


def show_transaction(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk band", row["risk_band"])
    c2.metric("Risk score", f"{float(row['risk_score_0_100']):.2f}")
    c3.metric("Amount", f"{float(row['TRANS_AMOUNT']):,.0f}")
    c4.metric("Hour", int(row["TRANS_HOUR"]))
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
    top_queue, customers, root_cause, metrics = load_demo_data()

    st.title("Fraud Risk Advisor")
    st.caption("Real-data demo for G'Contest 2026. This is a risk-ranking framework, not a confirmed-fraud classifier.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions scored", f"{metrics['row_counts']['transactions_scored']:,}")
    m2.metric("High/Critical", f"{metrics['review_queue']['high_or_critical_transactions']:,}")
    m3.metric("Critical", f"{metrics['review_queue']['critical_transactions']:,}")
    m4.metric("Customers impacted", f"{metrics['review_queue']['customers_with_high_or_critical']:,}")

    tab_overview, tab_case, tab_chat, tab_xai = st.tabs(["Overview", "Case review", "Advisor chat", "xAI"])

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
