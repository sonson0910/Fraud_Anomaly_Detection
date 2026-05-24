from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_scores(outputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(outputs_dir / "transaction_risk_scores.csv")
    customers = pd.read_csv(outputs_dir / "customer_risk_summary.csv")
    transactions["transaction_row_id"] = transactions["transaction_row_id"].astype(str)
    customers["CUSTOMER_NUMBER"] = customers["CUSTOMER_NUMBER"].astype(str)
    transactions["CUSTOMER_NUMBER"] = transactions["CUSTOMER_NUMBER"].astype(str)
    return transactions, customers


def explain_transaction(row: pd.Series) -> str:
    return f"""Risk Advisor - Transaction {row['transaction_row_id']}

Customer: {row['CUSTOMER_NUMBER']}
Risk band: {row['risk_band']} ({float(row['risk_score_0_100']):.2f}/100)
Primary cause branch: {row.get('primary_cause_branch', 'N/A')}
Transaction: {row['TRANS_LV1']} / {row['TRANS_LV2']} at hour {int(row['TRANS_HOUR'])} with amount {float(row['TRANS_AMOUNT']):,.0f}

Why flagged:
{row['top_reasons']}

Recommended action:
{row['recommended_action']}

Risk-officer wording:
Hệ thống không kết luận chắc chắn đây là fraud. Giao dịch được đưa vào mức rủi ro này vì lệch khỏi baseline của chính khách hàng, có tín hiệu truy cập/giao dịch/mạng lưới đáng chú ý, hoặc được Isolation Forest xem là bất thường so với toàn bộ tập giao dịch thật.
"""


def explain_customer(customer_row: pd.Series, transactions: pd.DataFrame) -> str:
    customer_id = str(customer_row["CUSTOMER_NUMBER"])
    top = transactions.loc[transactions["CUSTOMER_NUMBER"].eq(customer_id)].head(5)
    top_lines = "\n".join(
        f"- {row.transaction_row_id}: {row.risk_band} {float(row.risk_score_0_100):.2f}/100 | "
        f"{row.primary_cause_branch} | {row.top_reasons}"
        for row in top.itertuples()
    )
    return f"""Risk Advisor - Customer {customer_id}

Customer risk band: {customer_row['customer_risk_band']}
Max risk score: {float(customer_row['max_risk_score']):.2f}/100
Average risk score: {float(customer_row['avg_risk_score']):.2f}/100
High/Critical transactions: {int(customer_row['high_or_critical_count'])}
Main cause branch: {customer_row.get('main_cause_branch', 'N/A')}
Main reason: {customer_row['main_reason']}

Top risky transactions:
{top_lines}

Recommended action:
{customer_row['customer_recommended_action']}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local risk-advisor demo for the real-data G'Contest fraud framework.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--transaction-id", help="Explain a specific transaction_row_id.")
    parser.add_argument("--customer-id", help="Explain a specific CUSTOMER_NUMBER.")
    parser.add_argument("--top-critical", type=int, default=0, help="Print the top N riskiest transactions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions, customers = load_scores(args.outputs_dir)
    if args.transaction_id:
        match = transactions.loc[transactions["transaction_row_id"].eq(str(args.transaction_id))]
        if match.empty:
            raise SystemExit(f"Transaction not found: {args.transaction_id}")
        print(explain_transaction(match.iloc[0]))
        return
    if args.customer_id:
        match = customers.loc[customers["CUSTOMER_NUMBER"].eq(str(args.customer_id))]
        if match.empty:
            raise SystemExit(f"Customer not found: {args.customer_id}")
        print(explain_customer(match.iloc[0], transactions))
        return
    if args.top_critical:
        for _, row in transactions.head(args.top_critical).iterrows():
            print(explain_transaction(row))
            print("=" * 80)
        return
    raise SystemExit("Provide --transaction-id, --customer-id, or --top-critical.")


if __name__ == "__main__":
    main()
