from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260524
START_DATE = pd.Timestamp("2019-01-01")
END_DATE = pd.Timestamp("2026-05-31")
MONTHS = pd.date_range(START_DATE.normalize().replace(day=1), END_DATE.normalize().replace(day=1), freq="MS")


CUSTOMER_COLUMNS = [
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
]

TRANSACTION_COLUMNS = [
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
]

ACTIVITY_COLUMNS = [
    "ACTIVITY_DATE",
    "DAY_OF_WEEK",
    "ACTIVITY_HOUR",
    "ACTIVITY_NO",
    "CUSTOMER_NUMBER",
    "ACTIVITY_NAME",
]

DEPOSIT_COLUMNS = [
    "MONTH",
    "COUNT_CA_ACCT",
    "AVG_CA_BALANCE",
    "COUNT_TD_ACCT",
    "AVG_TD_BALANCE",
    "CUSTOMER_NUMBER",
]

LENDING_COLUMNS = [
    "MONTH",
    "COUNT_OF_LOAN",
    "AVG_LOAN_AMOUNT",
    "CUSTOMER_NUMBER",
    "OVERDUE_LENDING",
    "TERM_LENDING",
    "INTEREST_RATE",
]

CARD_COLUMNS = [
    "MONTH",
    "COUNT_CREDITCARD",
    "COUNT_DEBITCARD",
    "CUSTOMER_NUMBER",
    "OVERDUE_CREDIT",
    "LIMIT_AMT",
    "OUTSTANDING_BALANCE",
]


@dataclass(frozen=True)
class GenerationConfig:
    n_customers: int = 3_000
    target_transactions: int = 150_000
    target_activities: int = 220_000
    anomaly_rate: float = 0.018
    seed: int = SEED


def _customer_ids(n: int) -> np.ndarray:
    return np.array([f"CUS{i:06d}" for i in range(1, n + 1)])


def _random_dates(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.Series:
    day_span = (end.normalize() - start.normalize()).days
    offsets = rng.integers(0, day_span + 1, size=n)
    return pd.Series(start.normalize() + pd.to_timedelta(offsets, unit="D"))


def _sample_weighted(rng: np.random.Generator, values: list[str], probs: list[float], n: int) -> np.ndarray:
    probs_arr = np.array(probs, dtype=float)
    probs_arr = probs_arr / probs_arr.sum()
    return rng.choice(values, size=n, p=probs_arr)


def generate_customers(config: GenerationConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = _customer_ids(config.n_customers)
    ages = np.clip(rng.normal(36, 11, size=config.n_customers).round(), 18, 72).astype(int)
    dob = pd.Timestamp("2026-01-01") - pd.to_timedelta(ages * 365 + rng.integers(0, 365, config.n_customers), unit="D")
    client_create = _random_dates(rng, pd.Timestamp("2012-01-01"), pd.Timestamp("2026-01-31"), config.n_customers)
    ib_lag = rng.integers(0, 1200, size=config.n_customers)
    ib_register = pd.Series(client_create.values) + pd.to_timedelta(ib_lag, unit="D")
    ib_register = ib_register.clip(upper=pd.Timestamp("2026-05-01"))

    profile = pd.DataFrame(
        {
            "CUSTOMER_NUMBER": ids,
            "income_band": _sample_weighted(rng, ["mass", "mass_affluent", "affluent"], [0.58, 0.32, 0.10], config.n_customers),
            "risk_appetite": np.clip(rng.beta(2, 5, config.n_customers), 0.05, 0.95),
            "digital_intensity": np.clip(rng.gamma(2.5, 1.2, config.n_customers), 0.25, 8.0),
            "home_region": _sample_weighted(
                rng,
                ["HN", "HCM", "DN", "HP", "CT", "BD", "DNA", "KH", "NT", "QNI"],
                [0.27, 0.28, 0.09, 0.06, 0.06, 0.07, 0.06, 0.04, 0.04, 0.03],
                config.n_customers,
            ),
            "primary_device": [f"DEV_{rng.integers(100000, 999999)}" for _ in range(config.n_customers)],
            "age": ages,
        }
    )

    customers = pd.DataFrame(
        {
            "CUSTOMER_NUMBER": ids,
            "CLIENT_SEX": _sample_weighted(rng, ["F", "M", "U"], [0.51, 0.48, 0.01], config.n_customers),
            "CLIENT_CREATE_DATE": client_create.dt.strftime("%Y-%m-%d"),
            "DATE_OF_BIRTH": pd.Series(dob).dt.strftime("%Y-%m-%d"),
            "STAFF": _sample_weighted(rng, ["N", "Y"], [0.975, 0.025], config.n_customers),
            "IB_REGISTER_DATE": ib_register.dt.strftime("%Y-%m-%d"),
            "EB_REGISTER_CHANNEL": _sample_weighted(rng, ["Mobile App", "Branch", "Web", "Partner"], [0.52, 0.28, 0.15, 0.05], config.n_customers),
            "SMS": _sample_weighted(rng, ["Y", "N"], [0.72, 0.28], config.n_customers),
            "VERIFY_METHOD": _sample_weighted(rng, ["Smart OTP", "SMS OTP", "Token", "Biometric"], [0.48, 0.24, 0.12, 0.16], config.n_customers),
            "OCCUPATION_GROUP": _sample_weighted(
                rng,
                ["Office Worker", "Business Owner", "Student", "Retired", "Freelancer", "Factory Worker", "Professional"],
                [0.34, 0.17, 0.12, 0.06, 0.11, 0.10, 0.10],
                config.n_customers,
            ),
            "EDUCATION_LEVEL": _sample_weighted(rng, ["High School", "College", "Bachelor", "Postgraduate", "Unknown"], [0.19, 0.21, 0.45, 0.10, 0.05], config.n_customers),
            "MARITAL_STATUS": _sample_weighted(rng, ["Single", "Married", "Divorced", "Unknown"], [0.42, 0.49, 0.04, 0.05], config.n_customers),
        }
    )
    return customers[CUSTOMER_COLUMNS], profile


def _transaction_hour(rng: np.random.Generator, n: int) -> np.ndarray:
    daytime = rng.choice(np.arange(8, 22), size=n, p=np.array([0.03, 0.04, 0.06, 0.08, 0.10, 0.09, 0.08, 0.08, 0.08, 0.09, 0.09, 0.07, 0.06, 0.05]))
    night = rng.choice(np.r_[0:8, 22:24], size=n)
    return np.where(rng.random(n) < 0.94, daytime, night)


def _transaction_types(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    lv1 = _sample_weighted(rng, ["Transfer", "Payment", "Card", "Deposit", "Loan"], [0.48, 0.25, 0.13, 0.10, 0.04], n)
    lv2 = np.empty(n, dtype=object)
    options = {
        "Transfer": (["Internal Transfer", "External Transfer", "Fast Transfer", "Scheduled Transfer"], [0.34, 0.40, 0.20, 0.06]),
        "Payment": (["Bill Payment", "QR Payment", "E-commerce", "Top-up"], [0.30, 0.32, 0.24, 0.14]),
        "Card": (["Card Purchase", "Cash Withdrawal", "Card Repayment"], [0.55, 0.24, 0.21]),
        "Deposit": (["CASA Deposit", "Term Deposit Open", "Term Deposit Close"], [0.64, 0.28, 0.08]),
        "Loan": (["Loan Repayment", "Loan Disbursement"], [0.78, 0.22]),
    }
    for category, (values, probs) in options.items():
        mask = lv1 == category
        lv2[mask] = _sample_weighted(rng, values, probs, int(mask.sum()))
    return lv1, lv2


def generate_transactions(
    config: GenerationConfig,
    rng: np.random.Generator,
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_base = config.target_transactions - int(config.target_transactions * config.anomaly_rate)
    customer_probs = profile["digital_intensity"].to_numpy(dtype=float)
    customer_probs = customer_probs / customer_probs.sum()
    if n_base >= len(profile):
        guaranteed = np.arange(len(profile))
        sampled = rng.choice(np.arange(len(profile)), size=n_base - len(profile), p=customer_probs)
        customer_idx = np.concatenate([guaranteed, sampled])
        rng.shuffle(customer_idx)
    else:
        customer_idx = rng.choice(np.arange(len(profile)), size=n_base, replace=False)
    customer = profile.iloc[customer_idx].reset_index(drop=True)

    dates = _random_dates(rng, START_DATE, END_DATE, n_base)
    hours = _transaction_hour(rng, n_base)
    lv1, lv2 = _transaction_types(rng, n_base)

    income_multiplier = customer["income_band"].map({"mass": 1.0, "mass_affluent": 2.2, "affluent": 5.0}).to_numpy()
    type_multiplier = pd.Series(lv1).map({"Transfer": 2.0, "Payment": 0.65, "Card": 0.95, "Deposit": 3.1, "Loan": 2.4}).to_numpy()
    amount = rng.lognormal(mean=13.3, sigma=0.78, size=n_base) * income_multiplier * type_multiplier
    amount = np.clip(amount, 20_000, 950_000_000).round(-3).astype(int)
    trans_no = np.clip(rng.poisson(1.15, size=n_base) + 1, 1, 9)

    device = customer["primary_device"].to_numpy().astype(object)
    secondary_device = np.array([f"DEV_{rng.integers(100000, 999999)}" for _ in range(n_base)], dtype=object)
    device = np.where(rng.random(n_base) < 0.08, secondary_device, device)
    ip_region = np.where(rng.random(n_base) < 0.86, customer["home_region"].to_numpy(), rng.choice(["HN", "HCM", "DN", "HP", "CT", "BD", "DNA", "KH", "NT", "QNI"], size=n_base))

    merchants = np.array([f"MRC_{rng.integers(1000, 9999)}" for _ in range(n_base)], dtype=object)
    beneficiary = rng.choice(profile["CUSTOMER_NUMBER"].to_numpy(), size=n_base)
    beneficiary = np.where(lv1 == "Transfer", beneficiary, "NOT_APPLICABLE")

    base = pd.DataFrame(
        {
            "TRXN_LV1": lv1,
            "TRXN_LV2": lv2,
            "TRANS_DATE": dates.dt.strftime("%Y-%m-%d"),
            "DAY_OF_WEEK": dates.dt.day_name(),
            "TRANS_HOUR": hours,
            "TRANS_NO": trans_no,
            "TRANS_AMOUNT": amount,
            "CUSTOMER_NUMBER": customer["CUSTOMER_NUMBER"].to_numpy(),
            "IP_Address_Proxy": [f"IP_{x}_{rng.integers(1000, 9999)}" for x in ip_region],
            "Device_ID_Hash": device,
            "Device_OS": _sample_weighted(rng, ["iOS", "Android", "Windows", "macOS"], [0.33, 0.54, 0.09, 0.04], n_base),
            "Merchant_ID_Masked": merchants,
            "Beneficiary_CUSTOMER_NUMBER": beneficiary,
        }
    )

    anomalies, truth = _inject_anomalies(config, rng, profile)
    transactions = pd.concat([base, anomalies], ignore_index=True)
    transactions = transactions.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    transactions["transaction_row_id"] = [f"TRX{i:07d}" for i in range(1, len(transactions) + 1)]

    truth = transactions[
        [
            "transaction_row_id",
            "CUSTOMER_NUMBER",
            "TRANS_DATE",
            "TRANS_HOUR",
            "TRANS_AMOUNT",
            "Device_ID_Hash",
            "IP_Address_Proxy",
        ]
    ].merge(
        truth,
        left_on=["CUSTOMER_NUMBER", "TRANS_DATE", "TRANS_HOUR", "TRANS_AMOUNT", "Device_ID_Hash", "IP_Address_Proxy"],
        right_on=["CUSTOMER_NUMBER", "TRANS_DATE", "TRANS_HOUR", "TRANS_AMOUNT", "Device_ID_Hash", "IP_Address_Proxy"],
        how="left",
    )[["transaction_row_id", "CUSTOMER_NUMBER", "IS_SYNTHETIC_ANOMALY", "ANOMALY_TYPE", "INJECTION_REASON"]]
    truth["IS_SYNTHETIC_ANOMALY"] = truth["IS_SYNTHETIC_ANOMALY"].fillna(0).astype(int)
    truth["ANOMALY_TYPE"] = truth["ANOMALY_TYPE"].fillna("NORMAL")
    truth["INJECTION_REASON"] = truth["INJECTION_REASON"].fillna("Normal synthetic banking behavior")

    return transactions[["transaction_row_id"] + TRANSACTION_COLUMNS], truth


def _inject_anomalies(config: GenerationConfig, rng: np.random.Generator, profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(config.target_transactions * config.anomaly_rate)
    anomaly_types = _sample_weighted(
        rng,
        ["ACCOUNT_TAKEOVER", "UNAUTHORIZED_TRANSFER", "MONEY_LAUNDERING_PROXY", "BEHAVIORAL_OUTLIER"],
        [0.30, 0.28, 0.24, 0.18],
        n,
    )
    customer_probs = np.sqrt(profile["digital_intensity"].to_numpy(dtype=float))
    customer_probs = customer_probs / customer_probs.sum()
    customer_idx = rng.choice(np.arange(len(profile)), size=n, p=customer_probs)
    customer = profile.iloc[customer_idx].reset_index(drop=True)
    dates = _random_dates(rng, START_DATE + pd.Timedelta(days=45), END_DATE, n)
    hours = rng.choice(np.r_[0:6, 22:24], size=n)
    lv1 = np.where(anomaly_types == "BEHAVIORAL_OUTLIER", rng.choice(["Payment", "Card"], size=n), "Transfer")
    lv2 = np.select(
        [
            anomaly_types == "ACCOUNT_TAKEOVER",
            anomaly_types == "UNAUTHORIZED_TRANSFER",
            anomaly_types == "MONEY_LAUNDERING_PROXY",
        ],
        ["Fast Transfer", "External Transfer", "External Transfer"],
        default="E-commerce",
    )
    income_multiplier = customer["income_band"].map({"mass": 1.0, "mass_affluent": 2.2, "affluent": 5.0}).to_numpy()
    amount = rng.lognormal(mean=15.6, sigma=0.55, size=n) * income_multiplier
    laundering_mask = anomaly_types == "MONEY_LAUNDERING_PROXY"
    amount[laundering_mask] = rng.choice([9_900_000, 19_900_000, 49_900_000, 99_000_000], size=int(laundering_mask.sum()))
    outlier_mask = anomaly_types == "BEHAVIORAL_OUTLIER"
    amount[outlier_mask] = amount[outlier_mask] * rng.uniform(2.0, 4.0, size=int(outlier_mask.sum()))
    amount = np.clip(amount, 8_000_000, 1_850_000_000).round(-3).astype(int)
    trans_no = np.where(anomaly_types == "MONEY_LAUNDERING_PROXY", rng.integers(4, 10, size=n), rng.integers(1, 5, size=n))

    shared_ips = [f"IP_PROXY_{i:03d}" for i in range(1, 31)]
    shared_devices = [f"DEV_SHARED_{i:03d}" for i in range(1, 31)]
    new_device = np.array([f"DEV_NEW_{rng.integers(100000, 999999)}" for _ in range(n)], dtype=object)
    proxy_device = rng.choice(shared_devices, size=n)
    proxy_ip = rng.choice(shared_ips, size=n)
    ip = np.where(anomaly_types == "MONEY_LAUNDERING_PROXY", proxy_ip, proxy_ip)
    device = np.where(anomaly_types == "MONEY_LAUNDERING_PROXY", proxy_device, new_device)
    beneficiary = rng.choice(profile["CUSTOMER_NUMBER"].to_numpy(), size=n)

    reason_map = {
        "ACCOUNT_TAKEOVER": "New device/IP, unusual late-hour transfer, high-value amount",
        "UNAUTHORIZED_TRANSFER": "Sudden external beneficiary, amount far above customer baseline",
        "MONEY_LAUNDERING_PROXY": "Shared proxy device/IP and repeated round-value outward transfers",
        "BEHAVIORAL_OUTLIER": "Abnormal frequency and amount spike compared with customer history",
    }

    anomalies = pd.DataFrame(
        {
            "TRXN_LV1": lv1,
            "TRXN_LV2": lv2,
            "TRANS_DATE": dates.dt.strftime("%Y-%m-%d"),
            "DAY_OF_WEEK": dates.dt.day_name(),
            "TRANS_HOUR": hours,
            "TRANS_NO": trans_no,
            "TRANS_AMOUNT": amount,
            "CUSTOMER_NUMBER": customer["CUSTOMER_NUMBER"].to_numpy(),
            "IP_Address_Proxy": ip,
            "Device_ID_Hash": device,
            "Device_OS": _sample_weighted(rng, ["iOS", "Android", "Windows", "macOS"], [0.22, 0.67, 0.08, 0.03], n),
            "Merchant_ID_Masked": [f"MRC_RISK_{rng.integers(100, 999)}" for _ in range(n)],
            "Beneficiary_CUSTOMER_NUMBER": beneficiary,
        }
    )
    truth = anomalies[
        ["CUSTOMER_NUMBER", "TRANS_DATE", "TRANS_HOUR", "TRANS_AMOUNT", "Device_ID_Hash", "IP_Address_Proxy"]
    ].copy()
    truth["IS_SYNTHETIC_ANOMALY"] = 1
    truth["ANOMALY_TYPE"] = anomaly_types
    truth["INJECTION_REASON"] = pd.Series(anomaly_types).map(reason_map).to_numpy()
    return anomalies[TRANSACTION_COLUMNS], truth


def generate_activities(config: GenerationConfig, rng: np.random.Generator, profile: pd.DataFrame) -> pd.DataFrame:
    target = config.target_activities
    probs = profile["digital_intensity"].to_numpy(dtype=float)
    probs = probs / probs.sum()
    frames: list[pd.DataFrame] = []

    while sum(len(frame) for frame in frames) < target:
        current = sum(len(frame) for frame in frames)
        n = int((target - current) * 1.15) + 1_000
        customer = profile.iloc[rng.choice(np.arange(len(profile)), size=n, p=probs)].reset_index(drop=True)
        dates = _random_dates(rng, START_DATE, END_DATE, n)
        hours = np.where(rng.random(n) < 0.90, _transaction_hour(rng, n), rng.integers(0, 24, size=n))
        activities = _sample_weighted(
            rng,
            ["Login", "Balance Inquiry", "View Statement", "Add Beneficiary", "Change Password", "OTP Request", "Profile Update", "Transfer Form Open"],
            [0.34, 0.23, 0.12, 0.06, 0.03, 0.10, 0.04, 0.08],
            n,
        )
        activity_code = pd.Series(activities).map(
            {
                "Login": 1,
                "Balance Inquiry": 2,
                "View Statement": 3,
                "Profile Update": 4,
                "OTP Request": 5,
                "Transfer Form Open": 6,
                "Add Beneficiary": 7,
                "Change Password": 8,
            }
        )
        frames.append(
            pd.DataFrame(
                {
                    "ACTIVITY_DATE": dates.dt.strftime("%Y-%m-%d"),
                    "DAY_OF_WEEK": dates.dt.day_name(),
                    "ACTIVITY_HOUR": hours,
                    "ACTIVITY_NO": activity_code.astype(int),
                    "CUSTOMER_NUMBER": customer["CUSTOMER_NUMBER"].to_numpy(),
                    "ACTIVITY_NAME": activities,
                }
            )[ACTIVITY_COLUMNS]
        )
        frames = [pd.concat(frames, ignore_index=True).drop_duplicates().head(target)]

    return frames[0].head(target).reset_index(drop=True)


def generate_monthly_products(rng: np.random.Generator, profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    lending_rows = []
    card_rows = []
    for _, row in profile.iterrows():
        income_mult = {"mass": 1.0, "mass_affluent": 2.4, "affluent": 5.5}[row["income_band"]]
        base_balance = rng.lognormal(16.1, 0.65) * income_mult
        has_td = rng.random() < (0.16 + 0.08 * income_mult / 5.5)
        has_loan = rng.random() < (0.18 + 0.06 * row["risk_appetite"])
        has_credit = rng.random() < (0.24 + 0.10 * income_mult / 5.5)
        for month in MONTHS:
            balance_noise = rng.normal(1.0, 0.12)
            ca_balance = max(0, base_balance * balance_noise)
            rows.append(
                {
                    "MONTH": month.strftime("%Y-%m-%d"),
                    "COUNT_CA_ACCT": int(rng.choice([1, 1, 1, 2])),
                    "AVG_CA_BALANCE": int(round(ca_balance, -3)),
                    "COUNT_TD_ACCT": int(rng.integers(1, 3) if has_td else 0),
                    "AVG_TD_BALANCE": int(round(ca_balance * rng.uniform(1.5, 5.0), -3)) if has_td else 0,
                    "CUSTOMER_NUMBER": row["CUSTOMER_NUMBER"],
                }
            )
            loan_amount = int(round(rng.lognormal(17.1, 0.58) * income_mult, -3)) if has_loan else 0
            overdue = int(rng.choice([0, 0, 0, 3, 7, 15, 30, 60], p=[0.70, 0.08, 0.04, 0.05, 0.05, 0.04, 0.03, 0.01])) if has_loan else 0
            lending_rows.append(
                {
                    "MONTH": month.strftime("%Y-%m-%d"),
                    "COUNT_OF_LOAN": int(rng.integers(1, 3) if has_loan else 0),
                    "AVG_LOAN_AMOUNT": loan_amount,
                    "CUSTOMER_NUMBER": row["CUSTOMER_NUMBER"],
                    "OVERDUE_LENDING": overdue,
                    "TERM_LENDING": int(rng.choice([12, 24, 36, 60, 120, 180]) if has_loan else 0),
                    "INTEREST_RATE": round(float(rng.uniform(6.5, 14.5) if has_loan else 0), 2),
                }
            )
            limit = int(round(rng.lognormal(16.0, 0.5) * income_mult, -3)) if has_credit else 0
            utilization = rng.beta(2.0, 5.5) if has_credit else 0
            overdue_credit = int(rng.choice([0, 0, 3, 7, 15, 30, 60], p=[0.78, 0.08, 0.05, 0.04, 0.03, 0.015, 0.005])) if has_credit else 0
            card_rows.append(
                {
                    "MONTH": month.strftime("%Y-%m-%d"),
                    "COUNT_CREDITCARD": int(rng.integers(1, 3) if has_credit else 0),
                    "COUNT_DEBITCARD": int(rng.choice([1, 1, 1, 2])),
                    "CUSTOMER_NUMBER": row["CUSTOMER_NUMBER"],
                    "OVERDUE_CREDIT": overdue_credit,
                    "LIMIT_AMT": limit,
                    "OUTSTANDING_BALANCE": int(round(limit * utilization, -3)) if has_credit else 0,
                }
            )
    return (
        pd.DataFrame(rows)[DEPOSIT_COLUMNS],
        pd.DataFrame(lending_rows)[LENDING_COLUMNS],
        pd.DataFrame(card_rows)[CARD_COLUMNS],
    )


def write_outputs(config: GenerationConfig, output_dir: Path) -> None:
    rng = np.random.default_rng(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, profile = generate_customers(config, rng)
    transactions, truth = generate_transactions(config, rng, profile)
    activities = generate_activities(config, rng, profile)
    deposits, lending, cards = generate_monthly_products(rng, profile)

    outputs = {
        "Data_Customer.csv": customers,
        "Data_Transaction.csv": transactions[TRANSACTION_COLUMNS],
        "Data_Activity.csv": activities,
        "Data_Deposit.csv": deposits,
        "Data_Lending.csv": lending,
        "Data_Card.csv": cards,
        "synthetic_ground_truth.csv": truth,
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)

    metadata = pd.DataFrame(
        [
            {"table": name, "rows": len(df), "columns": len(df.columns)}
            for name, df in outputs.items()
        ]
    )
    metadata.to_csv(output_dir / "synthetic_metadata.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic G'Contest banking data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--customers", type=int, default=3_000)
    parser.add_argument("--transactions", type=int, default=150_000)
    parser.add_argument("--activities", type=int, default=220_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        n_customers=args.customers,
        target_transactions=args.transactions,
        target_activities=args.activities,
        seed=args.seed,
    )
    write_outputs(config, args.output_dir)
    print(f"Synthetic data written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
