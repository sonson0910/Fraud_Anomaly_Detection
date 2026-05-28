from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUSINESS_CELL_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
SHAP_CELL_INDICES = {20, 21, 22}
COLAB_RAW_DIR = "/content/drive/MyDrive/Gcontest/Processed_Data"
COLAB_CLEANED_DIR = "/content/drive/MyDrive/Gcontest/cleaned"
DEMO_LIGHT_COLS = [
    "CUSTOMER_NUMBER",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the business cells from the user's Colab notebook with only local path/runtime shims."
    )
    parser.add_argument("--notebook", default="Another copy of Welcome To Colab")
    parser.add_argument("--raw-dir", default="Processed_Data")
    parser.add_argument("--cleaned-dir", default="outputs/colab_exact_cleaned")
    parser.add_argument("--figures-dir", default="outputs/colab_exact_figures")
    parser.add_argument("--skip-shap", action="store_true", help="Skip the original SHAP cells if local runtime is too slow.")
    return parser.parse_args()


def notebook_cells(notebook_path: Path) -> dict[int, str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells: dict[int, str] = {}
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            cells[idx] = "".join(cell.get("source", []))
    return cells


def patch_colab_source(source: str, raw_dir: Path, cleaned_dir: Path) -> str:
    source = source.replace(COLAB_RAW_DIR, str(raw_dir))
    source = source.replace(COLAB_CLEANED_DIR, str(cleaned_dir))
    source = source.replace("df_360.fillna(0, inplace=True)", "df_360 = _colab_compat_fillna(df_360)")

    patched_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        if stripped == "from google.colab import drive":
            continue
        if "drive.mount(" in stripped:
            continue
        patched_lines.append(line)
    return "\n".join(patched_lines) + "\n"


def prepare_output_dirs(cleaned_dir: Path, figures_dir: Path) -> None:
    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def install_local_notebook_shims(figures_dir: Path) -> None:
    try:
        from IPython.display import display as ipy_display
    except Exception:
        ipy_display = print

    def display(obj: Any) -> None:
        ipy_display(obj)

    figure_counter = {"n": 0}

    def local_show(*_: Any, **__: Any) -> None:
        figure_counter["n"] += 1
        filename = figures_dir / f"colab_exact_figure_{figure_counter['n']:02d}.png"
        plt.gcf().savefig(filename, dpi=180, bbox_inches="tight")
        plt.close("all")
        print(f"    -> Saved figure: {filename}")

    plt.show = local_show
    globals()["display"] = display


def _colab_compat_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Match old Colab fillna behavior while running on stricter local pandas dtypes."""
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    other_cols = [col for col in result.columns if col not in numeric_cols]
    if len(numeric_cols):
        result[numeric_cols] = result[numeric_cols].fillna(0)
    if other_cols:
        result[other_cols] = result[other_cols].fillna("0")
    return result


def safe_series_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).to_dict().items()}


def export_demo_support_files(cleaned_dir: Path, namespace: dict[str, Any]) -> None:
    df_importance = namespace.get("df_importance")
    if isinstance(df_importance, pd.DataFrame):
        df_importance.to_csv(cleaned_dir / "colab_feature_importance.csv", index=False)
    elif "model_xgb" in namespace:
        try:
            booster = namespace["model_xgb"].get_booster()
            df_importance = pd.DataFrame(
                {
                    "Đặc trưng hành vi": booster.feature_names,
                    "Mức độ đóng góp (%)": namespace["model_xgb"].feature_importances_ * 100,
                }
            ).sort_values("Mức độ đóng góp (%)", ascending=False)
            df_importance.to_csv(cleaned_dir / "colab_feature_importance.csv", index=False)
        except Exception:
            pass

    df_shap_excel = namespace.get("df_shap_excel")
    if isinstance(df_shap_excel, pd.DataFrame):
        df_shap_excel.to_csv(cleaned_dir / "colab_shap_local_explanations.csv", index=False)

    shap_values = namespace.get("shap_values")
    x_all_input = namespace.get("X_test_clean")
    if not isinstance(x_all_input, pd.DataFrame):
        x_all_input = namespace.get("X_all_input")
    if shap_values is not None and isinstance(x_all_input, pd.DataFrame):
        try:
            values = shap_values.values if hasattr(shap_values, "values") else shap_values
            mean_abs = np.abs(values).mean(axis=0)
            shap_importance = pd.DataFrame(
                {"feature": x_all_input.columns, "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_importance.to_csv(cleaned_dir / "colab_shap_feature_importance.csv", index=False)
        except Exception:
            pass


def write_metrics(cleaned_dir: Path, raw_dir: Path, figures_dir: Path, namespace: dict[str, Any]) -> dict[str, Any]:
    master_path = cleaned_dir / "Customer_360_Master_Data.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Notebook did not create {master_path}")

    master = pd.read_csv(master_path, low_memory=False)
    demo_cols = [col for col in DEMO_LIGHT_COLS if col in master.columns]
    master[demo_cols].to_csv(cleaned_dir / "Customer_360_Demo_Light.csv", index=False)
    business_action = master["Business_Action"] if "Business_Action" in master.columns else pd.Series([], dtype=str)
    blocked = master.loc[business_action.eq("CRITICAL: BLOCK IMMEDIATELY")] if not business_action.empty else master.iloc[0:0]
    test_result = namespace.get("df_test_result")
    if not isinstance(test_result, pd.DataFrame):
        test_result = pd.DataFrame()

    cm = namespace.get("cm")
    model_type = type(namespace["model_xgb"]).__name__ if "model_xgb" in namespace else "unknown"
    model_metrics: dict[str, Any] = {"model_type": model_type}
    if cm is not None:
        model_metrics["confusion_matrix"] = np.asarray(cm).astype(int).tolist()
    if "y_test" in namespace:
        model_metrics["test_rows"] = int(len(namespace["y_test"]))
    if "X_train" in namespace:
        model_metrics["training_rows"] = int(len(namespace["X_train"]))
    if model_metrics.get("test_rows") is not None and model_metrics.get("training_rows") is not None:
        total_model_rows = model_metrics["test_rows"] + model_metrics["training_rows"]
        model_metrics["test_fraction"] = (
            float(model_metrics["test_rows"] / total_model_rows) if total_model_rows else 0.0
        )
    for source_name, metric_name in [
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("fpr", "false_positive_rate"),
        ("f1", "f1_score"),
        ("f2", "f2_score"),
    ]:
        if source_name in namespace:
            model_metrics[metric_name] = float(namespace[source_name])

    weak_fraud_count = int(master["Fraud"].sum()) if "Fraud" in master.columns else 0
    blocked_count = int(len(blocked))
    step_up_count = int(business_action.eq("WARNING: REQUIRE STEP-UP EKYC/OTP").sum()) if not business_action.empty else 0
    watchlist_count = int(business_action.eq("MONITOR: ADD TO SPECIAL WATCHLIST").sum()) if not business_action.empty else 0
    weak_fraud = master["Fraud"].eq(1) if "Fraud" in master.columns else pd.Series(False, index=master.index)
    blocked_mask = business_action.eq("CRITICAL: BLOCK IMMEDIATELY") if not business_action.empty else pd.Series(False, index=master.index)
    step_up_mask = business_action.eq("WARNING: REQUIRE STEP-UP EKYC/OTP") if not business_action.empty else pd.Series(False, index=master.index)
    watchlist_mask = business_action.eq("MONITOR: ADD TO SPECIAL WATCHLIST") if not business_action.empty else pd.Series(False, index=master.index)
    challenged_mask = blocked_mask | step_up_mask
    reviewed_mask = challenged_mask | watchlist_mask

    def weak_label_rate(mask: pd.Series) -> float:
        return float((weak_fraud & mask).sum() / weak_fraud_count) if weak_fraud_count else 0.0

    test_action = test_result["Action"].astype(str) if "Action" in test_result.columns else pd.Series([], dtype=str)
    test_block = test_action.eq("BLOCK")
    test_ekyc = test_action.eq("EKYC")
    test_watchlist = test_action.eq("WATCHLIST")
    test_allow = test_action.eq("ALLOW")
    test_count = int(len(test_result))
    test_weak = test_result["Fraud"].eq(1) if "Fraud" in test_result.columns else pd.Series(False, index=test_result.index)
    test_saved_amount = (
        float(pd.to_numeric(test_result.loc[test_block, "avg_balance_ca"], errors="coerce").fillna(0).sum())
        if "avg_balance_ca" in test_result.columns
        else 0.0
    )

    def test_rate(mask: pd.Series) -> float:
        return float(mask.sum() / test_count) if test_count else 0.0

    def test_weak_rate(mask: pd.Series) -> float:
        weak_count = int(test_weak.sum())
        return float((test_weak & mask).sum() / weak_count) if weak_count else 0.0

    metrics = {
        "runner": "src/run_colab_exact.py",
        "source_notebook": "Vòng_3_EAZII.ipynb / notebooks/colab_exact_business.ipynb",
        "execution_note": "Business cells were executed from the notebook source with only drive/path/shell-magic shims.",
        "data_source": str(raw_dir),
        "cleaned_output_dir": str(cleaned_dir),
        "figures_output_dir": str(figures_dir),
        "row_counts": {
            "customer_360_rows": int(len(master)),
            "weak_fraud_customers": weak_fraud_count,
            "high_or_critical_customers": int(master["Risk_Segment"].isin(["High", "Critical"]).sum())
            if "Risk_Segment" in master.columns
            else 0,
            "critical_customers": int(master["Risk_Segment"].eq("Critical").sum())
            if "Risk_Segment" in master.columns
            else 0,
        },
        "thresholds": {
            "THRESHOLD_AMOUNT": float(namespace["THRESHOLD_AMOUNT"]) if "THRESHOLD_AMOUNT" in namespace else None,
            "THRESHOLD_DEVICE": float(namespace["THRESHOLD_DEVICE"]) if "THRESHOLD_DEVICE" in namespace else None,
        },
        "risk_segment_distribution": safe_series_counts(master, "Risk_Segment"),
        "business_action_distribution": safe_series_counts(master, "Business_Action"),
        "business_impact": {
            "blocked_accounts": blocked_count,
            "step_up_accounts": step_up_count,
            "watchlist_accounts": watchlist_count,
            "allowed_accounts": int(business_action.eq("PASS: ALLOW TRANSACTION").sum()) if not business_action.empty else 0,
            "protected_avg_transaction_amount": float(blocked["avg_trans_amount"].sum())
            if "avg_trans_amount" in blocked.columns
            else 0.0,
            "max_single_avg_transaction_blocked": float(blocked["avg_trans_amount"].max())
            if "avg_trans_amount" in blocked.columns and len(blocked)
            else 0.0,
            "blocked_coverage_against_rule_label": weak_label_rate(blocked_mask),
            "step_up_coverage_against_rule_label": weak_label_rate(step_up_mask),
            "challenge_coverage_against_rule_label": weak_label_rate(challenged_mask),
            "review_coverage_against_rule_label": weak_label_rate(reviewed_mask),
            "test_rows": test_count,
            "test_action_distribution": safe_series_counts(test_result, "Action") if not test_result.empty else {},
            "test_blocked_accounts": int(test_block.sum()),
            "test_ekyc_accounts": int(test_ekyc.sum()),
            "test_watchlist_accounts": int(test_watchlist.sum()),
            "test_allowed_accounts": int(test_allow.sum()),
            "test_block_rate_of_all_cases": test_rate(test_block),
            "test_ekyc_rate_of_all_cases": test_rate(test_ekyc),
            "test_watchlist_rate_of_all_cases": test_rate(test_watchlist),
            "test_allow_rate_of_all_cases": test_rate(test_allow),
            "test_block_coverage_against_rule_label": test_weak_rate(test_block),
            "test_challenge_coverage_against_rule_label": test_weak_rate(test_block | test_ekyc),
            "test_total_saved_amount": test_saved_amount,
        },
        "model_metrics": model_metrics,
    }
    (cleaned_dir / "colab_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    notebook_path = Path(args.notebook)
    raw_dir = Path(args.raw_dir).resolve()
    cleaned_dir = Path(args.cleaned_dir).resolve()
    figures_dir = Path(args.figures_dir).resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    prepare_output_dirs(cleaned_dir, figures_dir)
    install_local_notebook_shims(figures_dir)

    cells = notebook_cells(notebook_path)
    namespace = globals()
    namespace.update({"__name__": "__colab_exact__", "display": globals()["display"]})

    for idx in BUSINESS_CELL_INDICES:
        if args.skip_shap and idx in SHAP_CELL_INDICES:
            print(f"\n--- SKIP COLAB CELL {idx} (SHAP) ---")
            continue
        source = cells.get(idx, "")
        if not source.strip():
            continue
        patched = patch_colab_source(source, raw_dir, cleaned_dir)
        print(f"\n--- RUN COLAB CELL {idx} ---")
        try:
            exec(compile(patched, f"{notebook_path.name}:cell-{idx}", "exec"), namespace)
        except Exception as exc:
            print(f"\nERROR while running notebook cell {idx}: {exc}", file=sys.stderr)
            raise

    export_demo_support_files(cleaned_dir, namespace)
    metrics = write_metrics(cleaned_dir, raw_dir, figures_dir, namespace)
    print("\n=== COLAB EXACT METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
