from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - depends on local libomp availability.
    XGBRegressor = None

from fraud_pipeline import MODEL_FEATURES


def train_surrogate(sample: pd.DataFrame, random_state: int):
    x = sample[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = sample["risk_score_0_100"].clip(0, 100)
    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.86,
            colsample_bytree=0.82,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        model = RandomForestRegressor(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=30,
            random_state=random_state,
            n_jobs=-1,
        )
    model.fit(x, y)
    return model


def build_shap_outputs(sample: pd.DataFrame, output_dir: Path, figures_dir: Path, random_state: int, explain_rows: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    train_sample = sample.sample(min(len(sample), 60_000), random_state=random_state)
    model = train_surrogate(train_sample, random_state=random_state)

    explain_pool = pd.concat(
        [
            sample.sort_values("risk_score_0_100", ascending=False).head(explain_rows // 2),
            sample.sample(min(len(sample), explain_rows), random_state=random_state),
        ],
        ignore_index=True,
    ).drop_duplicates("transaction_row_id").head(explain_rows)
    x_explain = explain_pool[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_explain)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    top_rows = explain_pool.sort_values("risk_score_0_100", ascending=False).head(50).copy()
    local_records = []
    top_index_positions = [explain_pool.index.get_loc(idx) for idx in top_rows.index]
    for row_position, (_, row) in zip(top_index_positions, top_rows.iterrows()):
        values = shap_values[row_position]
        order = np.argsort(np.abs(values))[::-1][:5]
        drivers = [f"{MODEL_FEATURES[i]}={row[MODEL_FEATURES[i]]:.3g} (SHAP {values[i]:+.2f})" for i in order]
        local_records.append(
            {
                "transaction_row_id": row["transaction_row_id"],
                "CUSTOMER_NUMBER": row["CUSTOMER_NUMBER"],
                "risk_score_0_100": row["risk_score_0_100"],
                "risk_band": row["risk_band"],
                "primary_cause_branch": row["primary_cause_branch"],
                "top_shap_drivers": "; ".join(drivers),
            }
        )
    pd.DataFrame(local_records).to_csv(output_dir / "shap_local_explanations.csv", index=False)

    plt.figure(figsize=(9, 6))
    top_importance = importance.head(15).iloc[::-1]
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"], color="#2F6B8F")
    plt.title(f"SHAP feature importance for {model.__class__.__name__} surrogate")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_feature_importance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, x_explain, max_display=15, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    predictions = model.predict(train_sample[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0))
    mae = float(np.abs(predictions - train_sample["risk_score_0_100"]).mean())
    corr = float(np.corrcoef(predictions, train_sample["risk_score_0_100"])[0, 1])
    metrics = {
        "method": f"{model.__class__.__name__} surrogate + SHAP TreeExplainer",
        "target": "Final hybrid prevention risk_score_0_100 after the Rule + ML decision matrix",
        "training_rows": int(len(train_sample)),
        "explained_rows": int(len(x_explain)),
        "surrogate_mae": mae,
        "surrogate_correlation": corr,
        "top_features": importance.head(10).to_dict(orient="records"),
        "note": "This explains the final hybrid prevention score with SHAP on a tree surrogate. The underlying labels are rule-derived weak labels, not confirmed fraud labels.",
    }
    (output_dir / "shap_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHAP xAI outputs for the real-data fraud risk framework.")
    parser.add_argument("--sample", type=Path, default=Path("outputs/xai_feature_sample.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--random-state", type=int, default=20260524)
    parser.add_argument("--explain-rows", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sample.exists():
        raise SystemExit(f"Missing {args.sample}. Run `python src/fraud_pipeline.py` first.")
    sample = pd.read_csv(args.sample)
    metrics = build_shap_outputs(sample, args.output_dir, args.figures_dir, args.random_state, args.explain_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
