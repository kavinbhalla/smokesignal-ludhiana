import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

FEATURES = [
    "pm25_today",
    "pm25_max_today",
    "pm25_lag1",
    "pm25_roll3",
    "fire_count_150km",
    "total_frp_mw",
    "upwind_frp_index",
    "upwind_frp_roll2",
    "nw_cluster_fires",
    "sw_cluster_fires",
    "fire_count_roll3",
    "wind_speed_mean",
    "wind_dir_deg",
    "blh_mean",
    "blh_min",
    "stagnation_index",
    "temp_mean",
    "humidity_mean",
    "precip_sum",
    "month",
    "day_of_year",
]


def pm25_to_naqi_category(pm25):
    """Maps 24-hour PM2.5 (ug/m3) to India CPCB National AQI category."""
    if pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderate"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very Poor"
    else:
        return "Severe"


def train_and_evaluate():
    df = pd.read_csv("data/merged_daily_ludhiana.csv")
    df["date_dt"] = pd.to_datetime(df["date"])

    # Drop only the final row where tomorrow's ground truth isn't known yet for training/testing
    df_valid = df.dropna(subset=FEATURES + ["target_pm25_tomorrow"]).copy()

    # Train on 2019-2023; Test on 2024 and 2025 seasons (Week 3 specification)
    train_df = df_valid[df_valid["date_dt"].dt.year <= 2023]
    test_df = df_valid[
        (df_valid["date_dt"].dt.year.isin([2024, 2025]))
        & (df_valid["date_dt"].dt.month.isin([10, 11]))
    ].copy()

    # Fallback if dataset is smaller
    if len(test_df) < 10:
        split_idx = int(len(df_valid) * 0.8)
        train_df = df_valid.iloc[:split_idx]
        test_df = df_valid.iloc[split_idx:].copy()

    X_train, y_train = train_df[FEATURES], train_df["target_pm25_tomorrow"]
    X_test, y_test = test_df[FEATURES], test_df["target_pm25_tomorrow"]

    # 1. Persistence Baseline: Tomorrow = Today
    y_pred_baseline = test_df["pm25_today"].values
    mae_base = mean_absolute_error(y_test, y_pred_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    cat_true = [pm25_to_naqi_category(v) for v in y_test]
    cat_base = [pm25_to_naqi_category(v) for v in y_pred_baseline]
    acc_base = accuracy_score(cat_true, cat_base)

    # 2. XGBoost Forecast Model
    model = xgb.XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.5,
        reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_xgb = model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    cat_xgb = [pm25_to_naqi_category(v) for v in y_pred_xgb]
    acc_xgb = accuracy_score(cat_true, cat_xgb)

    mae_gain_pct = ((mae_base - mae_xgb) / mae_base) * 100.0

    print("\n=== HONEST EVALUATION ON 2024–2025 STUBBLE BURNING SEASONS ===")
    print(f"Baseline (Tomorrow = Today) | MAE: {mae_base:.2f} ug/m3 | RMSE: {rmse_base:.2f} | AQI Cat Acc: {acc_base*100:.1f}%")
    print(f"SmokeSignal XGBoost         | MAE: {mae_xgb:.2f} ug/m3 | RMSE: {rmse_xgb:.2f} | AQI Cat Acc: {acc_xgb*100:.1f}%")
    print(f"Net Error Reduction vs Baseline: {mae_gain_pct:+.1f}%\n")

    # 3. SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Impact on Next-Day PM2.5 in Ludhiana")
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png", dpi=180)
    plt.close()

    # 4. Predict Tomorrow's Live Forecast from the Most Recent Row
    latest_row = df.iloc[[-1]].copy()
    X_latest = latest_row[FEATURES].ffill().fillna(0)
    pred_tomorrow = float(model.predict(X_latest)[0])
    today_pm25 = float(latest_row["pm25_today"].values[0])

    latest_shap = explainer(X_latest)
    shap_contribs = pd.Series(latest_shap.values[0], index=FEATURES).sort_values(key=abs, ascending=False)
    top_drivers = [
        {"feature": k, "value": round(float(X_latest[k].values[0]), 2), "shap_impact_ugm3": round(float(v), 2)}
        for k, v in shap_contribs.head(5).items()
    ]

    with open("artifacts/metadata.json", "r") as f:
        meta = json.load(f)

    forecast_payload = {
        "forecast_generated_on": latest_row["date"].values[0],
        "today_pm25": round(today_pm25, 1),
        "today_category": pm25_to_naqi_category(today_pm25),
        "tomorrow_pm25_xgb": round(pred_tomorrow, 1),
        "tomorrow_category_xgb": pm25_to_naqi_category(pred_tomorrow),
        "tomorrow_pm25_baseline": round(today_pm25, 1),
        "fire_count_150km_today": int(latest_row["fire_count_150km"].values[0]),
        "nw_cluster_fires_today": int(latest_row["nw_cluster_fires"].values[0]),
        "wind_speed_kmh": round(float(latest_row["wind_speed_mean"].values[0]), 1),
        "wind_dir_deg": round(float(latest_row["wind_dir_deg"].values[0]), 0),
        "blh_m": round(float(latest_row["blh_mean"].values[0]), 0),
        "best_lag_hours": meta.get("best_lag_hours", 10),
        "metrics_2024_2025": {
            "baseline_mae": round(mae_base, 2),
            "xgb_mae": round(mae_xgb, 2),
            "baseline_rmse": round(rmse_base, 2),
            "xgb_rmse": round(rmse_xgb, 2),
            "xgb_r2": round(r2_xgb, 3),
            "baseline_aqi_acc_pct": round(acc_base * 100, 1),
            "xgb_aqi_acc_pct": round(acc_xgb * 100, 1),
            "mae_improvement_pct": round(mae_gain_pct, 1),
        },
        "top_shap_drivers": top_drivers,
    }

    with open("artifacts/latest_forecast.json", "w") as f:
        json.dump(forecast_payload, f, indent=2)

    model.save_model("artifacts/xgb_smokesignal.json")
    print("Saved `artifacts/latest_forecast.json`, `artifacts/shap_summary.png`, and `artifacts/xgb_smokesignal.json`.")


if __name__ == "__main__":
    train_and_evaluate()
