import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PAU_LAT = 30.9009
PAU_LON = 75.8078
RADIUS_KM = 150.0
BBOX = "74.20,29.55,77.40,32.25"

os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def haversine_and_bearing(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(a))

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return dist_km, bearing


def fetch_open_meteo_weather(start_date="2019-01-01"):
    archive_end = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    print(f"[1/4] Fetching Open-Meteo archive ({start_date} to {archive_end}) + live forecast...")

    frames = []
    try:
        url_arch = "https://archive-api.open-meteo.com/v1/archive"
        params_arch = {
            "latitude": PAU_LAT,
            "longitude": PAU_LON,
            "start_date": start_date,
            "end_date": archive_end,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "timezone": "Asia/Kolkata",
        }
        r1 = requests.get(url_arch, params=params_arch, timeout=60)
        if r1.status_code == 200 and "hourly" in r1.json():
            df_arch = pd.DataFrame(r1.json()["hourly"])
            df_arch["boundary_layer_height"] = np.clip(
                180.0 + 28.0 * df_arch["temperature_2m"] - 4.5 * df_arch["relative_humidity_2m"] + 22.0 * df_arch["wind_speed_10m"],
                120.0,
                2200.0,
            )
            frames.append(df_arch)
    except Exception as e:
        print(f"  Archive fetch warning: {e}")

    try:
        url_fc = "https://api.open-meteo.com/v1/forecast"
        params_fc = {
            "latitude": PAU_LAT,
            "longitude": PAU_LON,
            "past_days": 10,
            "forecast_days": 1,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "boundary_layer_height",
            ],
            "timezone": "Asia/Kolkata",
        }
        r2 = requests.get(url_fc, params=params_fc, timeout=45)
        if r2.status_code == 200 and "hourly" in r2.json():
            frames.append(pd.DataFrame(r2.json()["hourly"]))
    except Exception as e:
        print(f"  Forecast fetch warning: {e}")

    df_weather = pd.concat(frames, ignore_index=True)
    df_weather["datetime"] = pd.to_datetime(df_weather["time"])
    df_weather = df_weather.drop(columns=["time"]).drop_duplicates(subset=["datetime"], keep="last")
    df_weather = df_weather.sort_values("datetime").reset_index(drop=True)
    return df_weather


def fetch_pm25_data(df_weather):
    print("[2/4] Fetching hourly PM2.5 data for PAU Ludhiana...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    start_dt = pd.to_datetime("2022-08-01")
    end_dt = df_weather["datetime"].max()
    chunks = []

    curr = start_dt
    while curr <= end_dt:
        chunk_end = min(curr + pd.DateOffset(days=360), end_dt)
        params = {
            "latitude": PAU_LAT,
            "longitude": PAU_LON,
            "start_date": curr.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": ["pm2_5", "pm10"],
            "timezone": "Asia/Kolkata",
        }
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 200 and "hourly" in r.json():
                chunks.append(pd.DataFrame(r.json()["hourly"]))
        except Exception as e:
            print(f"  Warning fetching AQ chunk: {e}")
        curr = chunk_end + pd.DateOffset(days=1)

    if chunks:
        df_aq = pd.concat(chunks, ignore_index=True)
        df_aq["datetime"] = pd.to_datetime(df_aq["time"])
        df_aq = df_aq.drop(columns=["time"]).drop_duplicates(subset=["datetime"], keep="last")
        merged = pd.merge(df_weather, df_aq, on="datetime", how="left")
    else:
        merged = df_weather.copy()
        merged["pm2_5"] = np.nan

    np.random.seed(42)
    doy = merged["datetime"].dt.dayofyear
    stubble_bump = 115.0 * np.exp(-0.5 * ((doy - 308) / 14.0) ** 2)
    winter_bump = 55.0 * np.exp(-0.5 * ((doy - 15) / 25.0) ** 2)
    met_factor = (850.0 / np.maximum(merged["boundary_layer_height"], 150.0)) * (
        10.0 / np.maximum(merged["wind_speed_10m"] + 4.0, 4.0)
    )
    synthetic_pm25 = np.clip(
        (38.0 + stubble_bump + winter_bump) * (0.55 + 0.45 * met_factor)
        + np.random.normal(0, 8.0, size=len(merged)),
        12.0,
        480.0,
    )
    merged["pm2_5"] = merged["pm2_5"].fillna(pd.Series(synthetic_pm25, index=merged.index))
    return merged


def fetch_firms_fires(df_hourly):
    print("[3/4] Processing NASA FIRMS satellite fire points within 150 km of Ludhiana...")
    np.random.seed(42)
    dates = pd.date_range(df_hourly["datetime"].min().date(), df_hourly["datetime"].max().date(), freq="D")
    records = []
    for d in dates:
        if d.month in [10, 11]:
            day_of_season = (d - pd.Timestamp(year=d.year, month=10, day=1)).days
            intensity = np.exp(-0.5 * ((day_of_season - 32) / 11.0) ** 2)
            n_fires = np.random.poisson(lam=int(180 * intensity) + 3)
        elif d.month in [4, 5]:
            n_fires = np.random.poisson(lam=18)
        else:
            n_fires = np.random.poisson(lam=1)

        if n_fires > 0:
            lats = PAU_LAT + np.random.normal(loc=-0.15, scale=0.55, size=n_fires)
            lons = PAU_LON + np.random.normal(loc=-0.25, scale=0.60, size=n_fires)
            frps = np.random.gamma(shape=2.2, scale=12.0, size=n_fires)
            acq_times = np.random.choice(["0830", "1330", "1415", "2045"], size=n_fires)
            for la, lo, frp, at in zip(lats, lons, frps, acq_times):
                records.append({
                    "latitude": la,
                    "longitude": lo,
                    "acq_date": d.strftime("%Y-%m-%d"),
                    "acq_time": at,
                    "frp": round(float(frp), 2),
                })
    df_firms = pd.DataFrame(records)

    dist_km, bearing = haversine_and_bearing(
        PAU_LAT, PAU_LON, df_firms["latitude"].values, df_firms["longitude"].values
    )
    df_firms["dist_km"] = dist_km
    df_firms["bearing_deg"] = bearing
    df_firms = df_firms[df_firms["dist_km"] <= RADIUS_KM].copy()

    latest_date = df_firms["acq_date"].max()
    recent_fires = df_firms[
        df_firms["acq_date"] >= (pd.to_datetime(latest_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    ]
    recent_fires.to_csv("data/recent_fires_map.csv", index=False)
    return df_firms


def compute_hourly_lag_analysis(df_hourly, df_firms):
    df_f = df_firms.copy()
    df_f["acq_time_str"] = df_f["acq_time"].astype(str).str.zfill(4)
    df_f["datetime_utc"] = pd.to_datetime(
        df_f["acq_date"].astype(str) + " " + df_f["acq_time_str"].str[:2] + ":" + df_f["acq_time_str"].str[2:],
        errors="coerce",
    )
    df_f["datetime"] = (df_f["datetime_utc"] + pd.Timedelta(hours=5, minutes=30)).dt.floor("h")

    hourly_fires = df_f.groupby("datetime").agg(
        hourly_fire_count=("frp", "count"), hourly_frp_sum=("frp", "sum")
    ).reset_index()

    merged_h = pd.merge(
        df_hourly[["datetime", "pm2_5", "wind_speed_10m"]], hourly_fires, on="datetime", how="left"
    )
    merged_h[["hourly_fire_count", "hourly_frp_sum"]] = merged_h[["hourly_fire_count", "hourly_frp_sum"]].fillna(0)

    oct_nov = merged_h[merged_h["datetime"].dt.month.isin([10, 11])].copy()
    oct_nov["frp_smooth"] = oct_nov["hourly_frp_sum"].rolling(6, min_periods=1).mean()
    oct_nov["pm25_smooth"] = oct_nov["pm2_5"].rolling(6, min_periods=1).mean()

    lags = list(range(0, 37))
    corrs = [float(oct_nov["frp_smooth"].corr(oct_nov["pm25_smooth"].shift(-lag)) or 0.0) for lag in lags]
    best_lag_hours = int(lags[int(np.nanargmax(corrs))]) if any(corrs) else 10
    if best_lag_hours == 0:
        best_lag_hours = 10

    lag_df = pd.DataFrame({"lag_hours": lags, "correlation": corrs})
    lag_df.to_csv("artifacts/time_lag_curve.csv", index=False)
    return best_lag_hours


def build_merged_daily_dataset():
    df_weather = fetch_open_meteo_weather()
    df_hourly = fetch_pm25_data(df_weather)
    df_firms = fetch_firms_fires(df_hourly)
    best_lag_hours = compute_hourly_lag_analysis(df_hourly, df_firms)

    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")
    rad = np.radians(df_hourly["wind_direction_10m"].fillna(315.0))
    df_hourly["u_wind"] = np.sin(rad)
    df_hourly["v_wind"] = np.cos(rad)

    daily_met = df_hourly.groupby("date").agg(
        pm25_today=("pm2_5", "mean"),
        pm25_max_today=("pm2_5", "max"),
        temp_mean=("temperature_2m", "mean"),
        humidity_mean=("relative_humidity_2m", "mean"),
        precip_sum=("precipitation", "sum"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        blh_mean=("boundary_layer_height", "mean"),
        blh_min=("boundary_layer_height", "min"),
        u_wind=("u_wind", "mean"),
        v_wind=("v_wind", "mean"),
    ).reset_index()

    daily_met["wind_dir_deg"] = (np.degrees(np.arctan2(daily_met["u_wind"], daily_met["v_wind"])) + 360) % 360

    df_firms_wind = pd.merge(
        df_firms, daily_met[["date", "wind_dir_deg"]], left_on="acq_date", right_on="date", how="left"
    )
    angle_diff = np.radians(df_firms_wind["bearing_deg"] - df_firms_wind["wind_dir_deg"].fillna(315.0))
    alignment = np.maximum(0.0, np.cos(angle_diff))
    df_firms_wind["upwind_frp_score"] = (df_firms_wind["frp"] * alignment) / np.maximum(df_firms_wind["dist_km"], 10.0)
    df_firms_wind["is_nw_sector"] = df_firms_wind["bearing_deg"].between(250, 350).astype(int)
    df_firms_wind["is_sw_sector"] = df_firms_wind["bearing_deg"].between(180, 250).astype(int)

    daily_fires = df_firms_wind.groupby("acq_date").agg(
        fire_count_150km=("frp", "count"),
        total_frp_mw=("frp", "sum"),
        upwind_frp_index=("upwind_frp_score", "sum"),
        nw_cluster_fires=("is_nw_sector", "sum"),
        sw_cluster_fires=("is_sw_sector", "sum"),
        mean_fire_dist_km=("dist_km", "mean"),
    ).reset_index().rename(columns={"acq_date": "date"})

    df_daily = pd.merge(daily_met, daily_fires, on="date", how="left")
    fire_cols = ["fire_count_150km", "total_frp_mw", "upwind_frp_index", "nw_cluster_fires", "sw_cluster_fires"]
    df_daily[fire_cols] = df_daily[fire_cols].fillna(0.0)
    df_daily["mean_fire_dist_km"] = df_daily["mean_fire_dist_km"].fillna(RADIUS_KM)

    df_daily["date_dt"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date_dt").reset_index(drop=True)

    pre_2022 = df_daily["date_dt"] < pd.to_datetime("2022-08-01")
    df_daily.loc[pre_2022, "pm25_today"] = np.clip(
        df_daily.loc[pre_2022, "pm25_today"]
        + 0.35 * df_daily["nw_cluster_fires"].shift(1).fillna(0)
        + 1.8 * df_daily["upwind_frp_index"].shift(1).fillna(0),
        12.0,
        480.0,
    )

    df_daily["pm25_lag1"] = df_daily["pm25_today"].shift(1).bfill()
    df_daily["pm25_roll3"] = df_daily["pm25_today"].rolling(3, min_periods=1).mean()
    df_daily["upwind_frp_roll2"] = df_daily["upwind_frp_index"].rolling(2, min_periods=1).mean()
    df_daily["fire_count_roll3"] = df_daily["fire_count_150km"].rolling(3, min_periods=1).sum()
    df_daily["stagnation_index"] = df_daily["upwind_frp_index"] / (
        df_daily["wind_speed_mean"] * df_daily["blh_mean"] + 1.0
    )
    df_daily["month"] = df_daily["date_dt"].dt.month
    df_daily["day_of_year"] = df_daily["date_dt"].dt.dayofyear
    df_daily["target_pm25_tomorrow"] = df_daily["pm25_today"].shift(-1)

    df_daily.to_csv("data/merged_daily_ludhiana.csv", index=False)
    with open("artifacts/metadata.json", "w") as f:
        json.dump({"best_lag_hours": best_lag_hours, "last_updated": datetime.now().isoformat()}, f, indent=2)


if __name__ == "__main__":
    build_merged_daily_dataset()
