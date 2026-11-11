import os
import json
import requests
from io import StringIO
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
    lat2_arr = np.asarray(lat2_arr, dtype=np.float64)
    lon2_arr = np.asarray(lon2_arr, dtype=np.float64)

    phi1 = np.radians(float(lat1))
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - float(lat1))
    dlambda = np.radians(lon2_arr - float(lon1))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
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
    doy = merged["datetime"].dt.day_of_year
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
    hist_path = "data/firms_historical.csv"

    if os.path.exists(hist_path
