import os
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants for Punjab Agricultural University (PAU), Ludhiana CPCB Station
PAU_LAT = 30.9009
PAU_LON = 75.8078
RADIUS_KM = 150.0
# Bounding box ~150km around Ludhiana: [west, south, east, north]
BBOX = "74.20,29.55,77.40,32.25"

os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def haversine_and_bearing(lat1, lon1, lat2_arr, lon2_arr):
    """
    Calculates distance (km) and compass bearing (degrees FROM Ludhiana TO the fire)
    for vectorized numpy arrays.
    """
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


def fetch_open_meteo_weather(start_date="2019-01-01", end_date=None):
    """Fetches hourly meteorological archive + recent forecast from Open-Meteo for PAU Ludhiana."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[1/4] Fetching Open-Meteo weather data ({start_date} to {end_date})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": PAU_LAT,
        "longitude": PAU_LON,
        "start_date": start_date,
        "end_date": end_date,
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
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df_weather = pd.DataFrame(hourly)
    df_weather["datetime"] = pd.to_datetime(df_weather["time"])
    df_weather.drop(columns=["time"], inplace=True)
    return df_weather


def fetch_pm25_data(start_date="2019-01-01", end_date=None):
    """
    Loads hourly PM2.5 from local CPCB PAU export if present, and merges/backfills
    with Open-Meteo Air Quality archive for PAU Ludhiana coordinates.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[2/4] Fetching hourly PM2.5 data for PAU Ludhiana ({start_date} to {end_date})...")
    # Fetch continuous hourly baseline from Open-Meteo Air Quality API (covers 2019-2026 seamlessly)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    # Open-Meteo AQ API works best in yearly chunks for long ranges
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    chunks = []

    curr = start_dt
    while curr <= end_dt:
        chunk_end = min(curr + pd.DateOffset(days=364), end_dt)
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
                cdf = pd.DataFrame(r.json()["hourly"])
                chunks.append(cdf)
        except Exception as e:
            print(f"  Warning fetching AQ chunk {curr.date()}: {e}")
        curr = chunk_end + pd.DateOffset(days=1)

    df_aq = pd.concat(chunks, ignore_index=True)
    df_aq["datetime"] = pd.to_datetime(df_aq["time"])
    df_aq.drop(columns=["time"], inplace=True)

    # Check if user placed an official CPCB PAU station CSV in data/cpcb_pau_raw.csv
    cpcb_path = "data/cpcb_pau_raw.csv"
    if os.path.exists(cpcb_path):
        print("  Found local CPCB PAU file (`data/cpcb_pau_raw.csv`). Merging official station readings...")
        cpcb_df = pd.read_csv(cpcb_path)
        # Standardize CPCB column names ("From Date", "PM2.5")
        date_col = [c for c in cpcb_df.columns if "date" in c.lower() or "time" in c.lower()][0]
        pm_col = [c for c in cpcb_df.columns if "pm2.5" in c.lower() or "pm2_5" in c.lower()][0]
        cpcb_df["datetime"] = pd.to_datetime(cpcb_df[date_col], errors="coerce")
        cpcb_df["pm2_5_cpcb"] = pd.to_numeric(cpcb_df[pm_col], errors="coerce")
        cpcb_df = cpcb_df[["datetime", "pm2_5_cpcb"]].dropna()

        df_aq = pd.merge(df_aq, cpcb_df, on="datetime", how="left")
        df_aq["pm2_5"] = df_aq["pm2_5_cpcb"].combine_first(df_aq["pm2_5"])
        df_aq.drop(columns=["pm2_5_cpcb"], inplace=True)

    return df_aq


def fetch_firms_fires(df_weather):
    """
    Loads historical NASA FIRMS CSVs from `data/firms_historical.csv` and fetches live NRT fires
    if NASA_FIRMS_MAP_KEY is set in environment variables.
    If no historical FIRMS CSV is present yet, generates a realistic calibrated dataset
    anchored to Punjab's Oct-Nov paddy harvest season so you can test the pipeline immediately.
    """
    print("[3/4] Processing NASA FIRMS satellite fire points within 150 km of Ludhiana...")
    hist_path = "data/firms_historical.csv"
    live_frames = []

    if os.path.exists(hist_path):
        print("  Loading `data/firms_historical.csv`...")
        df_firms = pd.read_csv(hist_path)
    else:
        print("  NOTE: `data/firms_historical.csv` not found. Generating calibrated seasonal fire dataset")
        print("  for immediate testing (replace with NASA FIRMS archive download for final submission).")
        np.random.seed(42)
        dates = pd.date_range(df_weather["datetime"].min().date(), df_weather["datetime"].max().date(), freq="D")
        records = []
        for d in dates:
            # Peak stubble burning occurs Oct 10 - Nov 25, minor wheat stubble Apr 15 - May 15
            if d.month in [10, 11]:
                day_of_season = (d - pd.Timestamp(year=d.year, month=10, day=1)).days
                # Gaussian peak around Nov 2 (day 32)
                intensity = np.exp(-0.5 * ((day_of_season - 32) / 11.0) ** 2)
                n_fires = np.random.poisson(lam=int(180 * intensity) + 3)
            elif d.month in [4, 5]:
                n_fires = np.random.poisson(lam=18)
            else:
                n_fires = np.random.poisson(lam=1)

            if n_fires > 0:
                # Generate fire clusters heavily weighted toward NW/W/SW districts (Sangrur, Barnala, Moga, Jalandhar)
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

    # Pull live NRT FIRMS data if API key is provided
    map_key = os.environ.get("NASA_FIRMS_MAP_KEY", "").strip()
    if map_key:
        print("  Fetching live past-5-days VIIRS_SNPP_NRT fires from NASA FIRMS API...")
        nrt_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{BBOX}/5"
        try:
            r = requests.get(nrt_url, timeout=30)
            if r.status_code == 200 and "latitude" in r.text:
                from io import StringIO
                df_live = pd.read_csv(StringIO(r.text))
                df_firms = pd.concat([df_firms, df_live], ignore_index=True).drop_duplicates(
                    subset=["latitude", "longitude", "acq_date", "acq_time"]
                )
        except Exception as e:
            print(f"  Warning fetching live FIRMS: {e}")

    # Calculate distance and compass bearing from PAU Ludhiana
    dist_km, bearing = haversine_and_bearing(
        PAU_LAT, PAU_LON, df_firms["latitude"].values, df_firms["longitude"].values
    )
    df_firms["dist_km"] = dist_km
    df_firms["bearing_deg"] = bearing

    # Keep only fires within 150 km of Ludhiana
    df_firms = df_firms[df_firms["dist_km"] <= RADIUS_KM].copy()

    # Save tonight's/latest fires for the Streamlit interactive map
    latest_date = df_firms["acq_date"].max()
    recent_fires = df_firms[df_firms["acq_date"] >= (pd.to_datetime(latest_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")]
    recent_fires.to_csv("data/recent_fires_map.csv", index=False)

    return df_firms


def compute_hourly_lag_analysis(df_hourly, df_firms):
    """
    Step 3 Requirement: Measure how many hours pass between fires and a PM2.5 rise in the city.
    Uses VIIRS overpass timestamp (`acq_date` + `acq_time`) cross-correlated against hourly PM2.5.
    """
    df_f = df_firms.copy()
    df_f["acq_time_str"] = df_f["acq_time"].astype(str).str.zfill(4)
    # Convert UTC satellite overpass to IST (+5:30) and round to hour
    df_f["datetime_utc"] = pd.to_datetime(
        df_f["acq_date"].astype(str) + " " + df_f["acq_time_str"].str[:2] + ":" + df_f["acq_time_str"].str[2:],
        errors="coerce"
    )
    df_f["datetime"] = (df_f["datetime_utc"] + pd.Timedelta(hours=5, minutes=30)).dt.floor("h")

    hourly_fires = df_f.groupby("datetime").agg(
        hourly_fire_count=("frp", "count"),
        hourly_frp_sum=("frp", "sum")
    ).reset_index()

    merged_h = pd.merge(df_hourly[["datetime", "pm2_5", "wind_speed_10m"]], hourly_fires, on="datetime", how="left")
    merged_h[["hourly_fire_count", "hourly_frp_sum"]] = merged_h[["hourly_fire_count", "hourly_frp_sum"]].fillna(0)

    # Filter to Oct-Nov stubble burning season for clean signal
    oct_nov = merged_h[merged_h["datetime"].dt.month.isin([10, 11])].copy()
    oct_nov["frp_smooth"] = oct_nov["hourly_frp_sum"].rolling(6, min_periods=1).mean()
    oct_nov["pm25_smooth"] = oct_nov["pm2_5"].rolling(6, min_periods=1).mean()

    lags = list(range(0, 37))
    corrs = [oct_nov["frp_smooth"].corr(oct_nov["pm25_smooth"].shift(-lag)) for lag in lags]
    best_lag_hours = int(lags[int(np.nanargmax(corrs))])

    lag_df = pd.DataFrame({"lag_hours": lags, "correlation": corrs})
    lag_df.to_csv("artifacts/time_lag_curve.csv", index=False)
    print(f"  -> Peak cross-correlation between upwind fires & PAU PM2.5 rise: {best_lag_hours} hours.")
    return best_lag_hours


def build_merged_daily_dataset():
    df_weather = fetch_open_meteo_weather()
    df_aq = fetch_pm25_data()
    df_hourly = pd.merge(df_weather, df_aq, on="datetime", how="inner")

    df_firms = fetch_firms_fires(df_hourly)
    best_lag_hours = compute_hourly_lag_analysis(df_hourly, df_firms)

    print("[4/4] Aggregating daily meteorological & directional upwind fire features...")
    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")

    # Vector-averaged daily wind direction
    rad = np.radians(df_hourly["wind_direction_10m"])
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

    # Join daily wind direction onto fires to compute exact upwind alignment
    df_firms_wind = pd.merge(
        df_firms, daily_met[["date", "wind_dir_deg"]], left_on="acq_date", right_on="date", how="left"
    )
    # Cosine alignment between wind direction (where air comes FROM) and fire bearing (where fire sits relative to PAU)
    angle_diff = np.radians(df_firms_wind["bearing_deg"] - df_firms_wind["wind_dir_deg"])
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

    # Sort and engineer time-series lag & rolling features
    df_daily["date_dt"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date_dt").reset_index(drop=True)

    df_daily["pm25_lag1"] = df_daily["pm25_today"].shift(1)
    df_daily["pm25_roll3"] = df_daily["pm25_today"].rolling(3, min_periods=1).mean()
    df_daily["upwind_frp_roll2"] = df_daily["upwind_frp_index"].rolling(2, min_periods=1).mean()
    df_daily["fire_count_roll3"] = df_daily["fire_count_150km"].rolling(3, min_periods=1).sum()
    df_daily["stagnation_index"] = df_daily["upwind_frp_index"] / (df_daily["wind_speed_mean"] * df_daily["blh_mean"] + 1.0)
    df_daily["month"] = df_daily["date_dt"].dt.month
    df_daily["day_of_year"] = df_daily["date_dt"].dt.dayofyear

    # Target variable: Next-Day PM2.5
    df_daily["target_pm25_tomorrow"] = df_daily["pm25_today"].shift(-1)

    df_daily.to_csv("data/merged_daily_ludhiana.csv", index=False)
    with open("artifacts/metadata.json", "w") as f:
        json.dump({"best_lag_hours": best_lag_hours, "last_updated": datetime.now().isoformat()}, f, indent=2)

    print(f"Saved `data/merged_daily_ludhiana.csv` ({len(df_daily)} daily rows).")


if __name__ == "__main__":
    build_merged_daily_dataset()import os
import json
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants for Punjab Agricultural University (PAU), Ludhiana CPCB Station
PAU_LAT = 30.9009
PAU_LON = 75.8078
RADIUS_KM = 150.0
# Bounding box ~150km around Ludhiana: [west, south, east, north]
BBOX = "74.20,29.55,77.40,32.25"

os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def haversine_and_bearing(lat1, lon1, lat2_arr, lon2_arr):
    """
    Calculates distance (km) and compass bearing (degrees FROM Ludhiana TO the fire)
    for vectorized numpy arrays.
    """
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


def fetch_open_meteo_weather(start_date="2019-01-01", end_date=None):
    """Fetches hourly meteorological archive + recent forecast from Open-Meteo for PAU Ludhiana."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[1/4] Fetching Open-Meteo weather data ({start_date} to {end_date})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": PAU_LAT,
        "longitude": PAU_LON,
        "start_date": start_date,
        "end_date": end_date,
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
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df_weather = pd.DataFrame(hourly)
    df_weather["datetime"] = pd.to_datetime(df_weather["time"])
    df_weather.drop(columns=["time"], inplace=True)
    return df_weather


def fetch_pm25_data(start_date="2019-01-01", end_date=None):
    """
    Loads hourly PM2.5 from local CPCB PAU export if present, and merges/backfills
    with Open-Meteo Air Quality archive for PAU Ludhiana coordinates.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[2/4] Fetching hourly PM2.5 data for PAU Ludhiana ({start_date} to {end_date})...")
    # Fetch continuous hourly baseline from Open-Meteo Air Quality API (covers 2019-2026 seamlessly)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    # Open-Meteo AQ API works best in yearly chunks for long ranges
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    chunks = []

    curr = start_dt
    while curr <= end_dt:
        chunk_end = min(curr + pd.DateOffset(days=364), end_dt)
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
                cdf = pd.DataFrame(r.json()["hourly"])
                chunks.append(cdf)
        except Exception as e:
            print(f"  Warning fetching AQ chunk {curr.date()}: {e}")
        curr = chunk_end + pd.DateOffset(days=1)

    df_aq = pd.concat(chunks, ignore_index=True)
    df_aq["datetime"] = pd.to_datetime(df_aq["time"])
    df_aq.drop(columns=["time"], inplace=True)

    # Check if user placed an official CPCB PAU station CSV in data/cpcb_pau_raw.csv
    cpcb_path = "data/cpcb_pau_raw.csv"
    if os.path.exists(cpcb_path):
        print("  Found local CPCB PAU file (`data/cpcb_pau_raw.csv`). Merging official station readings...")
        cpcb_df = pd.read_csv(cpcb_path)
        # Standardize CPCB column names ("From Date", "PM2.5")
        date_col = [c for c in cpcb_df.columns if "date" in c.lower() or "time" in c.lower()][0]
        pm_col = [c for c in cpcb_df.columns if "pm2.5" in c.lower() or "pm2_5" in c.lower()][0]
        cpcb_df["datetime"] = pd.to_datetime(cpcb_df[date_col], errors="coerce")
        cpcb_df["pm2_5_cpcb"] = pd.to_numeric(cpcb_df[pm_col], errors="coerce")
        cpcb_df = cpcb_df[["datetime", "pm2_5_cpcb"]].dropna()

        df_aq = pd.merge(df_aq, cpcb_df, on="datetime", how="left")
        df_aq["pm2_5"] = df_aq["pm2_5_cpcb"].combine_first(df_aq["pm2_5"])
        df_aq.drop(columns=["pm2_5_cpcb"], inplace=True)

    return df_aq


def fetch_firms_fires(df_weather):
    """
    Loads historical NASA FIRMS CSVs from `data/firms_historical.csv` and fetches live NRT fires
    if NASA_FIRMS_MAP_KEY is set in environment variables.
    If no historical FIRMS CSV is present yet, generates a realistic calibrated dataset
    anchored to Punjab's Oct-Nov paddy harvest season so you can test the pipeline immediately.
    """
    print("[3/4] Processing NASA FIRMS satellite fire points within 150 km of Ludhiana...")
    hist_path = "data/firms_historical.csv"
    live_frames = []

    if os.path.exists(hist_path):
        print("  Loading `data/firms_historical.csv`...")
        df_firms = pd.read_csv(hist_path)
    else:
        print("  NOTE: `data/firms_historical.csv` not found. Generating calibrated seasonal fire dataset")
        print("  for immediate testing (replace with NASA FIRMS archive download for final submission).")
        np.random.seed(42)
        dates = pd.date_range(df_weather["datetime"].min().date(), df_weather["datetime"].max().date(), freq="D")
        records = []
        for d in dates:
            # Peak stubble burning occurs Oct 10 - Nov 25, minor wheat stubble Apr 15 - May 15
            if d.month in [10, 11]:
                day_of_season = (d - pd.Timestamp(year=d.year, month=10, day=1)).days
                # Gaussian peak around Nov 2 (day 32)
                intensity = np.exp(-0.5 * ((day_of_season - 32) / 11.0) ** 2)
                n_fires = np.random.poisson(lam=int(180 * intensity) + 3)
            elif d.month in [4, 5]:
                n_fires = np.random.poisson(lam=18)
            else:
                n_fires = np.random.poisson(lam=1)

            if n_fires > 0:
                # Generate fire clusters heavily weighted toward NW/W/SW districts (Sangrur, Barnala, Moga, Jalandhar)
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

    # Pull live NRT FIRMS data if API key is provided
    map_key = os.environ.get("NASA_FIRMS_MAP_KEY", "").strip()
    if map_key:
        print("  Fetching live past-5-days VIIRS_SNPP_NRT fires from NASA FIRMS API...")
        nrt_url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{BBOX}/5"
        try:
            r = requests.get(nrt_url, timeout=30)
            if r.status_code == 200 and "latitude" in r.text:
                from io import StringIO
                df_live = pd.read_csv(StringIO(r.text))
                df_firms = pd.concat([df_firms, df_live], ignore_index=True).drop_duplicates(
                    subset=["latitude", "longitude", "acq_date", "acq_time"]
                )
        except Exception as e:
            print(f"  Warning fetching live FIRMS: {e}")

    # Calculate distance and compass bearing from PAU Ludhiana
    dist_km, bearing = haversine_and_bearing(
        PAU_LAT, PAU_LON, df_firms["latitude"].values, df_firms["longitude"].values
    )
    df_firms["dist_km"] = dist_km
    df_firms["bearing_deg"] = bearing

    # Keep only fires within 150 km of Ludhiana
    df_firms = df_firms[df_firms["dist_km"] <= RADIUS_KM].copy()

    # Save tonight's/latest fires for the Streamlit interactive map
    latest_date = df_firms["acq_date"].max()
    recent_fires = df_firms[df_firms["acq_date"] >= (pd.to_datetime(latest_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")]
    recent_fires.to_csv("data/recent_fires_map.csv", index=False)

    return df_firms


def compute_hourly_lag_analysis(df_hourly, df_firms):
    """
    Step 3 Requirement: Measure how many hours pass between fires and a PM2.5 rise in the city.
    Uses VIIRS overpass timestamp (`acq_date` + `acq_time`) cross-correlated against hourly PM2.5.
    """
    df_f = df_firms.copy()
    df_f["acq_time_str"] = df_f["acq_time"].astype(str).str.zfill(4)
    # Convert UTC satellite overpass to IST (+5:30) and round to hour
    df_f["datetime_utc"] = pd.to_datetime(
        df_f["acq_date"].astype(str) + " " + df_f["acq_time_str"].str[:2] + ":" + df_f["acq_time_str"].str[2:],
        errors="coerce"
    )
    df_f["datetime"] = (df_f["datetime_utc"] + pd.Timedelta(hours=5, minutes=30)).dt.floor("h")

    hourly_fires = df_f.groupby("datetime").agg(
        hourly_fire_count=("frp", "count"),
        hourly_frp_sum=("frp", "sum")
    ).reset_index()

    merged_h = pd.merge(df_hourly[["datetime", "pm2_5", "wind_speed_10m"]], hourly_fires, on="datetime", how="left")
    merged_h[["hourly_fire_count", "hourly_frp_sum"]] = merged_h[["hourly_fire_count", "hourly_frp_sum"]].fillna(0)

    # Filter to Oct-Nov stubble burning season for clean signal
    oct_nov = merged_h[merged_h["datetime"].dt.month.isin([10, 11])].copy()
    oct_nov["frp_smooth"] = oct_nov["hourly_frp_sum"].rolling(6, min_periods=1).mean()
    oct_nov["pm25_smooth"] = oct_nov["pm2_5"].rolling(6, min_periods=1).mean()

    lags = list(range(0, 37))
    corrs = [oct_nov["frp_smooth"].corr(oct_nov["pm25_smooth"].shift(-lag)) for lag in lags]
    best_lag_hours = int(lags[int(np.nanargmax(corrs))])

    lag_df = pd.DataFrame({"lag_hours": lags, "correlation": corrs})
    lag_df.to_csv("artifacts/time_lag_curve.csv", index=False)
    print(f"  -> Peak cross-correlation between upwind fires & PAU PM2.5 rise: {best_lag_hours} hours.")
    return best_lag_hours


def build_merged_daily_dataset():
    df_weather = fetch_open_meteo_weather()
    df_aq = fetch_pm25_data()
    df_hourly = pd.merge(df_weather, df_aq, on="datetime", how="inner")

    df_firms = fetch_firms_fires(df_hourly)
    best_lag_hours = compute_hourly_lag_analysis(df_hourly, df_firms)

    print("[4/4] Aggregating daily meteorological & directional upwind fire features...")
    df_hourly["date"] = df_hourly["datetime"].dt.strftime("%Y-%m-%d")

    # Vector-averaged daily wind direction
    rad = np.radians(df_hourly["wind_direction_10m"])
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

    # Join daily wind direction onto fires to compute exact upwind alignment
    df_firms_wind = pd.merge(
        df_firms, daily_met[["date", "wind_dir_deg"]], left_on="acq_date", right_on="date", how="left"
    )
    # Cosine alignment between wind direction (where air comes FROM) and fire bearing (where fire sits relative to PAU)
    angle_diff = np.radians(df_firms_wind["bearing_deg"] - df_firms_wind["wind_dir_deg"])
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

    # Sort and engineer time-series lag & rolling features
    df_daily["date_dt"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date_dt").reset_index(drop=True)

    df_daily["pm25_lag1"] = df_daily["pm25_today"].shift(1)
    df_daily["pm25_roll3"] = df_daily["pm25_today"].rolling(3, min_periods=1).mean()
    df_daily["upwind_frp_roll2"] = df_daily["upwind_frp_index"].rolling(2, min_periods=1).mean()
    df_daily["fire_count_roll3"] = df_daily["fire_count_150km"].rolling(3, min_periods=1).sum()
    df_daily["stagnation_index"] = df_daily["upwind_frp_index"] / (df_daily["wind_speed_mean"] * df_daily["blh_mean"] + 1.0)
    df_daily["month"] = df_daily["date_dt"].dt.month
    df_daily["day_of_year"] = df_daily["date_dt"].dt.dayofyear

    # Target variable: Next-Day PM2.5
    df_daily["target_pm25_tomorrow"] = df_daily["pm25_today"].shift(-1)

    df_daily.to_csv("data/merged_daily_ludhiana.csv", index=False)
    with open("artifacts/metadata.json", "w") as f:
        json.dump({"best_lag_hours": best_lag_hours, "last_updated": datetime.now().isoformat()}, f, indent=2)

    print(f"Saved `data/merged_daily_ludhiana.csv` ({len(df_daily)} daily rows).")


if __name__ == "__main__":
    build_merged_daily_dataset()
