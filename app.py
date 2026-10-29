import os
import json
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="SmokeSignal Ludhiana", page_icon="🌫️", layout="wide")

ADVISORIES = {
    "Good": {
        "color": "#2ecc71",
        "en": "Air quality is clean. Ideal conditions for outdoor sports, running, and cycling across Ludhiana.",
        "pa": "ਹਵਾ ਦੀ ਗੁਣਵੱਤਾ ਸਾਫ਼ ਹੈ। ਲੁਧਿਆਣਾ ਵਿੱਚ ਬਾਹਰੀ ਖੇਡਾਂ, ਦੌੜਨ ਅਤੇ ਸਾਈਕਲ ਚਲਾਉਣ ਲਈ ਮੌਸਮ ਬਿਲਕੁਲ ਅਨੁਕੂਲ ਹੈ।"
    },
    "Satisfactory": {
        "color": "#a3e635",
        "en": "Air quality is acceptable. Unusually sensitive individuals with asthma should monitor heavy outdoor exertion.",
        "pa": "ਹਵਾ ਦੀ ਗੁਣਵੱਤਾ ਤਸੱਲੀਬਖਸ਼ ਹੈ। ਸਾਹ ਜਾਂ ਦਮੇ ਦੇ ਮਰੀਜ਼ਾਂ ਨੂੰ ਜ਼ਿਆਦਾ ਭਾਰੀ ਕਸਰਤ ਵੇਲੇ ਥੋੜ੍ਹੀ ਸਾਵਧਾਨੀ ਵਰਤਣੀ ਚਾਹੀਦੀ ਹੈ।"
    },
    "Moderate": {
        "color": "#facc15",
        "en": "Moderate particulate buildup. Consider reducing prolonged high-intensity outdoor workouts during early morning inversion hours.",
        "pa": "ਹਵਾ ਵਿੱਚ ਪ੍ਰਦੂਸ਼ਣ ਦਾ ਪੱਧਰ ਦਰਮਿਆਨਾ ਹੈ। ਸਵੇਰ ਵੇਲੇ ਧੂੰਏਂ ਦੀ ਪਰਤ ਕਾਰਨ ਲੰਬੀ ਬਾਹਰੀ ਕਸਰਤ ਘੱਟ ਕਰਨ ਦੀ ਸਲਾਹ ਦਿੱਤੀ ਜਾਂਦੀ ਹੈ।"
    },
    "Poor": {
        "color": "#fb923c",
        "en": "Poor air quality driven by regional crop-residue smoke. Shift school sports indoors, wear an N95 mask during commutes, and keep windows closed after sunset.",
        "pa": "ਪਰਾਲੀ ਦੇ ਧੂੰਏਂ ਕਾਰਨ ਹਵਾ ਦੀ ਗੁਣਵੱਤਾ ਖਰਾਬ ਹੈ। ਸਕੂਲੀ ਖੇਡਾਂ ਅੰਦਰ ਕਰਵਾਓ, ਬਾਹਰ ਜਾਣ ਸਮੇਂ N95 ਮਾਸਕ ਪਹਿਨੋ ਅਤੇ ਸ਼ਾਮ ਤੋਂ ਬਾਅਦ ਖਿੜਕੀਆਂ ਬੰਦ ਰੱਖੋ।"
    },
    "Very Poor": {
        "color": "#ef4444",
        "en": "Very Poor air quality alert! Avoid outdoor morning runs/assemblies. Children, elders, and asthma patients must stay indoors with air filtration where possible.",
        "pa": "ਹਵਾ ਦੀ ਗੁਣਵੱਤਾ ਬਹੁਤ ਖਰਾਬ ਹੈ! ਸਵੇਰ ਦੀ ਸੈਰ ਅਤੇ ਬਾਹਰੀ ਸਭਾਵਾਂ ਤੋਂ ਬਚੋ। ਬੱਚਿਆਂ, ਬਜ਼ੁਰਗਾਂ ਅਤੇ ਸਾਹ ਦੇ ਮਰੀਜ਼ਾਂ ਨੂੰ ਘਰ ਦੇ ਅੰਦਰ ਰਹਿਣ ਦੀ ਸਖ਼ਤ ਸਲਾਹ ਦਿੱਤੀ ਜਾਂਦੀ ਹੈ।"
    },
    "Severe": {
        "color": "#7f1d1d",
        "en": "SEVERE SMOG EMERGENCY. Strict N95 usage outdoors, halt all outdoor physical training, and seal door/window drafts.",
        "pa": "ਗੰਭੀਰ ਧੂੰਏਂ ਦੀ ਚੇਤਾਵਨੀ! ਬਾਹਰ ਨਿਕਲਣ ਵੇਲੇ N95 ਮਾਸਕ ਲਾਜ਼ਮੀ ਪਹਿਨੋ, ਸਾਰੀਆਂ ਬਾਹਰੀ ਖੇਡਾਂ ਬੰਦ ਰੱਖੋ ਅਤੇ ਦਰਵਾਜ਼ੇ-ਖਿੜਕੀਆਂ ਚੰਗੀ ਤਰ੍ਹਾਂ ਬੰਦ ਰੱਖੋ।"
    },
}

st.title("🌫️ SmokeSignal Ludhiana: Farm-Fire to Next-Day PM2.5 AI Forecast")
st.caption("Tracking NASA VIIRS satellite thermal anomalies within 150 km of PAU Ludhiana & Open-Meteo boundary-layer physics.")

if not os.path.exists("artifacts/latest_forecast.json"):
    st.warning("Forecast artifacts not found. Run `python data_pipeline.py && python train_and_explain.py` first.")
    st.stop()

with open("artifacts/latest_forecast.json", "r") as f:
    fc = json.load(f)

cat = fc["tomorrow_category_xgb"]
adv = ADVISORIES.get(cat, ADVISORIES["Poor"])

# Top KPI Row
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Tomorrow's Predicted PM2.5 (PAU)",
    f"{fc['tomorrow_pm25_xgb']} µg/m³",
    f"{fc['tomorrow_pm25_xgb'] - fc['today_pm25']:+.1f} vs Today",
    delta_color="inverse",
)
c2.metric("Predicted AQI Category", cat, f"Today: {fc['today_category']}")
c3.metric("Fires Within 150 km Today", f"{fc['fire_count_150km_today']}", f"{fc['nw_cluster_fires_today']} in Upwind NW Sector")
c4.metric("Smoke Arrival Time Lag", f"~{fc['best_lag_hours']} Hours", f"Wind: {fc['wind_speed_kmh']} km/h ({int(fc['wind_dir_deg'])}°)")

# Bilingual Advisory Banner
st.markdown(
    f"""
    <div style="background-color:{adv['color']}22; border-left: 6px solid {adv['color']}; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="margin:0 0 8px 0;">🩺 Tomorrow's Health Advisory ({cat} Air Quality)</h4>
        <p style="margin:0 0 6px 0;"><b>English:</b> {adv['en']}</p>
        <p style="margin:0; font-size: 1.05em;"><b>ਪੰਜਾਬੀ (Punjabi):</b> {adv['pa']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_map, col_shap = st.columns([1.25, 1])

with col_map:
    st.subheader("🔥 Active Farm Fires Around Ludhiana (150 km Radius)")
    if os.path.exists("data/recent_fires_map.csv"):
        df_fires = pd.read_csv("data/recent_fires_map.csv")
        pau_point = pd.DataFrame([{"latitude": 30.9009, "longitude": 75.8078, "name": "PAU Ludhiana CPCB Station"}])

        fire_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_fires,
            get_position="[longitude, latitude]",
            get_radius=2200,
            get_fill_color="[255, 75, 0, 170]",
            pickable=True,
        )
        pau_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pau_point,
            get_position="[longitude, latitude]",
            get_radius=4500,
            get_fill_color="[0, 180, 255, 240]",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=30.9009, longitude=75.8078, zoom=7.5, pitch=25)
        st.pydeck_chart(pdk.Deck(layers=[fire_layer, pau_layer], initial_view_state=view_state))
        st.caption("🟠 Orange dots: NASA VIIRS fire points | 🔵 Blue dot: PAU Ludhiana Monitoring Station")

with col_shap:
    st.subheader("🧠 Why the AI Predicts This (SHAP Drivers)")
    st.write("Top physical factors pushing tomorrow's forecast up (+) or down (-) relative to average:")
    df_drivers = pd.DataFrame(fc["top_shap_drivers"])
    st.dataframe(df_drivers, use_container_width=True, hide_index=True)

    if os.path.exists("artifacts/shap_summary.png"):
        st.image("artifacts/shap_summary.png", caption="Global SHAP Feature Importance (2024–2025 Seasons)")

st.divider()
col_eval, col_lag = st.columns(2)

with col_eval:
    st.subheader("📊 Honest Model Evaluation (2024–2025 Seasons)")
    m = fc["metrics_2024_2025"]
    eval_table = pd.DataFrame([
        {"Model": "Persistence Baseline (Tomorrow = Today)", "MAE (µg/m³)": m["baseline_mae"], "RMSE (µg/m³)": m["baseline_rmse"], "AQI Category Accuracy": f"{m['baseline_aqi_acc_pct']}%"},
        {"Model": "SmokeSignal XGBoost (Fires + Wind + BLH)", "MAE (µg/m³)": m["xgb_mae"], "RMSE (µg/m³)": m["xgb_rmse"], "AQI Category Accuracy": f"{m['xgb_aqi_acc_pct']}%"},
    ])
    st.table(eval_table)
    st.success(f"**Gain over Baseline:** SmokeSignal reduces next-day PM2.5 error by **{m['mae_improvement_pct']}%**.")

with col_lag:
    st.subheader("⏱️ Plume Transport Time-Lag Analysis")
    if os.path.exists("artifacts/time_lag_curve.csv"):
        lag_df = pd.read_csv("artifacts/time_lag_curve.csv")
        st.line_chart(lag_df.set_index("lag_hours")["correlation"])
        st.caption(f"Cross-correlation peaks at **{fc['best_lag_hours']} hours** between satellite fire detection and PAU Ludhiana PM2.5 spike.")
