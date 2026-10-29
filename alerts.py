import os
import json
import requests

def send_daily_alert():
    with open("artifacts/latest_forecast.json", "r") as f:
        fc = json.load(f)

    msg = (
        f"🌫️ *SmokeSignal Ludhiana — Tomorrow's Air Forecast*\n"
        f"📅 Generated: {fc['forecast_generated_on']}\n\n"
        f"• *Tomorrow's Predicted PM2.5 (PAU):* `{fc['tomorrow_pm25_xgb']} µg/m³` ({fc['tomorrow_category_xgb']})\n"
        f"• *Today's PM2.5:* `{fc['today_pm25']} µg/m³`\n"
        f"• *Fires Detected (<150 km):* `{fc['fire_count_150km_today']}` ({fc['nw_cluster_fires_today']} in NW upwind sector)\n"
        f"• *Wind & Mixing Height:* `{fc['wind_speed_kmh']} km/h`, `{int(fc['blh_m'])} m`\n\n"
        f"🩺 *Advisory (English):* Wear an N95 mask outdoors if AQI is Poor/Very Poor and shift morning school exercise indoors.\n"
        f"🩺 *ਸਲਾਹ (ਪੰਜਾਬੀ):* ਖਰਾਬ ਹਵਾ ਦੌਰਾਨ ਬਾਹਰ ਜਾਣ ਸਮੇਂ N95 ਮਾਸਕ ਪਹਿਨੋ ਅਤੇ ਸਵੇਰ ਦੀ ਕਸਰਤ ਘਰ ਦੇ ਅੰਦਰ ਕਰੋ।"
    )

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=15)
        print(f"Telegram alert status: {resp.status_code}")
    else:
        print("Telegram credentials not set; printing formatted alert below:\n")
        print(msg)

if __name__ == "__main__":
    send_daily_alert()
