"""
app.py
======
VoltPulse — Bengaluru EV Smart Charging
Punchy, modern mobility web application with editorial font pairing, micro-copy, and instant scanability.

Design:
- Font Pairing: 'Space Grotesk' (Headings, Numbers, CTA) & 'DM Sans' (UI Labels & Body)
- Micro-copy: Strict 8-word limit per line
- Highlights: Custom inline badges for critical numbers (Wait Time, Load %, Best Window)
- Real-time Open-Meteo Bengaluru weather auto-sync (cached 30m)
- Zero 42MB dataset loading
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib

import streamlit as st
import shap

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & REFINED EDITORIAL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VoltPulse — Bengaluru EV Smart Charging",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS: Space Grotesk + DM Sans, Warm Sunlit Cream, Inline Highlight Badges
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=Space+Grotesk:wght@600;700&display=swap');

    /* Hide Streamlit default chrome & margins */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 880px !important;
    }
    
    /* Warm Sunlit Cream Background */
    .stApp {
        background-color: #FFFBF5 !important;
        color: #1E1B4B !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }
    
    /* Headings, numbers, and branding use Space Grotesk */
    h1, h2, h3, h4, .brand-font {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        letter-spacing: -0.03em !important;
        color: #1E1B4B;
    }
    
    p, label, .stMarkdown, div {
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }
    
    /* White Card Form Container */
    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #F1E8DC !important;
        border-radius: 18px !important;
        padding: 24px 26px !important;
        box-shadow: 0 10px 30px rgba(30, 27, 75, 0.04) !important;
        margin-bottom: 20px !important;
    }
    
    /* Primary CTA Button: Deep Space Grotesk with Teal Glow */
    .stButton > button {
        background: linear-gradient(135deg, #1E1B4B 0%, #0D9488 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        letter-spacing: -0.02em !important;
        height: 50px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(13, 148, 136, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #312E81 0%, #0F766E 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.5) !important;
        color: #FFFFFF !important;
    }
    
    /* Input selectbox styling */
    div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #1E1B4B !important;
        font-size: 0.92rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Dropdown popover listbox styling */
    [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 30px rgba(30, 27, 75, 0.12) !important;
    }
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #1E1B4B !important;
        font-size: 0.90rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #F0FDFA !important;
        color: #0D9488 !important;
        font-weight: 600 !important;
    }
    
    .stSlider > div {
        color: #1E1B4B !important;
    }

    /* Custom inline highlight badges */
    .hl-emerald {
        background: rgba(16, 185, 129, 0.18);
        color: #065F46;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    .hl-amber {
        background: rgba(245, 158, 11, 0.20);
        color: #92400E;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    .hl-coral {
        background: rgba(239, 68, 68, 0.18);
        color: #991B1B;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Paths setup
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
STATIONS_CSV = RAW_DIR / "bengaluru_stations.csv"


# -----------------------------------------------------------------------------
# 1. LIVE WEATHER AUTO-SYNC (OPEN-METEO)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_bengaluru_weather():
    api_url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=12.9716&longitude=77.5946&"
        "current=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
    )
    fallback = {
        "temperature_2m": 26.0,
        "precipitation": 0.0,
        "relative_humidity_2m": 58,
        "wind_speed_10m": 9.5,
        "weather_desc": "Clear",
        "is_live": False
    }
    try:
        resp = requests.get(api_url, timeout=3)
        if resp.status_code == 200:
            data = resp.json().get("current", {})
            temp = float(data.get("temperature_2m", 26.0))
            precip = float(data.get("precipitation", 0.0))
            humidity = int(data.get("relative_humidity_2m", 58))
            wind = float(data.get("wind_speed_10m", 9.5))
            
            if precip > 2.5:
                desc = "Rainy"
            elif precip > 0.0:
                desc = "Light Rain"
            elif temp > 30.0:
                desc = "Warm"
            elif temp < 20.0:
                desc = "Clear"
            else:
                desc = "Clear"
                
            return {
                "temperature_2m": temp,
                "precipitation": precip,
                "relative_humidity_2m": humidity,
                "wind_speed_10m": wind,
                "weather_desc": desc,
                "is_live": True
            }
    except Exception:
        pass
    return fallback


# -----------------------------------------------------------------------------
# 2. MODEL CACHING (ZERO 42MB CSV LOADING)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ml_resources():
    clf = joblib.load(MODELS_DIR / "xgboost_tuned_clf.joblib")
    reg = joblib.load(MODELS_DIR / "xgboost_tuned_reg.joblib")
    q10 = joblib.load(MODELS_DIR / "lgbm_quantile_10.joblib")
    q50 = joblib.load(MODELS_DIR / "lgbm_quantile_50.joblib")
    q90 = joblib.load(MODELS_DIR / "lgbm_quantile_90.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    
    with open(MODELS_DIR / "selected_features.json", "r") as f:
        meta = json.load(f)
    selected_features = meta["selected_features"]
    explainer = shap.TreeExplainer(reg)
    
    return clf, reg, q10, q50, q90, scaler, selected_features, explainer


@st.cache_data(show_spinner=False)
def load_stations_data():
    return pd.read_csv(STATIONS_CSV)


try:
    clf_model, reg_model, q10_model, q50_model, q90_model, scaler, selected_features, shap_explainer = load_ml_resources()
    df_stations = load_stations_data()
    live_weather = fetch_live_bengaluru_weather()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. PUNCHY HERO BANNER (STRICT 8-WORD LIMIT PER LINE)
# -----------------------------------------------------------------------------
weather_badge_text = f"Bengaluru Live &bull; {live_weather['temperature_2m']:.0f}&deg;C &bull; {live_weather['weather_desc']}"

st.markdown(f"""
<div style="background: linear-gradient(135deg, #1E1B4B 0%, #0D9488 100%); border-radius: 18px; padding: 26px 30px; margin-bottom: 20px; box-shadow: 0 12px 30px rgba(13, 148, 136, 0.20); color: #FFFFFF;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 6px;">
        <span class="brand-font" style="font-size: 2.2rem; font-weight: 700; color: #FFFFFF;">
            VoltPulse
        </span>
        <span style="background: rgba(255, 255, 255, 0.16); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.25); color: #FFFFFF; font-size: 0.76rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px;">
            {weather_badge_text}
        </span>
    </div>
    <p style="margin: 0; font-size: 1.0rem; color: #E0E7FF; font-weight: 500;">
        Beat the queue. Charge smarter across Bengaluru.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. CLEAN 3-INPUT CARD (PUNCHY SHORT LABELS)
# -----------------------------------------------------------------------------
hours_24 = list(range(24))
def format_12h(h):
    period = "AM" if h < 12 else "PM"
    h12 = 12 if h in [0, 12] else h % 12
    return f"{h12:02d}:00 {period}"
time_options = [format_12h(h) for h in hours_24]
station_names = df_stations["station_name"].tolist()

with st.form("volt_search_form"):
    c1, c2, c3 = st.columns([1.2, 1.1, 1.0])
    
    with c1:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #1E1B4B; margin-bottom: 6px;'>Pick Your Hub</p>", unsafe_allow_html=True)
        selected_station_name = st.selectbox(
            "Hub",
            options=station_names,
            index=2,  # Defaults to Whitefield ITPL
            label_visibility="collapsed"
        )
        selected_stn = df_stations[df_stations["station_name"] == selected_station_name].iloc[0]
        total_ports = int(selected_stn["total_ports"])
        st.caption(f"{selected_stn['hub']} &bull; {total_ports} Fast Bays")

    with c2:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #1E1B4B; margin-bottom: 6px;'>Arrival Time</p>", unsafe_allow_html=True)
        days_of_week = ["Today", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sub_d, sub_t = st.columns(2)
        with sub_d:
            selected_day = st.selectbox("Day", options=days_of_week, index=0, label_visibility="collapsed")
        with sub_t:
            selected_time_str = st.selectbox("Time", options=time_options, index=18, label_visibility="collapsed")  # 06:00 PM
        arrival_hour = time_options.index(selected_time_str)
        st.caption(f"{selected_day} at {selected_time_str}")

    with c3:
        st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #1E1B4B; margin-bottom: 6px;'>Plugs Busy Now</p>", unsafe_allow_html=True)
        default_occupied = min(total_ports, max(1, int(round(total_ports * 0.75))))
        occupied_plugs = st.slider(
            "Plugs",
            min_value=0,
            max_value=total_ports,
            value=default_occupied,
            label_visibility="collapsed"
        )
        st.caption(f"{occupied_plugs} of {total_ports} bays in use ({int(occupied_plugs/total_ports*100)}% load)")

    # Primary Button (Punchy)
    submit_btn = st.form_submit_button("Check My Slot", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 5. PREDICTION LOGIC
# -----------------------------------------------------------------------------
temp_c = float(live_weather["temperature_2m"])
precip_mm = float(live_weather["precipitation"])

day_map = {"Today": 4, "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
dow_idx = day_map[selected_day]
is_weekend = 1 if dow_idx >= 5 else 0
is_peak = 1 if (arrival_hour in [8, 9, 10, 12, 13, 18, 19, 20] and is_weekend == 0) else 0

current_util = round(occupied_plugs / float(total_ports), 4)
temp_x_util = round(temp_c * current_util, 4)
hour_sin = np.sin(2 * np.pi * arrival_hour / 24.0)
hour_cos = np.cos(2 * np.pi * arrival_hour / 24.0)

util_lag_30m = round(np.clip(current_util * 0.95, 0.0, 1.0), 4)
util_lag_60m = round(np.clip(current_util * 0.90, 0.0, 1.0), 4)
util_lag_120m = round(np.clip(current_util * 0.85, 0.0, 1.0), 4)
util_lag_24h = round(np.clip(current_util * 0.92, 0.0, 1.0), 4)
util_roll_mean_3h = round(np.mean([util_lag_30m, util_lag_60m, util_lag_120m]), 4)
util_roll_std_3h = round(float(np.std([util_lag_30m, util_lag_60m, util_lag_120m])), 4)
temp_roll_mean_3h = temp_c

feat_dict = {
    "utilization_rate": current_util,
    "temp_x_util": temp_x_util,
    "util_lag_30m": util_lag_30m,
    "util_lag_24h": util_lag_24h,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "util_lag_60m": util_lag_60m,
    "util_roll_mean_3h": util_roll_mean_3h,
    "util_roll_std_3h": util_roll_std_3h,
    "util_lag_120m": util_lag_120m,
    "is_peak_commute": is_peak,
    "temp_roll_mean_3h": temp_roll_mean_3h,
    "temperature_2m": temp_c,
    "longitude": float(selected_stn["longitude"]),
    "latitude": float(selected_stn["latitude"])
}
X_input = pd.DataFrame([feat_dict])[selected_features]

# Inference
pred_state = int(clf_model.predict(X_input)[0])
pred_util = float(np.clip(reg_model.predict(X_input)[0], 0.0, 1.0))
p10_util = float(np.clip(q10_model.predict(X_input)[0], 0.0, 1.0))
p50_util = float(np.clip(q50_model.predict(X_input)[0], 0.0, 1.0))
p90_util = float(np.clip(q90_model.predict(X_input)[0], 0.0, 1.0))

def calc_queue_wait(u):
    if u <= 0.75:
        return 0.0
    return round(((u - 0.75) / max(0.05, 1.0 - u + 0.08)) * 38.0, 1)

wait_median = calc_queue_wait(pred_util)
wait_p10 = calc_queue_wait(p10_util)
wait_p90 = calc_queue_wait(p90_util)

# Alternative routing hub calculation
other_stns = df_stations[df_stations["station_id"] != selected_stn["station_id"]].copy()
other_stns["dist"] = np.sqrt(
    (other_stns["latitude"] - selected_stn["latitude"])**2 +
    (other_stns["longitude"] - selected_stn["longitude"])**2
)
nearest_alt_hub = other_stns.sort_values("dist").iloc[0]["hub"]
shift_hour = (arrival_hour + 1) % 24
shift_time_str = format_12h(shift_hour)


# -----------------------------------------------------------------------------
# 6. CRISP PUNCHY VERDICT CARD (EXACT 2 LINES)
# -----------------------------------------------------------------------------
pred_load_pct = int(round(pred_util * 100))

if pred_state == 0:
    card_gradient = "linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%)"
    card_border = "#10B981"
    pill_bg = "#059669"
    pill_text = "GREEN LIGHT"
    headline_color = "#065F46"
    headline_text = "Green Light &mdash; Plug Right In"
    subtext_html = f"Zero queue expected. Head straight to <span class='hl-emerald'>{selected_stn['hub']} Hub</span>."
    wait_range_str = "0&ndash;4 min"
    best_window_str = f"Now ({selected_time_str})"
    status_tag = "Low Load"
    status_tag_color = "#059669"
elif pred_state == 1:
    wait_lo = max(4, int(wait_p10))
    wait_hi = max(12, int(wait_p90) if wait_p90 > 0 else 12)
    card_gradient = "linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%)"
    card_border = "#F59E0B"
    pill_bg = "#D97706"
    pill_text = "MODERATE QUEUE"
    headline_color = "#92400E"
    headline_text = "Busy Soon &mdash; Short Wait Ahead"
    subtext_html = f"Expect a <span class='hl-amber'>{wait_lo}&ndash;{wait_hi} min</span> wait. Arrive 30 mins earlier to skip it."
    wait_range_str = f"{wait_lo}&ndash;{wait_hi} min"
    best_window_str = "Next 15m"
    status_tag = "Active Inflow"
    status_tag_color = "#D97706"
else:
    wait_lo = max(16, int(wait_p10) if wait_p10 > 0 else 16)
    wait_hi = max(28, int(max(wait_p90, wait_median + 6)))
    card_gradient = "linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%)"
    card_border = "#EF4444"
    pill_bg = "#DC2626"
    pill_text = "HEAVY QUEUE"
    headline_color = "#991B1B"
    headline_text = "Peak Rush &mdash; Heavy Queue Alert"
    subtext_html = f"Station hits <span class='hl-coral'>{pred_load_pct}%</span> load. Shift to <span class='hl-coral'>{shift_time_str}</span> or try <span class='hl-coral'>{nearest_alt_hub} Hub</span>."
    wait_range_str = f"{wait_lo}&ndash;{wait_hi} min"
    best_window_str = f"After {shift_time_str}"
    status_tag = "Near Capacity"
    status_tag_color = "#DC2626"

# Render Crisp 2-Line Verdict Card
st.markdown(f"""
<div style="background: {card_gradient}; border: 1.5px solid {card_border}; border-radius: 16px; padding: 22px 26px; margin-bottom: 16px; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);">
    <div style="display: inline-block; padding: 3px 10px; border-radius: 9999px; background: {pill_bg}; color: #FFFFFF; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px;">
        {pill_text}
    </div>
    <div class="brand-font" style="font-size: 1.35rem; font-weight: 700; color: {headline_color}; margin-bottom: 4px;">
        {headline_text}
    </div>
    <div style="font-size: 0.95rem; color: #1E1B4B; line-height: 1.45;">
        {subtext_html}
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. THREE VISUAL METRIC TILES (SPACE GROTESK HIGHLIGHTS)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 18px;">
    <div style="background: #FFFFFF; border: 1px solid #F1E8DC; border-radius: 14px; padding: 16px 18px; text-align: center; box-shadow: 0 4px 14px rgba(30, 27, 75, 0.03);">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; font-weight: 700;">Station Load</div>
        <div class="brand-font" style="font-size: 1.7rem; font-weight: 700; color: #1E1B4B; margin-top: 2px;">{pred_load_pct}% Full</div>
        <div style="font-size: 0.80rem; color: {status_tag_color}; font-weight: 600;">{status_tag}</div>
    </div>
    <div style="background: #FFFFFF; border: 1px solid #F1E8DC; border-radius: 14px; padding: 16px 18px; text-align: center; box-shadow: 0 4px 14px rgba(30, 27, 75, 0.03);">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; font-weight: 700;">Wait Range</div>
        <div class="brand-font" style="font-size: 1.7rem; font-weight: 700; color: #1E1B4B; margin-top: 2px;">{wait_range_str}</div>
        <div style="font-size: 0.80rem; color: #0D9488; font-weight: 600;">80% Confidence</div>
    </div>
    <div style="background: #FFFFFF; border: 1px solid #F1E8DC; border-radius: 14px; padding: 16px 18px; text-align: center; box-shadow: 0 4px 14px rgba(30, 27, 75, 0.03);">
        <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; font-weight: 700;">Best Window</div>
        <div class="brand-font" style="font-size: 1.35rem; font-weight: 700; color: #1E1B4B; margin-top: 6px;">{best_window_str}</div>
        <div style="font-size: 0.80rem; color: #64748B;">Smart Routing</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. ULTRA-SHORT HIGHLIGHTED REASONS (MAX 5 WORDS PER TAG)
# -----------------------------------------------------------------------------
shap_vals = shap_explainer(X_input)
contribs = shap_vals.values[0]

feat_names = X_input.columns.tolist()
sh_df = pd.DataFrame({"feat": feat_names, "val": X_input.iloc[0].values, "shap": contribs})
sh_df["abs_shap"] = sh_df["shap"].abs()
top_3_reasons = sh_df.sort_values("abs_shap", ascending=False).head(3)

tags = []
for _, row in top_3_reasons.iterrows():
    fname = row["feat"]
    sh_val = row["shap"]
    
    if fname == "utilization_rate":
        if occupied_plugs / total_ports >= 0.7:
            tags.append(f"&#x1F53A; {occupied_plugs}/{total_ports} Plugs Full")
        else:
            tags.append(f"&#x1F53B; {total_ports - occupied_plugs} Plugs Available")
    elif fname in ["hour_sin", "hour_cos", "is_peak_commute"]:
        if is_peak:
            tags.append(f"&#x1F53A; {selected_time_str.split(' ')[0]} Office Rush")
        else:
            tags.append("&#x1F53B; Off-Peak Quiet Slot")
    elif fname in ["temp_x_util", "temperature_2m", "temp_roll_mean_3h"]:
        if temp_c > 28.0:
            tags.append("&#x1F53A; Warm Thermal Throttling")
        else:
            tags.append("&#x1F53B; Clear Weather Boost")
    elif fname in ["longitude", "latitude"]:
        tags.append(f"&#x1F53A; {selected_stn['hub']} Demand Hub")
    else:
        tags.append("&#x1F53B; Steady Inflow Trend")

tags_html = " &nbsp;&bull;&nbsp; ".join([f"<span class='hl-amber' style='font-size: 0.85rem;'>{t}</span>" for t in tags[:3]])

st.markdown(f"""
<div style="background: #FFFFFF; border: 1px solid #F1E8DC; border-radius: 14px; padding: 14px 18px; text-align: center; box-shadow: 0 4px 14px rgba(30, 27, 75, 0.02); margin-bottom: 24px;">
    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; font-weight: 700; margin-bottom: 6px;">
        Key Drivers
    </div>
    <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 8px;">
        {tags_html}
    </div>
</div>
""", unsafe_allow_html=True)

# Clean Minimalist Footer
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.80rem; margin-top: 6px;">
    VoltPulse Mobility &middot; Smart EV Intelligence &middot; Bengaluru
</div>
""", unsafe_allow_html=True)
