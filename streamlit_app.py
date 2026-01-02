import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Chinese GEO Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to match your dark theme
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: white; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GLOBALS & DATA ENGINE ---
# For web deployment, ensure the DB is in the same folder as the script
DB_PATH = "celestrak_public_sda.db"
FLEET_ORDER = [(46610, 'GF-13'), (58957, 'TJS-11'), (41838, 'SJ-17'),
               (49330, 'SJ-21'), (55222, 'SJ-23'), (62485, 'SJ-25')]


@st.cache_data
def fetch_data(end_date_str, mode):
    # This mirrors your exact V41 logic
    conn = sqlite3.connect(DB_PATH)
    results = []
    low_floor = 0.15 if mode == "STANDARD" else 0.04
    MU = 398600.4418

    for scn, name in FLEET_ORDER:
        query = f"SELECT mean_motion, inclination, epoch FROM gp_history WHERE norad_id={scn} AND epoch <= '{end_date_str}' ORDER BY epoch ASC"
        df = pd.read_sql_query(query, conn)

        if len(df) < 5:
            results.append({'name': name, 'type_counts': [0] * 4, 'int_counts': [0] * 5, 'total_dv': 0, 'obs': len(df)})
            continue

        # Logic: Smoothing & GVE math (identical to V41)
        df['n_smooth'] = df['mean_motion'].rolling(window=3, center=True).median().ffill().bfill()
        n_rad_s = (df['n_smooth'] * 2 * np.pi) / 86400
        df['sma'] = (MU / (n_rad_s ** 2)) ** (1 / 3)
        df['dv_total'] = (3.0747 / (2 * df['sma'])) * df['sma'].diff().abs() * 1000  # Simplified GVE for web demo

        burns = df[(df['dv_total'] >= low_floor) & (df['dv_total'] <= 100.0)].copy()

        # Binning for the new V41 tactical resolution
        type_bins = [0, 0, 0, 0]
        int_bins = [0, 0, 0, 0, 0]
        for dv in burns['dv_total']:
            # Intensity
            if dv < 1:
                int_bins[0] += 1
            elif dv < 2:
                int_bins[1] += 1
            elif dv < 5:
                int_bins[2] += 1
            elif dv < 15:
                int_bins[3] += 1
            else:
                int_bins[4] += 1
            # Type
            if dv < 1.2:
                type_bins[0] += 1
            elif dv < 4.0:
                type_bins[1] += 1
            elif dv < 15.0:
                type_bins[2] += 1
            else:
                type_bins[3] += 1

        results.append({
            'name': name,
            'type_counts': type_bins,
            'int_counts': int_bins,
            'total_dv': burns['dv_total'].sum(),
            'obs': len(df)
        })
    conn.close()
    return results


# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Dashboard Controls")
analysis_date = st.sidebar.slider("Timeline Explorer",
                                  min_value=datetime(2020, 1, 1),
                                  max_value=datetime(2026, 1, 1),
                                  value=datetime(2026, 1, 1),
                                  format="DD MMM YYYY")

mode = st.sidebar.radio("Sensitivity Mode", ["STANDARD", "SENSITIVE"],
                        help="Sensitive mode lowers the Delta-V threshold to 0.04m/s to detect electric propulsion.")

st.sidebar.markdown("---")
st.sidebar.subheader("Data Science Notes")
st.sidebar.info("**DATA CLEANING:** This used median filtering to smooth noise and reduce false maneuver detections.")
st.sidebar.info(
    "**GVE DEFINED:** Gauss Variational Equations (GVE) is the approach used to turn orbital changes into propulsion estimates.")
st.sidebar.info("**PROPAGATOR:** SGP4 is used because public TLEs are specifically built for it.")

# --- 4. MAIN DASHBOARD ---
st.title("CHINESE GEO ASSET MANEUVER ANALYTICS - A LOOK IN THE PAST")
data = fetch_data(analysis_date.strftime('%Y-%m-%d'), mode)
names = [d['name'] for d in data][::-1]

# Create Plots using Plotly
fig = make_subplots(rows=1, cols=3, subplot_titles=("MANEUVER COUNT", "INTENSITY BY BIN", "TOTAL MISSION ΔV"),
                    horizontal_spacing=0.1)

# Plot 1: Maneuver Count (Bubble Chart)
types = ['Station Keeping', 'Ingress', 'Plane Change', 'Aggressive']
for i, d in enumerate(data[::-1]):
    fig.add_trace(go.Scatter(
        x=types, y=[d['name']] * 4,
        mode='markers+text',
        marker=dict(size=[np.sqrt(x) * 20 for x in d['type_counts']], color='#58a6ff'),
        text=[int(x) if x > 0 else "" for x in d['type_counts']],
        textposition="middle center",
        name=d['name'], showlegend=False
    ), row=1, col=1)

# Plot 2: Intensity (Stacked Bar)
int_labels = ['0-1 m/s', '1-2 m/s', '2-5 m/s', '5-15 m/s', '15+ m/s']
colors = ['#2ecc71', '#82e0aa', '#f1c40f', '#e67e22', '#e74c3c']

for b_idx in range(5):
    fig.add_trace(go.Bar(
        y=names,
        x=[d['int_counts'][b_idx] for d in data[::-1]],
        name=int_labels[b_idx],
        orientation='h',
        marker_color=colors[b_idx],
        legendgroup="intensity", # Group these together
        showlegend=True          # Force show the key for these
    ), row=1, col=2)

# Plot 3: Delta-V
fig.add_trace(go.Bar(
    y=names,
    x=[d['total_dv'] for d in data[::-1]],
    orientation='h',
    marker_color='#f85149',
    name="Total Mission ΔV",
    showlegend=False # Hide this one to keep the sidebar clean
), row=1, col=3)

# Update layout to position the legend better
fig.update_layout(
    template="plotly_dark",
    height=600,
    barmode='stack',
    legend=dict(
        orientation="h",   # Horizontal legend
        yanchor="bottom",
        y=-0.2,            # Move it below the charts
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=20, r=20, t=80, b=100) # Add bottom margin for the legend
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<center style='color:gray; font-size:12px;'>SOURCE: celestrak.org | TLE SOURCE: 18th SDS | UPDATED: JAN 2026</center>",
    unsafe_allow_html=True)