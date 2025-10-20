from __future__ import annotations

import os
from datetime import datetime, timedelta, date
from typing import List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select, func

from .database import init_db, get_session, Asset, Telemetry, Fault, upsert_default_assets
from .analytics import compute_health_indicator, trend_summary


st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")


@st.cache_resource
def _init():
    init_db()
    upsert_default_assets()
    return True


_init()


def load_assets() -> List[Asset]:
    with get_session() as session:
        return list(session.query(Asset).order_by(Asset.name).all())


def load_telemetry(asset_id: int, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    with get_session() as session:
        q = session.query(Telemetry).filter(Telemetry.asset_id == asset_id)
        if start:
            q = q.filter(Telemetry.timestamp >= start)
        if end:
            q = q.filter(Telemetry.timestamp <= end)
        rows = q.order_by(Telemetry.timestamp.asc()).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "temperature_c", "vibration_mm_s", "voltage_v"])  
    return pd.DataFrame(
        {
            "timestamp": [r.timestamp for r in rows],
            "temperature_c": [r.temperature_c for r in rows],
            "vibration_mm_s": [r.vibration_mm_s for r in rows],
            "voltage_v": [r.voltage_v for r in rows],
        }
    )


def load_faults(asset_id: int, limit: int = 100) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(Fault)
            .filter(Fault.asset_id == asset_id)
            .order_by(Fault.timestamp.desc())
            .limit(limit)
            .all()
        )
    if not rows:
        return pd.DataFrame(columns=["timestamp", "code", "severity", "description"])  
    return pd.DataFrame(
        {
            "timestamp": [r.timestamp for r in rows],
            "code": [r.code for r in rows],
            "severity": [r.severity for r in rows],
            "description": [r.description for r in rows],
        }
    )


def gauge(title: str, value: float, min_v: float, max_v: float, steps: Optional[List[tuple]] = None) -> go.Figure:
    steps = steps or []
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [min_v, max_v]},
                "bar": {"color": "#2c7be5"},
                "steps": [{"range": r, "color": c} for r, c in steps],
            },
        )
    )


def line_chart(df: pd.DataFrame, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y, height=300)
    return fig


# Sidebar filters
st.sidebar.header("Filters")
assets = load_assets()
asset_names = [f"{a.name} ({a.type})" for a in assets]
asset_map = {f"{a.name} ({a.type})": a for a in assets}
selected_asset_label = st.sidebar.selectbox("Asset", asset_names)
selected_asset = asset_map[selected_asset_label]

range_label = st.sidebar.selectbox("Time Range", ["24h", "7d", "30d", "Custom"])
start: Optional[datetime] = None
end: Optional[datetime] = None
now = datetime.utcnow()
if range_label == "24h":
    start = now - timedelta(hours=24)
elif range_label == "7d":
    start = now - timedelta(days=7)
elif range_label == "30d":
    start = now - timedelta(days=30)
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_d: date = st.date_input("Start", value=(now - timedelta(days=7)).date())
    with col2:
        end_d: date = st.date_input("End", value=now.date())
    # convert to datetimes spanning full days
    start = datetime.combine(start_d, datetime.min.time())
    end = datetime.combine(end_d, datetime.max.time())

# Load data
telemetry_df = load_telemetry(selected_asset.id, start, end)

# Health indicator
health = compute_health_indicator(
    telemetry_df,
    asset_type=selected_asset.type,
    nominal_voltage=12000.0 if selected_asset.type == "transformer" else 400.0,
)

st.subheader(f"Health: {health.status} ({health.score}%)")
if health.drivers:
    st.caption("; ".join(health.drivers))

# Gauges row
if not telemetry_df.empty:
    latest = telemetry_df.sort_values("timestamp").iloc[-1]
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            gauge(
                "Temperature (C)",
                float(latest["temperature_c"]),
                0,
                120,
                steps=[([0, 70], "#34c38f"), ([70, 90], "#f1b44c"), ([90, 120], "#f46a6a")],
            ),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            gauge(
                "Vibration (mm/s)",
                float(latest["vibration_mm_s"]),
                0,
                10,
                steps=[([0, 3], "#34c38f"), ([3, 5], "#f1b44c"), ([5, 10], "#f46a6a")],
            ),
            use_container_width=True,
        )
    with g3:
        max_v = 12000.0 if selected_asset.type == "transformer" else 500.0
        st.plotly_chart(
            gauge(
                "Voltage (V)",
                float(latest["voltage_v"]),
                0,
                max_v,
                steps=[([0, 0.9 * max_v], "#34c38f"), ([0.9 * max_v, max_v], "#f1b44c")],
            ),
            use_container_width=True,
        )

# Line charts
st.subheader("Telemetry Trends")
chart_cols = st.columns(3)
for col, metric, title in zip(
    chart_cols,
    ["temperature_c", "vibration_mm_s", "voltage_v"],
    ["Temperature", "Vibration", "Voltage"],
):
    with col:
        if telemetry_df.empty:
            st.info("No telemetry available for selected range.")
        else:
            st.plotly_chart(line_chart(telemetry_df, metric, f"{title} over time"), use_container_width=True)

# Trend insights
st.subheader("Trend Analysis")
insights = trend_summary(telemetry_df)
if not insights:
    st.info("Not enough data for trend analysis.")
else:
    for ins in insights:
        cols = st.columns([2, 2, 2, 6])
        cols[0].markdown(f"**{ins.metric}**")
        cols[1].markdown(f"Slope/hr: `{ins.slope_per_hour}`")
        cols[2].markdown(f"Confidence: `{ins.confidence}`")
        cols[3].markdown(f"Interpretation: {ins.interpretation} {(' — ' + ins.action) if ins.action else ''}")

# Fault history
st.subheader("Fault History")
faults_df = load_faults(selected_asset.id)
if faults_df.empty:
    st.info("No recent faults.")
else:
    st.dataframe(faults_df, use_container_width=True)

st.caption("Use the simulator to populate data: `python -m app.run_sim`.")
