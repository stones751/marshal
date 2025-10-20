from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .database import Telemetry


@dataclass
class HealthIndicator:
    score: float  # 0-100
    status: str  # Good/Watch/Service Required
    drivers: List[str]


@dataclass
class TrendInsight:
    metric: str
    slope_per_hour: float
    confidence: float
    interpretation: str
    action: Optional[str]


def compute_health_indicator(
    df: pd.DataFrame,
    asset_type: Optional[str] = None,
    nominal_voltage: Optional[float] = None,
) -> HealthIndicator:
    """Compute 0-100 health score from latest telemetry row.

    Heuristics:
    - Temperature: 40% weight. 40C ideal; 90C critical threshold.
    - Vibration: 30% weight. 2.0 mm/s good; 7.1 mm/s ISO 10816 alarm.
    - Voltage: 30% weight. Nominal depends on asset (generator ~400V, transformer ~11kV) with +/-10% acceptable band.
    """
    if df.empty:
        return HealthIndicator(score=0.0, status="No Data", drivers=["No telemetry available"]) 

    latest = df.sort_values("timestamp").iloc[-1]
    temperature_c = float(latest["temperature_c"])  
    vibration = float(latest["vibration_mm_s"])  
    voltage = float(latest["voltage_v"])  

    # Determine nominal voltage if not supplied
    if nominal_voltage is None:
        if asset_type == "transformer":
            nominal = 11000.0
        elif asset_type == "generator":
            nominal = 400.0
        else:
            # infer from historical values
            median_v = float(df["voltage_v"].median()) if not df.empty else 400.0
            nominal = 11000.0 if median_v > 2000.0 else 400.0
    else:
        nominal = float(nominal_voltage)

    # Normalize sub-scores to 0..100 (higher is healthier)
    temp_score = np.clip(100 - (max(0.0, temperature_c - 40) / (90 - 40) * 100), 0, 100)
    vib_score = np.clip(100 - (max(0.0, vibration - 2.0) / (7.1 - 2.0) * 100), 0, 100)
    volt_dev = abs(voltage - nominal)
    volt_score = np.clip(100 - (max(0.0, volt_dev - 0) / (0.10 * nominal) * 100), 0, 100)

    score = 0.4 * temp_score + 0.3 * vib_score + 0.3 * volt_score

    if score >= 80:
        status = "Good"
    elif score >= 60:
        status = "Watch"
    else:
        status = "Service Required"

    drivers: List[str] = []
    if temperature_c >= 80:
        drivers.append("High temperature")
    if vibration >= 5.0:
        drivers.append("Elevated vibration")
    if volt_dev > 0.10 * nominal:
        drivers.append("Voltage out of band")

    return HealthIndicator(score=float(round(score, 1)), status=status, drivers=drivers)


def fit_trend(df: pd.DataFrame, metric: str) -> Optional[TrendInsight]:
    """Fit a simple linear regression over time and derive slope/hour and interpretation."""
    if df.empty or df[metric].nunique() <= 1:
        return None

    # Convert timestamps to hours since first point to keep scale reasonable
    series = df.dropna(subset=[metric]).sort_values("timestamp")
    if len(series) < 4:
        return None

    t0 = series["timestamp"].iloc[0]
    x = (series["timestamp"] - t0).dt.total_seconds().to_numpy() / 3600.0
    y = series[metric].to_numpy()

    # linear regression via least squares
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # crude confidence from R^2
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    interpretation = "Stable"
    action: Optional[str] = None

    if metric == "temperature_c":
        if m > 1.0:  # >1 C per hour
            interpretation = "Rising fast"
            action = "Check cooling, lubrication, and load."
        elif m > 0.2:
            interpretation = "Rising"
        elif m < -0.2:
            interpretation = "Falling"
    elif metric == "vibration_mm_s":
        if m > 0.2:
            interpretation = "Increasing"
            action = "Inspect bearings, misalignment, looseness."
        elif m < -0.1:
            interpretation = "Decreasing"
    elif metric == "voltage_v":
        if m > 2.0:
            interpretation = "Increasing"
        elif m < -2.0:
            interpretation = "Decreasing"

    return TrendInsight(metric=metric, slope_per_hour=float(round(m, 3)), confidence=float(round(r2, 3)), interpretation=interpretation, action=action)


def trend_summary(df: pd.DataFrame) -> List[TrendInsight]:
    insights: List[TrendInsight] = []
    for metric in ["temperature_c", "vibration_mm_s", "voltage_v"]:
        ins = fit_trend(df, metric)
        if ins:
            insights.append(ins)
    return insights
