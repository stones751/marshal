from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Iterable, List

import numpy as np

from .database import Asset, Telemetry, Fault, get_session, select, func


TEMPERATURE_BASE = {"generator": 60.0, "transformer": 55.0}
VIBRATION_BASE = {"generator": 2.5, "transformer": 1.5}
VOLTAGE_BASE = {"generator": 400.0, "transformer": 11000.0}


def _random_fault(asset: Asset) -> Fault | None:
    roll = random.random()
    if roll < 0.02:
        code, severity, desc = ("OVERTEMP", "critical", "Temperature exceeded threshold")
    elif roll < 0.04:
        code, severity, desc = ("VIB", "warn", "Vibration trending high")
    elif roll < 0.06:
        code, severity, desc = ("VOLT", "warn", "Voltage deviation detected")
    else:
        return None
    return Fault(asset_id=asset.id, timestamp=datetime.utcnow(), code=code, severity=severity, description=desc)


def generate_sine_noise(base: float, amplitude: float, noise: float, step: int, period: int = 144) -> float:
    return base + amplitude * np.sin(2 * np.pi * step / period) + random.gauss(0, noise)


def ingest_synthetic(minutes: int = 24 * 60, step_minutes: int = 5) -> None:
    """Generate and insert synthetic telemetry and faults for all assets."""
    from .database import upsert_default_assets

    upsert_default_assets()

    with get_session() as session:
        assets: List[Asset] = list(session.query(Asset).all())

    now = datetime.utcnow()
    steps = minutes // step_minutes

    with get_session() as session:
        for asset in assets:
            for i in range(steps):
                ts = now - timedelta(minutes=(steps - i) * step_minutes)
                temp = generate_sine_noise(TEMPERATURE_BASE[asset.type], amplitude=8.0, noise=1.5, step=i)
                vib = max(0.2, generate_sine_noise(VIBRATION_BASE[asset.type], amplitude=0.8, noise=0.2, step=i))
                volt = generate_sine_noise(VOLTAGE_BASE[asset.type], amplitude=5.0 if asset.type == "generator" else 120.0, noise=2.0 if asset.type == "generator" else 40.0, step=i)

                session.add(Telemetry(asset_id=asset.id, timestamp=ts, temperature_c=float(temp), vibration_mm_s=float(vib), voltage_v=float(volt)))

                fault = _random_fault(asset)
                if fault and random.random() < 0.3:  # not every step
                    session.add(fault)
        session.commit()
