from pathlib import Path
import numpy as np
import pandas as pd

SENSOR_COLUMNS = ["temperature", "vibration", "current", "pressure"]

def generate_sensor_data(n=2400, seed=42):
    """Generate realistic-looking DEMO telemetry. Labels are synthetic, not real failures."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="5min")
    machine_ids = rng.choice(["MTR-001", "PMP-002", "FAN-003", "CMP-004"], size=n, p=[.32, .26, .22, .20])
    age_days = rng.integers(30, 1800, size=n)
    load = np.clip(rng.normal(0.66, 0.17, size=n), 0.1, 1.2)
    degradation = np.clip((age_days / 1800) * .45 + rng.normal(.12, .09, n), 0, 1)
    temperature = 28 + 28*load + 18*degradation + rng.normal(0, 3.2, n)
    vibration = .4 + 2.2*load + 3.0*degradation + rng.normal(0, .45, n)
    current = 2.2 + 8.8*load + 3.5*degradation + rng.normal(0, .8, n)
    pressure = 2.4 + 1.8*load - .7*degradation + rng.normal(0, .22, n)
    # The target is generated from sensor-risk signals with noise, suitable for a demo classifier.
    risk_score = (
        (temperature > 72).astype(float)*.27 +
        (vibration > 3.5).astype(float)*.32 +
        (current > 10.5).astype(float)*.18 +
        (pressure < 2.5).astype(float)*.12 +
        degradation*.18 + rng.normal(0, .10, n)
    )
    failure = (risk_score > .52).astype(int)
    df = pd.DataFrame({
        "timestamp": timestamps, "machine_id": machine_ids, "age_days": age_days,
        "load": load.round(3), "temperature": temperature.round(2),
        "vibration": vibration.round(3), "current": current.round(3),
        "pressure": pressure.round(3), "failure": failure
    })
    # A few missing readings imitate imperfect telemetry; pipeline handles them.
    for col in SENSOR_COLUMNS:
        mask = rng.random(n) < .006
        df.loc[mask, col] = np.nan
    return df

def load_or_create_data(path="data/generated/iot_sensor_data.csv", n=2400):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    df = generate_sensor_data(n=n)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df

def sensor_status(row):
    vals = row
    if vals["temperature"] >= 78 or vals["vibration"] >= 4.2 or vals["current"] >= 12:
        return "Critical"
    if vals["temperature"] >= 68 or vals["vibration"] >= 3.2 or vals["current"] >= 10.2 or vals["pressure"] < 2.5:
        return "Warning"
    return "Healthy"
