from src.data_utils import generate_sensor_data
from pathlib import Path

if __name__ == "__main__":
    out = Path("data/generated/iot_sensor_data.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sensor_data(n=2400, seed=42)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} synthetic sensor rows -> {out}")
    print(df.head(5).to_string(index=False))
    print("\nSynthetic label distribution:")
    print(df["failure"].value_counts().rename(index={0:"normal",1:"failure"}))
