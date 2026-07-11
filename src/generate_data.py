"""
generate_data.py
-----------------
Generates a synthetic, Kaggle-style Flood Prediction dataset for
demo/testing purposes. If you have downloaded a real dataset from
Kaggle (e.g. search "Flood Prediction Dataset" or "Flood Prediction
Factors" on kaggle.com), just place the CSV file inside the
`data/` folder and rename it to `flood_data.csv` — main.py will
automatically use it instead of this synthetic one.

Feature columns mimic common real-world flood datasets:
    Rainfall (mm), Temperature (C), Humidity (%),
    RiverDischarge (m3/s), WaterLevel (m), Elevation (m),
    DrainageCapacity (m3/s), Urbanization (%),
    Deforestation (%), HistoricalFloods (count in last 10 yrs)
Target:
    FloodOccurred (0 = No, 1 = Yes)
"""

import numpy as np
import pandas as pd

np.random.seed(42)


def generate_synthetic_flood_data(n_samples: int = 2000) -> pd.DataFrame:
    rainfall = np.random.gamma(shape=2.0, scale=40, size=n_samples)          # mm
    temperature = np.random.normal(28, 4, n_samples)                        # C
    humidity = np.clip(np.random.normal(75, 12, n_samples), 20, 100)         # %
    river_discharge = np.random.gamma(shape=3.0, scale=80, size=n_samples)   # m3/s
    water_level = np.random.normal(4, 1.5, n_samples) + rainfall * 0.01     # m
    elevation = np.random.uniform(1, 50, n_samples)                        # m
    drainage_capacity = np.random.normal(150, 40, n_samples)                # m3/s
    urbanization = np.clip(np.random.normal(55, 20, n_samples), 0, 100)      # %
    deforestation = np.clip(np.random.normal(30, 15, n_samples), 0, 100)     # %
    historical_floods = np.random.poisson(2, n_samples)                     # count

    # A rough physical logic to decide flood probability, then sample outcome
    risk_score = (
        0.035 * rainfall
        + 0.02 * river_discharge
        + 0.5 * water_level
        - 0.03 * elevation
        - 0.02 * drainage_capacity
        + 0.015 * urbanization
        + 0.01 * deforestation
        + 0.8 * historical_floods
    )
    risk_score = (risk_score - risk_score.mean()) / risk_score.std()
    flood_prob = 1 / (1 + np.exp(-risk_score))  # sigmoid -> probability
    flood_occurred = np.random.binomial(1, flood_prob)

    df = pd.DataFrame({
        "Rainfall_mm": rainfall.round(2),
        "Temperature_C": temperature.round(2),
        "Humidity_pct": humidity.round(2),
        "RiverDischarge_m3s": river_discharge.round(2),
        "WaterLevel_m": water_level.round(2),
        "Elevation_m": elevation.round(2),
        "DrainageCapacity_m3s": drainage_capacity.round(2),
        "Urbanization_pct": urbanization.round(2),
        "Deforestation_pct": deforestation.round(2),
        "HistoricalFloods_count": historical_floods,
        "FloodOccurred": flood_occurred,
    })
    return df


if __name__ == "__main__":
    df = generate_synthetic_flood_data()
    df.to_csv("/home/claude/flood_project/data/flood_data_synthetic.csv", index=False)
    print("Synthetic dataset saved:", df.shape)
    print(df["FloodOccurred"].value_counts(normalize=True))
