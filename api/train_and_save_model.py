"""
train_and_save_model.py
------------------------
Trains Linear Regression + Random Forest on the real Kaggle flood
dataset and saves them to disk (models/*.joblib) so the FastAPI
server can load them instantly without retraining every time.

Run this once (or whenever data/flood_data.csv changes):
    python api/train_and_save_model.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flood_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

TARGET = "FloodProbability"

FEATURE_ORDER = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds", "DeterioratingInfrastructure",
    "PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        print("Place your Kaggle flood_data.csv there first.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURE_ORDER + [TARGET] if c not in df.columns]
    if missing:
        print(f"[ERROR] Dataset is missing expected columns: {missing}")
        sys.exit(1)

    X = df[FEATURE_ORDER]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Linear Regression...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    print(f"  R2={r2_score(y_test, lin_pred):.4f}  "
          f"MAE={mean_absolute_error(y_test, lin_pred):.5f}  "
          f"RMSE={mean_squared_error(y_test, lin_pred) ** 0.5:.5f}")

    print("Training Random Forest (this takes ~20-40s)...")
    sample_n = min(20000, len(X_train))
    X_train_rf = X_train.sample(sample_n, random_state=42)
    y_train_rf = y_train.loc[X_train_rf.index]
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_rf, y_train_rf)
    rf_pred = rf_model.predict(X_test)
    print(f"  R2={r2_score(y_test, rf_pred):.4f}  "
          f"MAE={mean_absolute_error(y_test, rf_pred):.5f}  "
          f"RMSE={mean_squared_error(y_test, rf_pred) ** 0.5:.5f}")

    joblib.dump(lin_model, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(FEATURE_ORDER, os.path.join(MODELS_DIR, "feature_order.joblib"))

    print(f"\nSaved models to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
