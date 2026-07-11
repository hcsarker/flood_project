"""
main.py
--------
Flood Prediction and Drainage Simulation — Full Project Pipeline

Run this file to execute the entire project:
  1. Load dataset (real Kaggle CSV if present in data/, else synthetic demo data)
  2. EDA (correlation with target, distribution)
  3. Train ML regression models (Linear Regression, Random Forest) -> flood probability prediction
  4. Run drainage simulation (Rational Method) -> waterlogging/overflow detection
  5. Save all plots + a summary report + metrics.json into outputs/

Usage:
    python main.py
"""

import os
import sys
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from generate_data import generate_synthetic_flood_data
from flood_prediction import load_dataset, run_eda, train_models, TARGET
from drainage_simulation import run_drainage_simulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
REAL_DATA_PATH = os.path.join(DATA_DIR, "flood_data.csv")   # <- your real Kaggle CSV
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "flood_data_synthetic.csv")


def get_dataset() -> pd.DataFrame:
    if os.path.exists(REAL_DATA_PATH):
        print(f"[INFO] Real dataset found -> using {REAL_DATA_PATH}")
        df = load_dataset(REAL_DATA_PATH)
        if TARGET not in df.columns:
            raise ValueError(
                f"Your CSV must contain a target column named '{TARGET}' (0-1 continuous). "
                "Rename your target column accordingly, or edit flood_prediction.py."
            )
    else:
        print("[INFO] No real Kaggle CSV found in data/flood_data.csv")
        print("[INFO] Generating synthetic demo dataset instead...")
        df = generate_synthetic_flood_data()
        df.to_csv(SYNTHETIC_DATA_PATH, index=False)
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n================ STEP 1: LOAD DATA ================")
    df = get_dataset()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")

    print("\n================ STEP 2: EDA ================")
    run_eda(df, OUTPUT_DIR)

    print("\n================ STEP 3: ML FLOOD PREDICTION (REGRESSION) ================")
    results = train_models(df, OUTPUT_DIR)

    print("\n================ STEP 4: DRAINAGE SIMULATION ================")
    drainage_df = run_drainage_simulation(OUTPUT_DIR)
    drainage_df.to_csv(os.path.join(OUTPUT_DIR, "drainage_simulation_results.csv"), index=False)

    print("\n================ STEP 5: SUMMARY REPORT + METRICS ================")
    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {
            "r2": round(float(res["r2"]), 4),
            "mae": round(float(res["mae"]), 4),
            "rmse": round(float(res["rmse"]), 4),
        }
    rf_importances = results["Random Forest"]["importances"].sort_values(ascending=False)
    metrics_out["feature_importance"] = [[k, round(float(v), 4)] for k, v in rf_importances.items()]
    lin_coefs = results["Linear Regression"]["coefficients"].sort_values(ascending=False)
    metrics_out["linear_coefficients"] = [[k, round(float(v), 5)] for k, v in lin_coefs.items()]
    metrics_out["target_mean"] = round(float(df[TARGET].mean()), 5)
    metrics_out["target_std"] = round(float(df[TARGET].std()), 5)
    metrics_out["feature_ranges"] = {
        col: [float(df[col].min()), float(df[col].max())] for col in df.columns if col != TARGET
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
        f.write("FLOOD PREDICTION AND DRAINAGE SIMULATION - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: real Kaggle Flood Prediction dataset\n")
        f.write(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Target: {TARGET} (continuous, 0-1) -> mean {metrics_out['target_mean']}, "
                f"std {metrics_out['target_std']}\n\n")

        f.write("MODULE 1: ML MODEL PERFORMANCE (Regression)\n")
        f.write("-" * 60 + "\n")
        for name in ["Linear Regression", "Random Forest"]:
            m = metrics_out[name]
            f.write(f"{name}:\n")
            f.write(f"   R2 Score : {m['r2']:.4f}\n")
            f.write(f"   MAE      : {m['mae']:.4f}\n")
            f.write(f"   RMSE     : {m['rmse']:.4f}\n\n")

        f.write("Top 5 most important features (Random Forest):\n")
        for feat, val in metrics_out["feature_importance"][:5]:
            f.write(f"   {feat}: {val}\n")
        f.write("\n")

        f.write("MODULE 2: DRAINAGE SIMULATION\n")
        f.write("-" * 60 + "\n")
        overflow_hours = drainage_df[drainage_df["Overflow"]]["Hour"].tolist()
        f.write(f"Simulated storm duration: 48 hours\n")
        f.write(f"Overflow (waterlogging) hours: {overflow_hours}\n")
        f.write(f"Total overflow duration: {len(overflow_hours)} hour(s)\n\n")

        f.write("Generated files (in outputs/):\n")
        f.write(" 1_correlation_with_target.png\n")
        f.write(" 2_flood_probability_distribution.png\n")
        f.write(" 3_predicted_vs_actual.png\n")
        f.write(" 4_feature_importance.png\n")
        f.write(" 5_model_comparison.png\n")
        f.write(" 6_rainfall_event.png\n")
        f.write(" 7_drainage_simulation.png\n")
        f.write(" metrics.json (for dashboard)\n")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
