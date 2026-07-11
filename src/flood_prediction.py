"""
flood_prediction.py
--------------------
Module 1: Flood Prediction using Machine Learning.

The real Kaggle dataset's target `FloodProbability` is a CONTINUOUS
value (0-1), so this is a REGRESSION problem, not classification.

Trains and compares Linear Regression and Random Forest Regressor
to predict FloodProbability from 20 environmental/human risk factors
(MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation,
Urbanization, ClimateChange, DamsQuality, Siltation, ...).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGET = "FloodProbability"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def run_eda(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 9))
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    colors = ["#C13F3F" if v < 0 else "#4FA88B" for v in corr.values]
    corr.plot(kind="barh", color=colors)
    plt.title("Correlation with Flood Probability")
    plt.xlabel("Correlation coefficient")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_correlation_with_target.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=50, color="#2E86AB", kde=True)
    plt.title("Distribution of Flood Probability")
    plt.xlabel("Flood Probability")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_flood_probability_distribution.png", dpi=150)
    plt.close()


def train_models(df: pd.DataFrame, output_dir: str, sample_for_rf: int = 20000):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)
    y_pred_lin = lin_model.predict(X_test_scaled)
    results["Linear Regression"] = {
        "r2": r2_score(y_test, y_pred_lin),
        "mae": mean_absolute_error(y_test, y_pred_lin),
        "rmse": mean_squared_error(y_test, y_pred_lin) ** 0.5,
        "model": lin_model,
        "y_pred": y_pred_lin,
        "coefficients": pd.Series(lin_model.coef_, index=X.columns).sort_values(),
    }

    if sample_for_rf and len(X_train) > sample_for_rf:
        X_train_rf = X_train.sample(sample_for_rf, random_state=42)
        y_train_rf = y_train.loc[X_train_rf.index]
    else:
        X_train_rf, y_train_rf = X_train, y_train

    rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf_model.predict(X_test)
    results["Random Forest"] = {
        "r2": r2_score(y_test, y_pred_rf),
        "mae": mean_absolute_error(y_test, y_pred_rf),
        "rmse": mean_squared_error(y_test, y_pred_rf) ** 0.5,
        "model": rf_model,
        "y_pred": y_pred_rf,
        "importances": pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(),
    }

    for name, res in results.items():
        print(f"\n=== {name} ===")
        print(f"R2:   {res['r2']:.4f}")
        print(f"MAE:  {res['mae']:.4f}")
        print(f"RMSE: {res['rmse']:.4f}")

    plt.figure(figsize=(6, 6))
    sample_idx = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    plt.scatter(y_test.values[sample_idx], y_pred_rf[sample_idx], alpha=0.25, s=10, color="#2E86AB")
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "k--", label="Perfect prediction")
    plt.xlabel("Actual Flood Probability")
    plt.ylabel("Predicted Flood Probability")
    plt.title("Predicted vs Actual (Random Forest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_predicted_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 7))
    results["Random Forest"]["importances"].plot(kind="barh", color="#3A7CA5")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    r2_vals = [results[n]["r2"] for n in names]
    plt.bar(names, r2_vals, color=["#E8A33D", "#4FA88B"])
    plt.ylabel("R\u00b2 Score")
    plt.title("Model Comparison (R\u00b2)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/5_model_comparison.png", dpi=150)
    plt.close()

    return results
