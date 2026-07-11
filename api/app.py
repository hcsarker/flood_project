"""
app.py
-------
FastAPI backend for the Flood Prediction dashboard.

Serves the trained Linear Regression and Random Forest models over
a REST API so the dashboard.html UI can call a REAL model instead of
a hardcoded JS formula.

Run:
    pip install fastapi uvicorn joblib scikit-learn pandas
    python api/train_and_save_model.py     # do this once first
    uvicorn api.app:app --reload --port 8000

Then open dashboard.html in your browser — it will auto-detect the
running API at http://127.0.0.1:8000 and switch from "offline formula"
mode to "live model" mode.
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

app = FastAPI(title="Flood Prediction API", version="1.0")

# Allow the dashboard (opened as a local file, or from any origin during
# development) to call this API. For a real production deployment,
# replace "*" with your actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_ORDER = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds", "DeterioratingInfrastructure",
    "PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors",
]

_models = {"linear": None, "rf": None}


def load_models():
    lin_path = os.path.join(MODELS_DIR, "linear_regression.joblib")
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    if not (os.path.exists(lin_path) and os.path.exists(rf_path)):
        raise RuntimeError(
            "Model files not found. Run `python api/train_and_save_model.py` first."
        )
    _models["linear"] = joblib.load(lin_path)
    _models["rf"] = joblib.load(rf_path)


@app.on_event("startup")
def startup_event():
    load_models()


class FloodFeatures(BaseModel):
    MonsoonIntensity: float = Field(..., ge=0, le=20)
    TopographyDrainage: float = Field(..., ge=0, le=20)
    RiverManagement: float = Field(..., ge=0, le=20)
    Deforestation: float = Field(..., ge=0, le=20)
    Urbanization: float = Field(..., ge=0, le=20)
    ClimateChange: float = Field(..., ge=0, le=20)
    DamsQuality: float = Field(..., ge=0, le=20)
    Siltation: float = Field(..., ge=0, le=20)
    AgriculturalPractices: float = Field(..., ge=0, le=20)
    Encroachments: float = Field(..., ge=0, le=20)
    IneffectiveDisasterPreparedness: float = Field(..., ge=0, le=20)
    DrainageSystems: float = Field(..., ge=0, le=20)
    CoastalVulnerability: float = Field(..., ge=0, le=20)
    Landslides: float = Field(..., ge=0, le=20)
    Watersheds: float = Field(..., ge=0, le=20)
    DeterioratingInfrastructure: float = Field(..., ge=0, le=20)
    PopulationScore: float = Field(..., ge=0, le=20)
    WetlandLoss: float = Field(..., ge=0, le=20)
    InadequatePlanning: float = Field(..., ge=0, le=20)
    PoliticalFactors: float = Field(..., ge=0, le=20)


@app.get("/")
def root():
    return {"status": "ok", "message": "Flood Prediction API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _models["linear"] is not None}


@app.post("/predict")
def predict(features: FloodFeatures):
    if _models["linear"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    row = [[getattr(features, col) for col in FEATURE_ORDER]]

    lin_pred = float(_models["linear"].predict(row)[0])
    rf_pred = float(_models["rf"].predict(row)[0])

    def classify(p):
        if p < 0.40:
            return "Low"
        elif p < 0.55:
            return "Moderate"
        else:
            return "High"

    return {
        "linear_regression": {
            "flood_probability": round(lin_pred, 5),
            "risk_level": classify(lin_pred),
        },
        "random_forest": {
            "flood_probability": round(rf_pred, 5),
            "risk_level": classify(rf_pred),
        },
    }
