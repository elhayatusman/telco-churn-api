from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Load artefacts created during model training
BASE_DIR       = Path(__file__).parent
ARTIFACTS_DIR  = BASE_DIR / "model_artifacts"

log.info("Loading model artefacts from %s", ARTIFACTS_DIR)

model            = joblib.load(ARTIFACTS_DIR / "churn_model.pkl")
scaler           = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
feature_columns  = [str(c) for c in joblib.load(ARTIFACTS_DIR / "feature_columns.pkl")]
threshold        = float(joblib.load(ARTIFACTS_DIR / "optimal_threshold.pkl"))
model_name       = str(type(model).__name__)

log.info("Model loaded: %s  |  Features: %d  |  Threshold: %.4f",
         model_name, len(feature_columns), threshold)

# FastAPI application setup
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Predicts the probability that a telecom customer will churn "
        "based on their account profile and service subscriptions. "
        "Built with FastAPI and a trained Random Forest classifier."
    ),
    version="1.0.0",
    contact={
        "name": "Abdulrahman Hayatu Usman",
        "url":  "https://github.com/AbdulrahmanHayatuUsman",
    },
    license_info={"name": "MIT"},
)

# Allow all origins for portfolio / demo use (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Pydantic input schema with validation

class CustomerInput(BaseModel):
    """
    All fields a customer record can contain.
    Mirrors the original Telco dataset columns (post-preprocessing).
    Categorical fields use Literal types — invalid values are rejected
    with a clear validation error before they reach the model.
    """

    # Demographics 
    gender:         Literal["Male", "Female"] = Field(..., example="Male")
    SeniorCitizen:  Literal[0, 1]             = Field(..., example=0,
                        description="1 = senior citizen (65+), 0 = not")
    Partner:        Literal["Yes", "No"]      = Field(..., example="Yes")
    Dependents:     Literal["Yes", "No"]      = Field(..., example="No")

    # Account info
    tenure:           int   = Field(..., ge=0, le=120, example=12,
                                    description="Months as a customer (0–120)")
    Contract:         Literal["Month-to-month", "One year", "Two year"] = Field(
                          ..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod:    Literal[
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)",
                      ] = Field(..., example="Electronic check")
    MonthlyCharges:   float = Field(..., gt=0, le=200, example=70.35,
                                    description="Monthly bill in USD")
    TotalCharges:     float = Field(..., ge=0,          example=842.20,
                                    description="Total billed to date in USD")

    # Phone services
    PhoneService:  Literal["Yes", "No"]                        = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"]    = Field(..., example="No")

    # Internet services 
    InternetService:  Literal["DSL", "Fiber optic", "No"] = Field(..., example="Fiber optic")
    OnlineSecurity:   Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup:     Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport:      Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV:      Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingMovies:  Literal["Yes", "No", "No internet service"] = Field(..., example="No")

    # Cross-field validation
    @model_validator(mode="after")
    def check_total_charges_consistency(self) -> "CustomerInput":
        """
        TotalCharges should be approximately >= MonthlyCharges when tenure > 0.
        We allow a tolerance for plan changes over time.
        """
        if self.tenure > 0 and self.TotalCharges < self.MonthlyCharges * 0.5:
            raise ValueError(
                f"TotalCharges ({self.TotalCharges}) seems too low for "
                f"tenure={self.tenure} months and MonthlyCharges={self.MonthlyCharges}. "
                "Please verify the values."
            )
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(...,
        description="Model's estimated probability that this customer will churn (0–1)")
    churn_prediction:  bool  = Field(...,
        description="True = predicted to churn (probability ≥ threshold)")
    risk_label:        str   = Field(...,
        description="Human-readable risk tier: Low / Medium / High / Critical")
    threshold_used:    float = Field(...,
        description="Classification threshold applied to produce the prediction")
    model_name:        str   = Field(...,
        description="Name of the underlying ML model")
    processing_time_ms: float = Field(...,
        description="Server-side processing time in milliseconds")


class BatchInput(BaseModel):
    customers: list[CustomerInput] = Field(
        ..., min_length=1, max_length=500,
        description="List of 1–500 customer records"
    )


class BatchPredictionResponse(BaseModel):
    predictions:  list[PredictionResponse]
    total_records: int
    predicted_churners: int
    churn_rate_pct: float
    processing_time_ms: float


# Preprocessing helper function

def preprocess(customer: CustomerInput) -> pd.DataFrame:
    """
    Applies the same transformations used during model training:
      1. Binary encoding (Yes/No → 1/0)
      2. Recode 'No internet/phone service' → 0
      3. One-hot encoding for multi-class categoricals
      4. Feature engineering (AvgMonthlySpend, ServiceCount)
      5. Column alignment to training feature set
    The scaler is intentionally NOT applied here — Random Forest is
    scale-invariant. If you swap to Logistic Regression, add scaling.
    """
    d = customer.model_dump()

    # 1. Binary mappings
    yes_no = {"Yes": 1, "No": 0}
    for col in [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]:
        d[col] = 1 if d[col] == "Yes" else 0

    d["gender"]       = 1 if d["gender"] == "Male" else 0
    d["MultipleLines"] = 1 if d["MultipleLines"] == "Yes" else 0

    # 2. One-hot: InternetService
    d["InternetService_Fiber optic"] = int(d["InternetService"] == "Fiber optic")
    d["InternetService_No"]          = int(d["InternetService"] == "No")
    # DSL is the dropped reference category

    # 3. One-hot: Contract
    d["Contract_One year"] = int(d["Contract"] == "One year")
    d["Contract_Two year"] = int(d["Contract"] == "Two year")
    # Month-to-month is the dropped reference category

    # 4. One-hot: PaymentMethod
    d["PaymentMethod_Credit card (automatic)"] = int(
        d["PaymentMethod"] == "Credit card (automatic)")
    d["PaymentMethod_Electronic check"]        = int(
        d["PaymentMethod"] == "Electronic check")
    d["PaymentMethod_Mailed check"]            = int(
        d["PaymentMethod"] == "Mailed check")
    # Bank transfer is the dropped reference category

    # 5. Feature engineering — must match training notebook exactly
    tenure = d["tenure"]
    d["AvgMonthlySpend"] = (
        d["MonthlyCharges"] if tenure == 0
        else d["TotalCharges"] / tenure
    )
    d["ServiceCount"] = sum([
        d["OnlineSecurity"], d["OnlineBackup"], d["DeviceProtection"],
        d["TechSupport"], d["StreamingTV"], d["StreamingMovies"],
    ])

    # 6. Drop raw categorical columns that were one-hot encoded
    for col in ["InternetService", "Contract", "PaymentMethod"]:
        d.pop(col, None)

    # 7. Build DataFrame with exact training column order
    df = pd.DataFrame([d])[feature_columns]
    return df


def risk_label(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.50:
        return "Medium"
    if prob < 0.70:
        return "High"
    return "Critical"


# Routes

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is live and returns model metadata."""
    return {
        "status":      "online",
        "api":         "Telco Customer Churn Prediction API",
        "version":     "1.0.0",
        "model":       model_name,
        "features":    len(feature_columns),
        "threshold":   round(threshold, 4),
        "author":      "Abdulrahman Hayatu Usman",
        "docs":        "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """
    Predict churn probability for a **single customer**.

    Returns the churn probability, a binary prediction at the optimised
    threshold, and a human-readable risk tier (Low / Medium / High / Critical).
    """
    t0 = time.perf_counter()

    try:
        X = preprocess(customer)
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        log.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    elapsed_ms = float((time.perf_counter() - t0) * 1000)
    log.info("Single prediction — prob=%.4f  label=%s  time=%.1fms",
             prob, risk_label(prob), elapsed_ms)

    return PredictionResponse(
        churn_probability   = float(round(prob, 4)),
        churn_prediction    = bool(prob >= threshold),
        risk_label          = risk_label(prob),
        threshold_used      = float(round(threshold, 4)),
        model_name          = model_name,
        processing_time_ms  = float(round(elapsed_ms, 2)),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict churn for a **batch of up to 500 customers** in one request.

    Also returns aggregate statistics: total predicted churners and
    the predicted churn rate across the submitted batch.
    """
    t0 = time.perf_counter()

    try:
        frames = [preprocess(c) for c in batch.customers]
        X_batch = pd.concat(frames, ignore_index=True)
        probs   = model.predict_proba(X_batch)[:, 1]
    except Exception as exc:
        log.exception("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")

    total_ms = float((time.perf_counter() - t0) * 1000)
    per_record_ms = float(total_ms / len(probs))

    predictions = [
        PredictionResponse(
            churn_probability   = float(round(float(p), 4)),
            churn_prediction    = bool(float(p) >= threshold),
            risk_label          = risk_label(float(p)),
            threshold_used      = float(round(threshold, 4)),
            model_name          = model_name,
            processing_time_ms  = float(round(per_record_ms, 2)),
        )
        for p in probs
    ]

    churners = int(sum(1 for p in predictions if p.churn_prediction))
    log.info("Batch prediction — %d records  churners=%d  time=%.1fms",
             len(probs), churners, total_ms)

    return BatchPredictionResponse(
        predictions         = predictions,
        total_records        = int(len(predictions)),
        predicted_churners   = churners,
        churn_rate_pct       = float(round(churners / len(predictions) * 100, 2)),
        processing_time_ms   = float(round(total_ms, 2)),
    )


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the loaded model and its feature set."""
    return {
        "model_type":          model_name,
        "n_features":          len(feature_columns),
        "feature_names":       feature_columns,
        "classification_threshold": round(threshold, 4),
        "threshold_rationale": (
            "Threshold was optimised on the training set to maximise F1-score "
            "for the churner class, balancing precision and recall."
        ),
        "risk_tiers": {
            "Low":      "prob < 0.30",
            "Medium":   "0.30 ≤ prob < 0.50",
            "High":     "0.50 ≤ prob < 0.70",
            "Critical": "prob ≥ 0.70",
        },
    }
