from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import numpy as np

from model import load_model, predict
from features import FEATURE_COLUMNS, features_from_dict, CHURN_THRESHOLD_DAYS


# ──────────────────────────────────────────────
# Load model once at startup, not on every request
# ──────────────────────────────────────────────

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["rf"] = load_model()
    print("[app] Model loaded successfully.")
    yield
    ml_model.clear()

app = FastAPI(
    title="Customer Churn Predictor",
    description="Predicts GitHub user churn using behavioral features.",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Request schema — raw GitHub API fields
# ──────────────────────────────────────────────

class UserInput(BaseModel):
    """
    Raw GitHub profile fields. The API computes all 8 features internally
    so callers don't need to pre-compute anything.
    """
    updated_at:   str   = Field(..., example="2022-03-15T10:00:00Z",
                                description="ISO timestamp of last GitHub activity (updated_at field)")
    created_at:   str   = Field(..., example="2018-06-01T08:00:00Z",
                                description="ISO timestamp of account creation (created_at field)")
    followers:    int   = Field(0,   ge=0, example=12)
    following:    int   = Field(0,   ge=0, example=30)
    public_repos: int   = Field(0,   ge=0, example=5)
    public_gists: int   = Field(0,   ge=0, example=1)
    bio:          str | None = Field(None, example="Open source developer")


# ──────────────────────────────────────────────
# Response schema
# ──────────────────────────────────────────────

class PredictionResponse(BaseModel):
    churned:           bool
    churn_probability: float
    risk_level:        str
    features_used:     list[str]


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    """Standard health check. Used by Docker to verify the container is up."""
    return {"status": "ok"}


@app.get("/features")
def list_features():
    """Returns the feature names and churn threshold the model was trained on."""
    return {
        "features":              FEATURE_COLUMNS,
        "churn_threshold_days":  CHURN_THRESHOLD_DAYS,
        "feature_count":         len(FEATURE_COLUMNS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(user: UserInput):
    """
    Accepts raw GitHub user fields, computes the 8 behavioral features
    internally, and returns a churn prediction with probability.

    Example curl:
        curl -X POST http://localhost:8000/predict \\
             -H "Content-Type: application/json" \\
             -d '{
               "updated_at":   "2022-03-15T10:00:00Z",
               "created_at":   "2018-06-01T08:00:00Z",
               "followers":    12,
               "following":    30,
               "public_repos": 5,
               "public_gists": 1,
               "bio":          "Open source developer"
             }'
    """
    try:
        feature_array = features_from_dict(user.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature generation failed: {str(e)}"
        )

    try:
        result = predict(ml_model["rf"], feature_array)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    prob = result["churn_probability"]
    risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

    return PredictionResponse(
        churned=result["churned"],
        churn_probability=prob,
        risk_level=risk,
        features_used=FEATURE_COLUMNS,
    )