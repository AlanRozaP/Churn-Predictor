from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, List
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
    description="Predicts GitHub user churn using 14 behavioral features engineered from GitHub REST + GraphQL APIs.",
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Request schema — raw GitHub API fields
# (features.py computes all 14 model features internally)
# ──────────────────────────────────────────────

class UserInput(BaseModel):
    """
    Raw GitHub profile fields collected by scraper.py.
    The API computes all 14 engineered features internally via features.py —
    callers never need to pre-compute anything.

    Required fields:
        created_at        — ISO timestamp of account creation
        last_activity_at  — ISO timestamp of most recent public event

    Optional timestamp fields (null → treated as inactive for 60+ days):
        last_push_at, last_pr_at, last_issue_at, last_comment_at

    Count fields default to 0 when omitted.
    """
    # Required timestamps
    created_at:       str  = Field(..., example="2018-06-01T08:00:00Z",
                                   description="ISO timestamp of account creation (created_at field)")
    last_activity_at: str  = Field(..., example="2022-03-15T10:00:00Z",
                                   description="ISO timestamp of most recent public event (updated_at fallback)")

    # Per-type last-event timestamps — None means that event type hasn't occurred in the 30d window
    last_push_at:    Optional[str] = Field(None, example="2022-03-10T08:00:00Z",
                                           description="ISO timestamp of last PushEvent (null if none in 30d window)")
    last_pr_at:      Optional[str] = Field(None, example=None,
                                           description="ISO timestamp of last PullRequestEvent (null if none)")
    last_issue_at:   Optional[str] = Field(None, example=None,
                                           description="ISO timestamp of last IssuesEvent (null if none)")
    last_comment_at: Optional[str] = Field(None, example=None,
                                           description="ISO timestamp of last IssueCommentEvent (null if none)")

    # PR success fields
    prs_merged:  int = Field(0, ge=0, example=3,  description="Total merged pull requests (Search API)")
    prs_opened:  int = Field(0, ge=0, example=5,  description="Total opened pull requests (Search API)")

    # Social engagement fields
    issue_comments: int = Field(0, ge=0, example=10, description="Total issue contributions (GraphQL)")
    pr_comments:    int = Field(0, ge=0, example=2,  description="Total PR review contributions (GraphQL)")

    # Commit / activity fields
    total_commits:  int = Field(0, ge=0, example=100,
                                description="Total commit contributions in trailing 1-year window (GraphQL)")
    events_30d:     int = Field(0, ge=0, example=0,
                                description="Count of all public events in the trailing ~30-day window (REST events API)")
    distinct_repos: int = Field(0, ge=0, example=4,
                                description="Number of distinct repos contributed to (GraphQL, union of types)")

    # Activity timeline for inactivity-streak calculation
    activity_dates: List[str] = Field(default_factory=list,
                                      example=[],
                                      description="ISO timestamps of individual events in the 30d window (for streak detection)")

    # Profile structural fields
    org_count:     int = Field(0, ge=0, example=1, description="Number of organisations the user belongs to")
    ssh_key_count: int = Field(0, ge=0, example=1, description="Number of SSH public keys on the account")
    gpg_key_count: int = Field(0, ge=0, example=0, description="Number of GPG keys on the account")


# ──────────────────────────────────────────────
# Response schema
# ──────────────────────────────────────────────

class PredictionResponse(BaseModel):
    churned:           bool
    churn_probability: float
    risk_level:        str
    features_used:     List[str]


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
        "features":             FEATURE_COLUMNS,
        "churn_threshold_days": CHURN_THRESHOLD_DAYS,
        "feature_count":        len(FEATURE_COLUMNS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(user: UserInput):
    """
    Accepts raw GitHub profile fields, computes the 14 behavioral features
    internally via features.py, and returns a churn prediction with probability.

    Risk levels:
        low    — churn_probability < 0.4
        medium — churn_probability 0.4–0.69
        high   — churn_probability >= 0.7

    Example curl (active user):
        curl -X POST http://localhost:8000/predict \\
             -H "Content-Type: application/json" \\
             -d '{
               "created_at":       "2018-06-01T08:00:00Z",
               "last_activity_at": "2024-05-20T10:00:00Z",
               "last_push_at":     "2024-05-18T08:00:00Z",
               "last_pr_at":       null,
               "last_issue_at":    null,
               "last_comment_at":  null,
               "prs_merged":       12,
               "prs_opened":       15,
               "issue_comments":   40,
               "pr_comments":      8,
               "total_commits":    300,
               "events_30d":       22,
               "distinct_repos":   7,
               "activity_dates":   ["2024-05-01T00:00:00Z","2024-05-10T00:00:00Z"],
               "org_count":        2,
               "ssh_key_count":    1,
               "gpg_key_count":    1
             }'

    Example curl (churned user):
        curl -X POST http://localhost:8000/predict \\
             -H "Content-Type: application/json" \\
             -d '{
               "created_at":       "2018-01-01T00:00:00Z",
               "last_activity_at": "2022-01-01T00:00:00Z",
               "last_push_at":     "2022-01-01T00:00:00Z",
               "last_pr_at":       null,
               "last_issue_at":    null,
               "last_comment_at":  null,
               "prs_merged":       3,
               "prs_opened":       5,
               "issue_comments":   10,
               "pr_comments":      2,
               "total_commits":    100,
               "events_30d":       0,
               "distinct_repos":   4,
               "activity_dates":   [],
               "org_count":        1,
               "ssh_key_count":    1,
               "gpg_key_count":    0
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
