import pandas as pd
import numpy as np
from datetime import datetime, timezone


# Features the model expects — must match main.py's Pydantic model
FEATURE_COLUMNS = [
    "days_since_last_activity",
    "account_age_days",
    "follower_ratio",
    "inactive_ratio",
    "repos_per_year",
    "total_engagement_score",
    "has_no_repos",
    "has_bio",
]

CHURN_THRESHOLD_DAYS = 180  # users inactive beyond this are labelled churned


# ──────────────────────────────────────────────
# Core pipeline — call this from notebook & model.py
# ──────────────────────────────────────────────

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw GitHub API fields into model-ready features.
    Input:  raw DataFrame from scraper.py
    Output: DataFrame with FEATURE_COLUMNS + 'churned' label
    """
    df = df.copy()
    now = datetime.now(timezone.utc)

    # ── parse timestamps ──────────────────────
    df["last_active"] = pd.to_datetime(df["updated_at"], utc=True)
    df["created"]     = pd.to_datetime(df["created_at"], utc=True)

    # ── time-based features ───────────────────
    df["days_since_last_activity"] = (now - df["last_active"]).dt.days
    df["account_age_days"]         = (now - df["created"]).dt.days

    # ── ratio features ────────────────────────
    df["follower_ratio"] = df["followers"] / (df["following"] + 1)
    df["inactive_ratio"] = (
        df["days_since_last_activity"] / (df["account_age_days"] + 1)
    )

    # ── aggregation features ──────────────────
    df["repos_per_year"] = df["public_repos"] / (
        df["account_age_days"] / 365 + 0.01   # avoid division by zero
    )
    df["total_engagement_score"] = (
        df["followers"] + df["public_repos"] + df["public_gists"]
    )

    # ── binary features ───────────────────────
    df["has_no_repos"] = (df["public_repos"] == 0).astype(int)
    df["has_bio"]      = df["bio"].notna().astype(int)

    # ── churn label ───────────────────────────
    df["churned"] = (
        df["days_since_last_activity"] > CHURN_THRESHOLD_DAYS
    ).astype(int)

    return df


def get_X_y(df: pd.DataFrame):
    """
    Returns feature matrix X and target vector y from a featured DataFrame.
    Call generate_features() first.
    """
    featured = generate_features(df)

    # drop any rows where features couldn't be computed
    featured = featured.dropna(subset=FEATURE_COLUMNS)

    X = featured[FEATURE_COLUMNS]
    y = featured["churned"]

    return X, y


def check_class_balance(y: pd.Series) -> None:
    """
    Prints class balance. If churned rate is below 10% or above 90%,
    warns you to adjust CHURN_THRESHOLD_DAYS or use class_weight='balanced'.
    """
    counts = y.value_counts()
    churn_rate = counts.get(1, 0) / len(y)

    print(f"[features] Class balance:")
    print(f"  Retained (0): {counts.get(0, 0)} ({1 - churn_rate:.1%})")
    print(f"  Churned  (1): {counts.get(1, 0)} ({churn_rate:.1%})")

    if churn_rate < 0.10:
        print("[features] WARNING: Less than 10% churned. Lower CHURN_THRESHOLD_DAYS.")
    elif churn_rate > 0.90:
        print("[features] WARNING: More than 90% churned. Raise CHURN_THRESHOLD_DAYS.")
    else:
        print("[features] Class balance looks healthy.")


def features_from_dict(data: dict) -> np.ndarray:
    """
    Converts a single user's raw field dictionary into a feature array
    ready for model.predict(). Used by the FastAPI /predict endpoint.
    """
    now = datetime.now(timezone.utc)

    last_active  = pd.to_datetime(data["updated_at"], utc=True)
    created      = pd.to_datetime(data["created_at"], utc=True)

    days_since   = (now - last_active).days
    account_age  = (now - created).days
    followers    = data.get("followers", 0)
    following    = data.get("following", 0)
    public_repos = data.get("public_repos", 0)
    public_gists = data.get("public_gists", 0)
    bio          = data.get("bio")

    feature_vector = [
        days_since,
        account_age,
        followers / (following + 1),
        days_since / (account_age + 1),
        public_repos / (account_age / 365 + 0.01),
        followers + public_repos + public_gists,
        int(public_repos == 0),
        int(bio is not None and bio != ""),
    ]

    return np.array(feature_vector).reshape(1, -1)