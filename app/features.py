import pandas as pd
import numpy as np
from datetime import datetime, timezone


# Features the model expects — must match main.py's Pydantic model
FEATURE_COLUMNS = [
    "pr_success_rate",
    "social_engagement_ratio",
    "recency_gap",
    "recency_gap_push",
    "recency_gap_pr",
    "recency_gap_issue",
    "recency_gap_comment",
    "max_inactivity_streak",
    "recent_velocity",
    "commit_velocity_1y",
    "velocity_drop",
    "repo_density",
    "is_org_member",
    "has_secure_profile",
]

# A user is considered churned if they have had zero public events
# of any type in the last ~30 days. Because the GitHub public events API
# only guarantees ~30 days of history, we align the threshold to the data.
CHURN_THRESHOLD_DAYS = 30


# ──────────────────────────────────────────────
# Core pipeline — call this from notebook & model.py
# ──────────────────────────────────────────────

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw GitHub API fields into model-ready features.
    Input:  raw DataFrame from scraper.py
    Output: DataFrame with FEATURE_COLUMNS + 'churned' label

    Expected raw columns from scraper.py:
        created_at, last_activity_at,
        last_push_at, last_pr_at, last_issue_at, last_comment_at,
        prs_merged, prs_opened,
        issue_comments, pr_comments, total_commits,
        events_30d, distinct_repos, activity_dates,
        org_count, ssh_key_count, gpg_key_count
    """
    df = df.copy()
    now = datetime.now(timezone.utc)

    # ── parse timestamps ──────────────────────
    df["last_activity"] = pd.to_datetime(df["last_activity_at"], utc=True)
    df["created"]       = pd.to_datetime(df["created_at"],       utc=True)

    # Parse per-type last-activity timestamps (None → NaT)
    df["last_push"]    = pd.to_datetime(df.get("last_push_at"),    utc=True, errors="coerce")
    df["last_pr"]      = pd.to_datetime(df.get("last_pr_at"),      utc=True, errors="coerce")
    df["last_issue"]   = pd.to_datetime(df.get("last_issue_at"),   utc=True, errors="coerce")
    df["last_comment"] = pd.to_datetime(df.get("last_comment_at"), utc=True, errors="coerce")

    # ── account age ───────────────────────────
    df["account_age_days"]  = (now - df["created"]).dt.days
    df["account_age_years"] = df["account_age_days"] / 365.0

    # ── 1. ratio features ─────────────────────

    # Low acceptance rate → repeated rejections → higher abandonment risk
    df["pr_success_rate"] = df["prs_merged"] / (df["prs_opened"] + 1)

    # High ratio → community member; low ratio → isolated "solo coder"
    total_social = df["issue_comments"] + df["pr_comments"]
    df["social_engagement_ratio"] = total_social / (
        df["total_commits"] + total_social + 1
    )

    # ── 2. time-based features ────────────────

    # Primary distance from the observation end
    df["recency_gap"] = (now - df["last_activity"]).dt.days

    # Per-type recency gaps.
    # If a specific event type is missing from the ~30-day window, we know the
    # true gap is at least 30 days. For already-inactive users we fall back to
    # the overall gap so the model sees consistency.
    def _type_gap(series: pd.Series) -> pd.Series:
        gap = (now - series).dt.days
        return gap.fillna(60)

    df["recency_gap_push"]    = _type_gap(df["last_push"])
    df["recency_gap_pr"]      = _type_gap(df["last_pr"])
    df["recency_gap_issue"]   = _type_gap(df["last_issue"])
    df["recency_gap_comment"] = _type_gap(df["last_comment"])

    # Longest consecutive gap between any two activity events
    df["max_inactivity_streak"] = df["activity_dates"].apply(_compute_max_streak)



    # ── 3. aggregation features ───────────────

    # Sum of all events in the trailing ~30-day window
    df["recent_velocity"] = df["events_30d"]

    # Historical commit pace over the trailing 1-year GraphQL window.
    # This establishes a baseline "normal" activity level for the user.
    df["commit_velocity_1y"] = df["total_commits"] / 365.0

    # Deceleration: positive values mean recent activity is BELOW historical
    # baseline; large positive values are strong churn signals.
    # (events_30d / 30) approximates their recent all-event daily pace.
    df["velocity_drop"] = df["commit_velocity_1y"] - (df["events_30d"] / 30.0)


    # Distinct repos / account_age_years; diverse anchors reduce churn risk
    df["repo_density"] = df["distinct_repos"] / (df["account_age_years"] + 0.01)

    # ── 4. binary features ────────────────────

    # Org membership implies team/professional obligations → lower churn risk
    df["is_org_member"] = (df["org_count"] > 0).astype(int)

    # SSH/GPG keys proxy development maturity
    df["has_secure_profile"] = (
        (df["ssh_key_count"] + df["gpg_key_count"]) > 0
    ).astype(int)

    # ── churn label ───────────────────────────
    # A user is churned only if they were historically active but have gone
    # 30+ days without ANY public event. This prevents labeling dormant
    # accounts and short-term vacationers as churned.
    #
    # Historical engagement is proven by 1-year GraphQL aggregates or all-time
    # PR counts — signals that the user once treated GitHub as a workstation.
    had_prior_engagement = (
        (df["total_commits"] > 0) |
        (df["prs_opened"] > 0) |
        (df["issue_comments"] > 0) |
        (df["distinct_repos"] > 0)
    )

    # Also require the account to be older than the threshold so brand-new
    # signups with zero activity aren't misclassified.
    mature_account = df["account_age_days"] > CHURN_THRESHOLD_DAYS

    df["churned"] = (
        (df["recency_gap"] > CHURN_THRESHOLD_DAYS) &
        had_prior_engagement &
        mature_account
    ).astype(int)

    return df


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _compute_max_streak(activity_dates) -> int:
    """
    Given a list of ISO datetime strings representing individual activity
    events, returns the longest consecutive gap (in days) between any two
    adjacent events, i.e. max(t_i - t_{i-1}).

    Returns 0 if fewer than 2 events are present (no gap can be computed).
    """
    if not isinstance(activity_dates, (list, np.ndarray)) or len(activity_dates) < 2:
        return 0

    dates = sorted(pd.to_datetime(activity_dates, utc=True))
    gaps  = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    return int(max(gaps)) if gaps else 0


# ──────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────

def get_X_y(df: pd.DataFrame):
    """
    Returns feature matrix X and target vector y from a raw DataFrame.
    Calls generate_features() internally — do not pre-transform the input.
    Rows where any feature could not be computed are dropped silently.
    """
    featured = generate_features(df)
    featured = featured.dropna(subset=FEATURE_COLUMNS)

    X = featured[FEATURE_COLUMNS]
    y = featured["churned"]

    return X, y


def check_class_balance(y: pd.Series) -> None:
    """
    Prints class balance. If the churned rate is below 10% or above 90%,
    warns you to adjust CHURN_THRESHOLD_DAYS or use class_weight='balanced'.
    """
    counts     = y.value_counts()
    churn_rate = counts.get(1, 0) / len(y)

    print("[features] Class balance:")
    print(f"  Retained (0): {counts.get(0, 0)} ({1 - churn_rate:.1%})")
    print(f"  Churned  (1): {counts.get(1, 0)} ({churn_rate:.1%})")

    if churn_rate < 0.10:
        print("[features] WARNING: Less than 10% churned — lower CHURN_THRESHOLD_DAYS.")
    elif churn_rate > 0.90:
        print("[features] WARNING: More than 90% churned — raise CHURN_THRESHOLD_DAYS.")
    else:
        print("[features] Class balance looks healthy.")


def features_from_dict(data: dict) -> np.ndarray:
    """
    Converts a single user's raw field dictionary into a feature array
    ready for model.predict(). Used by the FastAPI /predict endpoint.

    Expected keys:
        created_at, last_activity_at,
        last_push_at, last_pr_at, last_issue_at, last_comment_at,
        prs_merged, prs_opened,
        issue_comments, pr_comments, total_commits,
        events_30d, distinct_repos, activity_dates,
        org_count, ssh_key_count, gpg_key_count
    """
    now = datetime.now(timezone.utc)

    last_activity = pd.to_datetime(data["last_activity_at"], utc=True)
    created       = pd.to_datetime(data["created_at"],       utc=True)

    # Parse per-type timestamps (None / missing → NaT)
    last_push    = pd.to_datetime(data.get("last_push_at"),    utc=True, errors="coerce")
    last_pr      = pd.to_datetime(data.get("last_pr_at"),      utc=True, errors="coerce")
    last_issue   = pd.to_datetime(data.get("last_issue_at"),   utc=True, errors="coerce")
    last_comment = pd.to_datetime(data.get("last_comment_at"), utc=True, errors="coerce")

    recency_gap       = (now - last_activity).days
    account_age_days  = (now - created).days
    account_age_years = account_age_days / 365.0

    def _gap(ts):
        if pd.isna(ts):
            return max(recency_gap, 30)
        return (now - ts).days

    prs_merged     = data.get("prs_merged",     0)
    prs_opened     = data.get("prs_opened",     0)
    issue_comments = data.get("issue_comments", 0)
    pr_comments    = data.get("pr_comments",    0)
    total_commits  = data.get("total_commits",  0)
    events_30d     = data.get("events_30d",     0)
    distinct_repos = data.get("distinct_repos", 0)
    activity_dates = data.get("activity_dates", [])
    org_count      = data.get("org_count",      0)
    ssh_key_count  = data.get("ssh_key_count",  0)
    gpg_key_count  = data.get("gpg_key_count",  0)

    total_social = issue_comments + pr_comments

    commits_per_day_1y = total_commits / 365.0
    events_per_day_30d = events_30d / 30.0

    feature_vector = [
        # 1. ratio features
        prs_merged / (prs_opened + 1),                              # pr_success_rate
        total_social / (total_commits + total_social + 1),          # social_engagement_ratio
        # 2. time-based features
        recency_gap,                                                 # recency_gap
        _gap(last_push),                                             # recency_gap_push
        _gap(last_pr),                                               # recency_gap_pr
        _gap(last_issue),                                            # recency_gap_issue
        _gap(last_comment),                                          # recency_gap_comment
        _compute_max_streak(activity_dates),                         # max_inactivity_streak
        # 3. aggregation features
        events_30d,                                                  # recent_velocity
        commits_per_day_1y,                                         # commit_velocity_1y
        commits_per_day_1y - events_per_day_30d,                   # velocity_drop
        distinct_repos / (account_age_years + 0.01),                # repo_density
        # 4. binary features
        int(org_count > 0),                                         # is_org_member
        int((ssh_key_count + gpg_key_count) > 0),                   # has_secure_profile
    ]

    return np.array(feature_vector).reshape(1, -1)