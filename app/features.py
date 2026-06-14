import pandas as pd
import numpy as np
from datetime import datetime, timezone


# Features the model expects — must match main.py's Pydantic model
FEATURE_COLUMNS = [
    "pr_success_rate",
    "social_engagement_ratio",
    "recency_gap",
    "max_inactivity_streak",
    "recent_velocity",
    "repo_density",
    "is_org_member",
    "has_secure_profile",
]

# A user is considered churned if they have had zero contributions
# of any type (pushes, PRs, comments, issues) for this many days.
CHURN_THRESHOLD_DAYS = 90


# ──────────────────────────────────────────────
# Core pipeline — call this from notebook & model.py
# ──────────────────────────────────────────────

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw GitHub API fields into model-ready features.
    Input:  raw DataFrame from scraper.py
    Output: DataFrame with FEATURE_COLUMNS + 'churned' label

    Expected raw columns from scraper.py:
        created_at          str  — ISO timestamp of account creation
                                   Source: GET /users/{username} → created_at

        last_activity_at    str  — ISO timestamp of the user's most recent
                                   contribution of ANY type (push, PR, comment,
                                   issue, review). Should be the max timestamp
                                   across all event sources.
                                   Source: GET /users/{username}/events/public
                                           → max(created_at)

        prs_merged          int  — Total merged pull requests authored
                                   Source: GET /search/issues
                                           ?q=author:{u}+type:pr+is:merged

        prs_opened          int  — Total pull requests opened (all states)
                                   Source: GET /search/issues
                                           ?q=author:{u}+type:pr

        issue_comments      int  — Total issue comments authored
                                   Source: GET /search/issues
                                           ?q=commenter:{u}+type:issue

        pr_comments         int  — Total PR review / inline comments authored
                                   Source: GET /repos/.../pulls/comments
                                           filtered by user, aggregated

        total_commits       int  — Total commits across all repos (public)
                                   Source: GraphQL contributionsCollection
                                           → totalCommitContributions

        events_90d          int  — Count of ALL public events in the trailing
                                   90 days of the observation window
                                   Source: GET /users/{username}/events/public
                                           filtered to last 90 days

        distinct_repos      int  — Count of unique repos the user has
                                   contributed to (commits, PRs, issues, reviews)
                                   Source: GraphQL contributionsCollection
                                           → commitContributionsByRepository
                                           (count distinct repoNameWithOwner)

        activity_dates      list[str] — ISO date strings of every individual
                                        activity event within the observation
                                        window. Used to compute the maximum
                                        inactivity streak.
                                        Source: GET /users/{username}/events/public
                                                → [e["created_at"] for e in events]

        org_count           int  — Number of GitHub organizations the user
                                   belongs to
                                   Source: GET /users/{username}/orgs → len(...)

        ssh_key_count       int  — Number of public SSH keys on the account
                                   Source: GET /users/{username}/keys → len(...)

        gpg_key_count       int  — Number of GPG keys on the account
                                   Source: GET /user/gpg_keys → len(...)
                                   (requires authenticated scope)
    """
    df = df.copy()
    now = datetime.now(timezone.utc)

    # ── parse timestamps ──────────────────────
    df["last_activity"] = pd.to_datetime(df["last_activity_at"], utc=True)
    df["created"]       = pd.to_datetime(df["created_at"],       utc=True)

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

    # Primary distance from the observation end; Δt_recency in the spec
    df["recency_gap"] = (now - df["last_activity"]).dt.days

    # Longest consecutive gap between any two activity events;
    # users near 30–45 days are structurally at risk of breaching 90 days
    df["max_inactivity_streak"] = df["activity_dates"].apply(_compute_max_streak)

    # ── 3. aggregation features ───────────────

    # Sum of all events in the trailing 90-day window;
    # a steep decline signals a systematic wind-down
    df["recent_velocity"] = df["events_90d"]

    # Distinct repos / account_age_years; diverse anchors reduce churn risk
    df["repo_density"] = df["distinct_repos"] / (df["account_age_years"] + 0.01)

    # ── 4. binary features ────────────────────

    # Org membership implies team/professional obligations → lower churn risk
    df["is_org_member"] = (df["org_count"] > 0).astype(int)

    # SSH/GPG keys proxy development maturity; treats platform as core workstation
    df["has_secure_profile"] = (
        (df["ssh_key_count"] + df["gpg_key_count"]) > 0
    ).astype(int)

    # ── churn label ───────────────────────────
    # A user is churned if they have gone 90+ days without ANY contribution:
    # pushes, PRs, issue/PR comments, reviews — anything surfaced in the
    # public events feed or contribution graph.
    df["churned"] = (df["recency_gap"] > CHURN_THRESHOLD_DAYS).astype(int)

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

    Expected keys — see generate_features() docstring for field descriptions
    and their GitHub API sources:
        created_at, last_activity_at,
        prs_merged, prs_opened,
        issue_comments, pr_comments, total_commits,
        events_90d, distinct_repos, activity_dates,
        org_count, ssh_key_count, gpg_key_count
    """
    now = datetime.now(timezone.utc)

    last_activity     = pd.to_datetime(data["last_activity_at"], utc=True)
    created           = pd.to_datetime(data["created_at"],       utc=True)

    recency_gap       = (now - last_activity).days
    account_age_days  = (now - created).days
    account_age_years = account_age_days / 365.0

    prs_merged     = data.get("prs_merged",     0)
    prs_opened     = data.get("prs_opened",     0)
    issue_comments = data.get("issue_comments", 0)
    pr_comments    = data.get("pr_comments",    0)
    total_commits  = data.get("total_commits",  0)
    events_90d     = data.get("events_90d",     0)
    distinct_repos = data.get("distinct_repos", 0)
    activity_dates = data.get("activity_dates", [])
    org_count      = data.get("org_count",      0)
    ssh_key_count  = data.get("ssh_key_count",  0)
    gpg_key_count  = data.get("gpg_key_count",  0)

    total_social = issue_comments + pr_comments

    feature_vector = [
        # 1. ratio features
        prs_merged / (prs_opened + 1),                              # pr_success_rate
        total_social / (total_commits + total_social + 1),          # social_engagement_ratio
        # 2. time-based features
        recency_gap,                                                 # recency_gap
        _compute_max_streak(activity_dates),                         # max_inactivity_streak
        # 3. aggregation features
        events_90d,                                                  # recent_velocity
        distinct_repos / (account_age_years + 0.01),                # repo_density
        # 4. binary features
        int(org_count > 0),                                         # is_org_member
        int((ssh_key_count + gpg_key_count) > 0),                   # has_secure_profile
    ]

    return np.array(feature_vector).reshape(1, -1)