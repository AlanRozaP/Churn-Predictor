import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL     = "https://api.github.com"
GRAPHQL_URL  = "https://api.github.com/graphql"

_headers = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    _headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
else:
    print("[WARNING] No GITHUB_TOKEN found — rate-limited to 60 req/hr and GraphQL unavailable.")
    print("[WARNING] Add GITHUB_TOKEN to .env for 5,000 req/hr and full feature coverage.")


# ──────────────────────────────────────────────
# Low-level HTTP helpers
# ──────────────────────────────────────────────

def _handle_rate_limit(r: requests.Response, label: str = "") -> None:
    """
    Checks response headers and sleeps proactively when the remaining
    request budget drops below 5. Avoids hitting hard 403 blocks mid-run.
    """
    remaining = int(r.headers.get("X-RateLimit-Remaining", 999))
    if remaining < 5:
        reset_ts = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
        sleep_s  = max(reset_ts - int(time.time()), 1) + 2
        print(f"[scraper] Rate limit nearly exhausted ({label}). Sleeping {sleep_s}s...")
        time.sleep(sleep_s)


def _get(url: str, params: dict = None, label: str = "") -> requests.Response | None:
    """
    GET with 3 retries. Handles:
        404 → returns None silently (user/resource not found)
        403 → rate limit: sleeps until reset, then retries
        429 → search rate limit: respects Retry-After header
        5xx → exponential backoff, up to 3 attempts
    """
    for attempt in range(3):
        try:
            r = requests.get(url, headers=_headers, params=params, timeout=15)
        except requests.RequestException as e:
            print(f"[scraper] Network error ({label}): {e}")
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code == 404:
            return None

        if r.status_code in (403, 429):
            reset_ts    = int(r.headers.get("X-RateLimit-Reset",  time.time() + 60))
            retry_after = int(r.headers.get("Retry-After", 0))
            sleep_s     = max(reset_ts - int(time.time()), retry_after, 1) + 2
            print(f"[scraper] Rate limited ({label}). Sleeping {sleep_s}s...")
            time.sleep(sleep_s)
            continue

        if r.status_code == 422:
            # Search API: query unprocessable (malformed or unsupported)
            print(f"[scraper] 422 Unprocessable ({label}) — returning None")
            return None

        if r.ok:
            _handle_rate_limit(r, label)
            return r

        print(f"[scraper] HTTP {r.status_code} ({label}) — skipping")
        return None

    print(f"[scraper] All 3 retries exhausted ({label}).")
    return None


def _graphql(query: str, variables: dict, label: str = "") -> dict | None:
    """
    POST to the GraphQL endpoint. Returns the parsed `data` dict or None.
    Requires a valid GITHUB_TOKEN — returns None silently if no token is set.
    """
    if not GITHUB_TOKEN:
        return None

    try:
        r = requests.post(
            GRAPHQL_URL,
            headers=_headers,
            json={"query": query, "variables": variables},
            timeout=20,
        )
    except requests.RequestException as e:
        print(f"[scraper] GraphQL network error ({label}): {e}")
        return None

    if r.status_code in (401, 403):
        print(f"[scraper] GraphQL auth error ({label}) — check GITHUB_TOKEN")
        return None

    if not r.ok:
        print(f"[scraper] GraphQL HTTP {r.status_code} ({label})")
        return None

    body = r.json()
    if "errors" in body:
        msgs = [e.get("message", "?") for e in body["errors"]]
        print(f"[scraper] GraphQL errors ({label}): {'; '.join(msgs)}")
        return None

    _handle_rate_limit(r, label)
    return body.get("data")


# ──────────────────────────────────────────────
# Per-field fetchers
# ──────────────────────────────────────────────

# GraphQL query for a 1-year rolling contribution window.
# All three *ContributionsByRepository lists are merged to produce
# the count of DISTINCT repos the user has contributed to.
_CONTRIBS_QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      totalCommitContributions
      totalIssueContributions
      totalPullRequestReviewContributions
      commitContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
      }
      pullRequestContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
      }
      issueContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner }
      }
    }
  }
}
"""


def _fetch_contributions(username: str, created_at: str) -> dict:
    """
    Fetches contribution statistics for the trailing 1-year window via GraphQL.

    Field mapping → features.py column:
        totalCommitContributions            → total_commits
        totalIssueContributions             → issue_comments   (proxy: issues created/engaged)
        totalPullRequestReviewContributions → pr_comments      (proxy: PR reviews submitted)
        union of *ContributionsByRepository → distinct_repos

    NOTE — proxy fields:
        GitHub's contributionsCollection does not expose raw comment counts.
        `issue_comments` uses issues-created/engaged as a proxy for issue-thread
        engagement. `pr_comments` uses review submissions as a proxy for PR
        inline/review comment activity. Both are directionally correct signals for
        the social_engagement_ratio feature.

    NOTE — prs_opened is intentionally excluded here. It is fetched separately
    from the Search API alongside prs_merged so both use the same all-time basis,
    making pr_success_rate a consistent ratio.
    """
    empty = {
        "total_commits":  0,
        "issue_comments": 0,
        "pr_comments":    0,
        "distinct_repos": 0,
    }

    now          = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    created      = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    from_date    = max(one_year_ago, created)  # clamp — can't query before account existed

    data = _graphql(
        _CONTRIBS_QUERY,
        {"login": username, "from": from_date.isoformat(), "to": now.isoformat()},
        label=f"graphql:{username}",
    )

    if not data or not data.get("user"):
        return empty

    col = data["user"]["contributionsCollection"]

    # Union of distinct repos touched via commits, PRs, and issues
    repo_set = set()
    for key in (
        "commitContributionsByRepository",
        "pullRequestContributionsByRepository",
        "issueContributionsByRepository",
    ):
        for entry in col.get(key, []):
            repo = entry.get("repository")
            if repo:
                repo_set.add(repo["nameWithOwner"])

    return {
        "total_commits":  col.get("totalCommitContributions", 0),
        "issue_comments": col.get("totalIssueContributions", 0),
        "pr_comments":    col.get("totalPullRequestReviewContributions", 0),
        "distinct_repos": len(repo_set),
    }


def _fetch_pr_counts(username: str) -> tuple[int, int]:
    """
    Returns (prs_opened, prs_merged) — all-time totals via the Search API.
    Uses per_page=1 since only total_count is needed; minimises data transfer.

    Both counts use the same all-time basis so pr_success_rate = merged / (opened + 1)
    is a consistent lifetime ratio rather than a mixed time-window one.

    The 2.1-second inter-call sleep respects the Search API rate limit (30 req/min).
    """
    r_open = _get(
        f"{BASE_URL}/search/issues",
        params={"q": f"author:{username} type:pr", "per_page": 1},
        label=f"prs_open:{username}",
    )
    prs_opened = r_open.json().get("total_count", 0) if r_open else 0
    time.sleep(2.1)

    r_merged = _get(
        f"{BASE_URL}/search/issues",
        params={"q": f"author:{username} type:pr is:merged", "per_page": 1},
        label=f"prs_merged:{username}",
    )
    prs_merged = r_merged.json().get("total_count", 0) if r_merged else 0
    time.sleep(2.1)

    return prs_opened, prs_merged


def _fetch_events(username: str) -> tuple[str | None, int, list[str]]:
    """
    Fetches up to 300 public events (GitHub's hard cap) for the user.
    GitHub guarantees all events are within the last 90 days, so:
        events_90d     = total event count returned
        activity_dates = list of all event ISO timestamps
        last_activity_at = max(activity_dates)

    Uses per_page=100 to minimise API calls (3 pages × 100 = 300 events).

    If no events are found (user has been inactive for > 90 days), returns
    (None, 0, []) — the caller falls back to profile-level updated_at.
    """
    all_dates: list[str] = []

    for page in range(1, 4):  # 3 pages × 100 = 300 events max
        r = _get(
            f"{BASE_URL}/users/{username}/events/public",
            params={"per_page": 100, "page": page},
            label=f"events:{username}",
        )
        if not r:
            break

        batch = r.json()
        if not batch:
            break

        all_dates.extend(e["created_at"] for e in batch if e.get("created_at"))

        if len(batch) < 100:  # last page reached
            break

        time.sleep(0.3)

    if not all_dates:
        return None, 0, []

    return max(all_dates), len(all_dates), all_dates


def _fetch_org_count(username: str) -> int:
    """
    Returns the number of public organisations the user belongs to.
    Source: GET /users/{username}/orgs
    """
    r = _get(
        f"{BASE_URL}/users/{username}/orgs",
        params={"per_page": 100},
        label=f"orgs:{username}",
    )
    return len(r.json()) if r else 0


def _fetch_key_counts(username: str) -> tuple[int, int]:
    """
    Returns (ssh_key_count, gpg_key_count).
    Both endpoints are publicly accessible — no extra OAuth scope needed.
    Sources:
        GET /users/{username}/keys      → SSH public keys
        GET /users/{username}/gpg_keys  → GPG public keys
    """
    ssh_r = _get(f"{BASE_URL}/users/{username}/keys",     label=f"ssh:{username}")
    gpg_r = _get(f"{BASE_URL}/users/{username}/gpg_keys", label=f"gpg:{username}")
    return (
        len(ssh_r.json()) if ssh_r else 0,
        len(gpg_r.json()) if gpg_r else 0,
    )


# ──────────────────────────────────────────────
# Step 1 — collect usernames via /users list
# ──────────────────────────────────────────────

def get_usernames(count: int = 300, since: int = 0) -> list[str]:
    """
    Walks GitHub's paginated /users endpoint to collect `count` usernames.
    `since` is the user ID to start after — the API uses it as a cursor,
    returning the next page of users whose ID is greater than `since`.
    """
    usernames: list[str] = []
    print(f"[scraper] Collecting {count} usernames starting from ID {since}...")

    while len(usernames) < count:
        r = _get(
            f"{BASE_URL}/users",
            params={"since": since, "per_page": min(count - len(usernames), 100)},
            label="usernames",
        )
        if not r:
            break

        batch = r.json()
        if not batch:
            print("[scraper] No more users returned by API.")
            break

        usernames += [u["login"] for u in batch]
        since = batch[-1]["id"]  # correct cursor: last returned user ID

        remaining = r.headers.get("X-RateLimit-Remaining", "?")
        print(f"[scraper] Collected {len(usernames)} usernames | rate limit remaining: {remaining}")

        time.sleep(0.7)

    return usernames[:count]


def get_usernames_spread(total: int = 900) -> list[str]:
    """
    Samples usernames from six points across GitHub's user ID space so the
    dataset spans multiple account-age eras — avoids the early-adopter bias
    that results from starting at ID 0 every time.
    """
    checkpoints = [
        0,            # 2008 — founders era
        1_000_000,    # ~2012
        5_000_000,    # ~2013
        20_000_000,   # ~2015
        50_000_000,   # ~2017
        100_000_000,  # ~2021
    ]

    per_checkpoint = total // len(checkpoints)
    usernames: list[str] = []

    for since in checkpoints:
        print(f"[scraper] Sampling {per_checkpoint} users from ID {since:,}...")
        usernames += get_usernames(count=per_checkpoint, since=since)
        time.sleep(1)

    return usernames[:total]


# ──────────────────────────────────────────────
# Step 2 — fetch full profile for each username
# ──────────────────────────────────────────────

def fetch_user_profile(username: str) -> dict | None:
    """
    Fetches all raw fields required by features.py for a single user.
    Makes up to 8 API calls per user:

        Call  Endpoint                              Fields produced
        ────  ──────────────────────────────────    ─────────────────────────────────────
        1     GET /users/{username}                 created_at
        2     POST /graphql (1-year window)         total_commits, issue_comments,
                                                    pr_comments, distinct_repos
        3     GET /search/issues (×2)               prs_opened, prs_merged  (all-time)
        4     GET /users/{u}/events/public (×1-3)   last_activity_at, events_90d,
                                                    activity_dates
        5     GET /users/{u}/orgs                   org_count
        6     GET /users/{u}/keys                   ssh_key_count
        7     GET /users/{u}/gpg_keys               gpg_key_count

    Returns None if the base profile fetch fails or created_at is missing.
    """
    # ── 1. base profile ───────────────────────────────────────────────────
    r = _get(f"{BASE_URL}/users/{username}", label=f"profile:{username}")
    if not r:
        return None

    profile    = r.json()
    created_at = profile.get("created_at")
    if not created_at:
        print(f"[scraper] Missing created_at for {username} — skipping")
        return None

    # ── 2. graphql contributions (1-year rolling window) ─────────────────
    contribs = _fetch_contributions(username, created_at)
    time.sleep(0.2)

    # ── 3. PR counts via Search API (all-time, consistent window) ─────────
    prs_opened, prs_merged = _fetch_pr_counts(username)
    # _fetch_pr_counts already sleeps 2.1s between its two calls

    # ── 4. events (last 90 days, max 300) ─────────────────────────────────
    last_activity_at, events_90d, activity_dates = _fetch_events(username)
    if not last_activity_at:
        # No public events in 90 days — user is almost certainly churned.
        # Fall back to profile-level updated_at as the best available signal.
        last_activity_at = profile.get("updated_at")
    time.sleep(0.2)

    # ── 5. org membership ─────────────────────────────────────────────────
    org_count = _fetch_org_count(username)
    time.sleep(0.2)

    # ── 6 & 7. SSH and GPG key counts ─────────────────────────────────────
    ssh_key_count, gpg_key_count = _fetch_key_counts(username)

    return {
        "username":         username,
        # ── timestamps ───────────────────────────────────────────────────
        "created_at":       created_at,        # → account_age_days / account_age_years
        "last_activity_at": last_activity_at,  # → recency_gap
        # ── PR features ──────────────────────────────────────────────────
        "prs_merged":       prs_merged,        # → pr_success_rate numerator
        "prs_opened":       prs_opened,        # → pr_success_rate denominator
        # ── social features (GraphQL proxies) ────────────────────────────
        "issue_comments":   contribs["issue_comments"],  # → social_engagement_ratio
        "pr_comments":      contribs["pr_comments"],     # → social_engagement_ratio
        "total_commits":    contribs["total_commits"],   # → social_engagement_ratio
        # ── activity features ─────────────────────────────────────────────
        "events_90d":       events_90d,        # → recent_velocity
        "distinct_repos":   contribs["distinct_repos"],  # → repo_density
        "activity_dates":   activity_dates,    # → max_inactivity_streak
        # ── structural features ───────────────────────────────────────────
        "org_count":        org_count,         # → is_org_member
        "ssh_key_count":    ssh_key_count,     # → has_secure_profile
        "gpg_key_count":    gpg_key_count,     # → has_secure_profile
    }


# ──────────────────────────────────────────────
# Step 3 — collect all profiles into a DataFrame
# ──────────────────────────────────────────────

def fetch_all_users(count: int = 900) -> pd.DataFrame:
    """
    Full pipeline: get usernames → fetch profiles → return raw DataFrame.
    Checkpoints to data/raw/users_raw_checkpoint.json every 50 users so
    progress is not lost if the run is interrupted.

    Approximate runtime (authenticated token):
        ~8 API calls/user × 900 users = 7,200 calls
        Bottleneck: Search API (30 req/min, 2 calls/user) → ~60 min for search alone
        Total wall time: ~2–3 hours depending on network latency
    """
    usernames = get_usernames_spread(total=count)
    records: list[dict] = []

    print(f"\n[scraper] Fetching profiles for {len(usernames)} users...")
    print(f"[scraper] Estimated wall time: ~{len(usernames) * 6 / 60:.0f} min (search API is the bottleneck)")

    for i, username in enumerate(usernames):
        profile = fetch_user_profile(username)
        if profile:
            records.append(profile)

        time.sleep(0.5)

        if (i + 1) % 50 == 0:
            os.makedirs("data/raw", exist_ok=True)
            pd.DataFrame(records).to_json(
                "data/raw/users_raw_checkpoint.json",
                orient="records",
                indent=2,
            )
            print(f"[scraper] [{i + 1}/{len(usernames)}] Checkpoint saved — {len(records)} profiles so far")

    df = pd.DataFrame(records)

    # Drop rows where the two timestamps essential for every time-based feature are absent
    before = len(df)
    df = df.dropna(subset=["created_at", "last_activity_at"])
    if dropped := before - len(df):
        print(f"[scraper] Dropped {dropped} rows with missing critical timestamps")

    print(f"\n[scraper] Done. {len(df)} clean profiles collected.")
    return df


# ──────────────────────────────────────────────
# Loader — use instead of fetch_all_users()
# once you already have the JSON file saved
# ──────────────────────────────────────────────

def load_raw_data(path: str = "data/raw/users_raw.json") -> pd.DataFrame:
    """
    Loads the JSON produced by fetch_all_users() into a DataFrame.

    The activity_dates column is stored as a JSON array per row and is
    restored as a Python list automatically by read_json with orient="records".
    Pass this DataFrame directly into features.generate_features() or get_X_y().

    Usage:
        from app.scraper import load_raw_data
        df = load_raw_data()
    """
    df = pd.read_json(path, orient="records")

    print(f"[scraper] Loaded {len(df)} records from {path}")
    print(f"[scraper] Null counts:\n{df.isnull().sum()}\n")

    return df


# ──────────────────────────────────────────────
# Entry point — run directly to save raw data
# ──────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("../data/raw", exist_ok=True)

    df = fetch_all_users(count=90)

    output_path = "../data/raw/users_raw.json"
    df.to_json(output_path, orient="records", indent=2)

    print(f"\n[scraper] Raw data saved to {output_path}")
    print(df.drop(columns=["activity_dates"]).head())  # suppress long list column in preview
    print(f"\nShape: {df.shape}")
    print(f"Null counts:\n{df.isnull().sum()}")