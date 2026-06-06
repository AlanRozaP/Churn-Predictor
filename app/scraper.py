import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
BASE_URL = "https://api.github.com"

headers = {}
if GITHUB_TOKEN:
    headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
else:
    print("[WARNING] No GITHUB_TOKEN found. Rate limit: 60 req/hr. Add token to .env for 5000 req/hr.")


# ──────────────────────────────────────────────
# Step 1 — collect usernames via /users list
# ──────────────────────────────────────────────

def get_usernames(count: int = 300, since: int = 0) -> list[str]:
    """
    Walks GitHub's paginated /users endpoint to collect `count` usernames.
    `since` is the user ID to start from — increase it to sample different
    parts of the user base (e.g. since=5000000 for newer accounts).
    """
    usernames = []

    print(f"[scraper] Collecting {count} usernames starting from ID {since}...")

    while len(usernames) < count:
        try:
            r = requests.get(
                f"{BASE_URL}/users",
                headers=headers,
                params={"since": since, "per_page": 100},
                timeout=10,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[scraper] Error fetching user list: {e}")
            break

        batch = r.json()
        if not batch:
            print("[scraper] No more users returned by API.")
            break

        usernames += [u["login"] for u in batch]
        since = batch[-1]["id"]

        remaining = r.headers.get("X-RateLimit-Remaining", "?")
        print(f"[scraper] Collected {len(usernames)} usernames | rate limit remaining: {remaining}")

        time.sleep(0.5)

    return usernames[:count]


# ──────────────────────────────────────────────
# Step 2 — fetch full profile for each username
# ──────────────────────────────────────────────

def fetch_user_profile(username: str) -> dict | None:
    """
    Fetches /users/{username} and returns only the raw fields
    needed for feature generation. Returns None on any error.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/users/{username}",
            headers=headers,
            timeout=10,
        )

        if r.status_code == 404:
            print(f"[scraper] User not found: {username} — skipping")
            return None

        if r.status_code == 403:
            reset = r.headers.get("X-RateLimit-Reset", "unknown")
            print(f"[scraper] Rate limit hit. Resets at {reset}. Sleeping 60s...")
            time.sleep(60)
            return None

        r.raise_for_status()
        data = r.json()

        return {
            "username":     data.get("login"),
            "followers":    data.get("followers", 0),
            "following":    data.get("following", 0),
            "public_repos": data.get("public_repos", 0),
            "public_gists": data.get("public_gists", 0),
            "created_at":   data.get("created_at"),   # → account_age_days
            "updated_at":   data.get("updated_at"),   # → days_since_last_activity
            "bio":          data.get("bio"),           # → has_bio
        }

    except requests.RequestException as e:
        print(f"[scraper] Error fetching {username}: {e}")
        return None


# ──────────────────────────────────────────────
# Step 3 — collect all profiles into a DataFrame
# ──────────────────────────────────────────────

def fetch_all_users(count: int = 300, since: int = 0) -> pd.DataFrame:
    """
    Full pipeline: get usernames → fetch profiles → return raw DataFrame.
    Saves a checkpoint to data/raw/users_raw.csv after every 50 users
    so you don't lose progress if the script crashes mid-run.
    """
    usernames = get_usernames(count=count, since=since)
    records = []

    print(f"\n[scraper] Fetching profiles for {len(usernames)} users...")

    for i, username in enumerate(usernames):
        profile = fetch_user_profile(username)

        if profile:
            records.append(profile)

        # sleep to stay well within rate limits
        time.sleep(1)

        # checkpoint every 50 users
        if (i + 1) % 50 == 0:
            checkpoint = pd.DataFrame(records)
            checkpoint.to_csv("data/raw/users_raw_checkpoint.csv", index=False)
            print(f"[scraper] Checkpoint saved — {len(records)} profiles collected so far")

    df = pd.DataFrame(records)

    # drop rows where critical timestamp fields are missing
    before = len(df)
    df = df.dropna(subset=["created_at", "updated_at"])
    dropped = before - len(df)
    if dropped:
        print(f"[scraper] Dropped {dropped} rows with missing timestamps")

    print(f"\n[scraper] Done. {len(df)} clean profiles collected.")
    return df


# ──────────────────────────────────────────────
# Entry point — run directly to save raw data
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)

    df = fetch_all_users(count=300, since=0)

    output_path = "data/raw/users_raw.json"
    df.to_json(output_path, index=False)
    print(f"\n[scraper] Raw data saved to {output_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Null counts:\n{df.isnull().sum()}")