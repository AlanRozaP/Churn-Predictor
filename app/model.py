import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report

from features import FEATURE_COLUMNS, get_X_y

MODEL_PATH = Path(__file__).parent / "model.pkl"


# ──────────────────────────────────────────────
# Training — run this from the notebook, not at API startup
# ──────────────────────────────────────────────

def train(df: pd.DataFrame, save: bool = True) -> RandomForestClassifier:
    """
    Trains a Random Forest on the raw DataFrame from scraper.py.
    Prints a classification report and cross-validation scores.
    Saves the model to model.pkl if save=True.

    Usage from notebook:
        from app.model import train
        model = train(raw_df)
    """
    X, y = get_X_y(df)

    print(f"[model] Training on {len(X)} samples | {y.sum()} churned, {(y==0).sum()} retained")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",   # handles class imbalance automatically
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── evaluation ────────────────────────────
    y_pred = rf.predict(X_test)
    print("\n[model] Test set performance:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    cv_results = cross_validate(
        rf, X, y, cv=5,
        scoring=["accuracy", "f1", "precision", "recall"],
    )
    print("[model] 5-fold cross-validation:")
    for metric, scores in cv_results.items():
        if metric.startswith("test_"):
            name = metric.replace("test_", "")
            print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # ── feature importance summary ─────────────
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
    print("\n[model] Feature importances (Random Forest):")
    print(importances.sort_values(ascending=False).to_string())

    if save:
        save_model(rf)

    return rf


def save_model(model: RandomForestClassifier) -> None:
    joblib.dump(model, MODEL_PATH)
    print(f"\n[model] Model saved to {MODEL_PATH}")


# ──────────────────────────────────────────────
# Inference — used by the FastAPI app at runtime
# ──────────────────────────────────────────────

def load_model() -> RandomForestClassifier:
    """
    Loads the trained model from disk.
    Raises a clear error if model.pkl doesn't exist yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run train() from the notebook first to generate it."
        )
    return joblib.load(MODEL_PATH)


def predict(model: RandomForestClassifier, feature_array: np.ndarray) -> dict:
    """
    Runs inference on a pre-built feature array (from features.features_from_dict).
    Returns a dict with 'churned' (bool) and 'churn_probability' (float).
    """
    churn_class   = model.predict(feature_array)[0]
    churn_prob    = model.predict_proba(feature_array)[0][1]

    return {
        "churned":           bool(churn_class),
        "churn_probability": round(float(churn_prob), 3),
    }