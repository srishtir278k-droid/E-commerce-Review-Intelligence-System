"""
fake_review_detector.py
-----------------------
Detects potentially fake/spam reviews using:
  - Engineered heuristic features (length, rating anomaly, exclamation density, etc.)
  - Random Forest binary classifier
  - Rule-based flag for extreme patterns

Output: is_fake (0/1), fake_confidence (0.0-1.0), fake_flags (list of reasons)
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fake_detector.pkl")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_EXCLAIM_RE = re.compile(r"!")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature matrix from raw review columns.
    Returns a DataFrame of numeric features.
    """
    feats = pd.DataFrame(index=df.index)

    texts = df["reviewText"].fillna("").astype(str)
    ratings = df["overall"].fillna(3).astype(float)
    verified = df["verified"].astype(bool) if "verified" in df.columns else pd.Series(True, index=df.index)

    # Text-based features
    feats["review_length"]     = texts.apply(lambda x: len(x.split()))
    feats["char_length"]       = texts.apply(len)
    feats["exclamation_count"] = texts.apply(lambda x: len(_EXCLAIM_RE.findall(x)))
    feats["caps_word_count"]   = texts.apply(lambda x: len(_CAPS_WORD_RE.findall(x)))
    feats["unique_word_ratio"] = texts.apply(
        lambda x: len(set(x.lower().split())) / max(len(x.split()), 1)
    )
    feats["avg_word_length"]   = texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    feats["exclamation_density"] = feats["exclamation_count"] / feats["review_length"].clip(lower=1)
    feats["caps_density"]        = feats["caps_word_count"]   / feats["review_length"].clip(lower=1)

    # Rating-based features
    feats["rating"] = ratings
    feats["is_extreme_rating"] = ((ratings == 1) | (ratings == 5)).astype(int)
    feats["is_five_star"] = (ratings == 5).astype(int)

    # Product-level rating anomaly
    if "asin" in df.columns:
        product_avg = df.groupby("asin")["overall"].transform("mean")
        feats["rating_deviation"] = (ratings - product_avg).abs()
    else:
        feats["rating_deviation"] = 0.0

    # Verification
    feats["is_unverified"] = (~verified).astype(int)

    # Short review with extreme rating -> suspicious
    feats["short_extreme"] = (
        (feats["review_length"] < 10) & feats["is_extreme_rating"].astype(bool)
    ).astype(int)

    return feats


# ---------------------------------------------------------------------------
# Rule-based flagging
# ---------------------------------------------------------------------------

def rule_based_flags(row_feats: pd.Series) -> list:
    """Return a list of human-readable flag strings for a single review."""
    flags = []
    if row_feats["review_length"] < 5:
        flags.append("Very short review")
    if row_feats["exclamation_density"] > 0.3:
        flags.append("Excessive exclamations")
    if row_feats["caps_density"] > 0.2:
        flags.append("Excessive caps words")
    if row_feats["unique_word_ratio"] < 0.4:
        flags.append("Repetitive language")
    if row_feats["is_unverified"] == 1 and row_feats["is_five_star"] == 1:
        flags.append("Unverified 5-star review")
    if row_feats["short_extreme"] == 1:
        flags.append("Short + extreme rating")
    return flags


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_fake_detector(df: pd.DataFrame):
    """
    Train a Random Forest on engineered features.
    Uses 'true_fake' column if present; otherwise derives labels heuristically.
    Saves model to /models/.
    Returns (model, report_dict).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    feats = extract_features(df)

    if "true_fake" in df.columns:
        labels = df["true_fake"].fillna(0).astype(int)
    else:
        # Heuristic labeling: unverified + short + extreme rating -> fake
        labels = (
            (feats["is_unverified"] == 1) &
            (feats["review_length"] < 15) &
            (feats["is_extreme_rating"] == 1)
        ).astype(int)

    X = feats.values
    y = labels.values

    if y.sum() < 5:
        print("[fake_detector] Too few fake samples for training. Using rule-based detection only.")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("[fake_detector] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["genuine", "fake"], zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"[fake_detector] Model saved to {MODEL_PATH}")
    return model, report


def load_fake_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def apply_fake_detection(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Add is_fake, fake_confidence, fake_flags columns to df.
    Falls back to rule-based detection if no trained model.
    """
    df = df.copy()
    feats = extract_features(df)

    if model is not None:
        try:
            proba = model.predict_proba(feats.values)
            # Handle case where model only learned 1 class
            if proba.shape[1] >= 2:
                df["fake_confidence"] = proba[:, 1]
            else:
                df["fake_confidence"] = proba[:, 0]
        except Exception:
            df["fake_confidence"] = 0.1
        df["is_fake"] = (df["fake_confidence"] >= 0.5).astype(int)
    else:
        # Rule-based fallback score
        score = (
            feats["is_unverified"] * 0.30 +
            feats["exclamation_density"].clip(0, 1) * 0.20 +
            feats["caps_density"].clip(0, 1) * 0.15 +
            (1 - feats["unique_word_ratio"]) * 0.15 +
            feats["short_extreme"] * 0.20
        )
        df["fake_confidence"] = score.clip(0, 1)
        df["is_fake"]         = (df["fake_confidence"] >= 0.5).astype(int)

    df["fake_flags"] = feats.apply(rule_based_flags, axis=1)
    return df
