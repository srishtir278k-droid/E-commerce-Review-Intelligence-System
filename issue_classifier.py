"""
issue_classifier.py
-------------------
Classifies reviews into issue categories:
  - delivery   : shipping delays, lost packages, tracking issues
  - quality    : product defects, poor materials, broken items
  - service    : customer support problems, return/refund issues
  - positive   : no complaint, happy customer
  - other      : does not fit above

Two approaches:
  1. Keyword lexicon (fast, no training needed)
  2. Naive Bayes + TF-IDF (trained classifier, more accurate)
"""

import os
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "models", "issue_classifier.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "issue_vectorizer.pkl")
ENCODER_PATH    = os.path.join(os.path.dirname(__file__), "models", "issue_encoder.pkl")


# ---------------------------------------------------------------------------
# Keyword lexicon
# ---------------------------------------------------------------------------

ISSUE_KEYWORDS = {
    "delivery": [
        "late", "delay", "delayed", "shipping", "ship", "arrived", "arrival",
        "package", "transit", "courier", "tracking", "dispatch", "lost",
        "missing", "delivery", "deliver", "slow", "weeks", "days late",
        "not arrived", "still waiting", "never received",
    ],
    "quality": [
        "broke", "broken", "defect", "defective", "cheap", "poor quality",
        "low quality", "bad quality", "material", "flimsy", "fragile",
        "fall apart", "fell apart", "crack", "cracked", "not working",
        "stopped working", "useless", "fake", "counterfeit", "not as described",
        "misleading", "wrong item", "different from picture",
    ],
    "service": [
        "customer service", "support", "refund", "return", "replacement",
        "response", "rude", "unhelpful", "no help", "complaint", "contact",
        "seller", "dispute", "resolution", "ignored", "waiting for reply",
        "no response", "terrible service", "horrible service",
    ],
    "positive": [
        "amazing", "excellent", "perfect", "wonderful", "fantastic", "great",
        "love", "happy", "satisfied", "recommend", "best", "awesome",
        "outstanding", "superb", "brilliant", "highly recommend",
    ],
}


def keyword_classify(text: str) -> str:
    """Return the best-matching issue category using keyword matching."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in ISSUE_KEYWORDS}
    for cat, keywords in ISSUE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _derive_labels(df: pd.DataFrame) -> pd.Series:
    """Derive issue labels from true_category or keyword matching."""
    category_map = {
        "delivery": "delivery",
        "quality":  "quality",
        "service":  "service",
        "positive": "positive",
        "negative": "quality",
        "neutral":  "other",
        "fake":     "other",
    }
    if "true_category" in df.columns:
        return df["true_category"].map(category_map).fillna("other")
    return df["reviewText"].apply(keyword_classify)


def train_issue_classifier(df: pd.DataFrame):
    """
    Train a Naive Bayes multi-class classifier.
    Saves model, vectorizer, and label encoder.
    Returns (model, vectorizer, encoder, report_dict).
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = df[df["clean_text"].str.strip().ne("")].copy()
    df["issue_label"] = _derive_labels(df)

    # Need enough samples per class
    class_counts = df["issue_label"].value_counts()
    valid_classes = class_counts[class_counts >= 3].index.tolist()
    df = df[df["issue_label"].isin(valid_classes)]

    if len(df) < 20:
        print("[issue_classifier] Not enough data for training. Using keyword-based classification.")
        return None, None, None, {}

    X = df["clean_text"].values
    y = df["issue_label"].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    target_names = encoder.inverse_transform(np.unique(y_test))
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    print("[issue_classifier] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder,    ENCODER_PATH)
    print(f"[issue_classifier] Model saved to {MODEL_PATH}")

    return model, vectorizer, encoder, report


def load_issue_model():
    if all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]):
        return (
            joblib.load(MODEL_PATH),
            joblib.load(VECTORIZER_PATH),
            joblib.load(ENCODER_PATH),
        )
    return None, None, None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def apply_issue_classification(df: pd.DataFrame, model=None, vectorizer=None, encoder=None) -> pd.DataFrame:
    """Add issue_category column. Falls back to keyword matching if no model."""
    df = df.copy()

    if model is not None and vectorizer is not None and encoder is not None:
        vecs = vectorizer.transform(df["clean_text"].fillna(""))
        preds = model.predict(vecs)
        df["issue_category"] = encoder.inverse_transform(preds)
    else:
        df["issue_category"] = df["reviewText"].apply(keyword_classify)

    return df
