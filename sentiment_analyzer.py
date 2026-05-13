"""
sentiment_analyzer.py
---------------------
Two-stage sentiment analysis:
  1. Lexicon-based scorer (inline positive/negative word lists, no download)
  2. Logistic Regression on TF-IDF features -> trained classifier

Output labels: positive | negative | neutral
"""

import os
import numpy as np
import pandas as pd
import joblib

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_vectorizer.pkl")

# ---------------------------------------------------------------------------
# Inline sentiment lexicon (subset of VADER + common review words)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = set("""
amazing awesome excellent fantastic wonderful brilliant outstanding superb perfect
great good love loved loving happy satisfied pleased delighted impressed
recommend recommend worth fast quick beautiful nice quality sturdy durable
reliable efficient helpful easy comfortable convenient affordable best better
flawless gorgeous incredible remarkable exceptional splendid magnificent
""".split())

_NEGATIVE_WORDS = set("""
terrible awful horrible bad poor worst waste garbage trash junk broken defective
damaged disappointed disappointing useless cheap flimsy fake counterfeit misleading
slow late delayed missing lost damaged wrong dirty smells ugly rude unhelpful
refused refused refused refused scam fraud never avoid regret return refund
broke cracked stopped working fell apart low quality poor quality not working
""".split())

_INTENSIFIERS = {"very", "extremely", "absolutely", "completely", "totally", "really",
                 "so", "such", "incredibly", "unbelievably", "super", "highly"}
_NEGATORS     = {"not", "no", "never", "cannot", "cant", "neither", "nor", "without"}


def lexicon_sentiment(text: str) -> dict:
    """Score sentiment using inline word lists."""
    words = text.lower().split()
    pos_score = 0.0
    neg_score = 0.0
    total = max(len(words), 1)

    i = 0
    while i < len(words):
        w = words[i]
        multiplier = 1.0
        # Check previous word for intensifier/negator
        prev = words[i-1] if i > 0 else ""
        if prev in _INTENSIFIERS:
            multiplier = 1.5
        elif prev in _NEGATORS:
            multiplier = -1.0

        if w in _POSITIVE_WORDS:
            if multiplier < 0:
                neg_score += abs(multiplier)
            else:
                pos_score += multiplier
        elif w in _NEGATIVE_WORDS:
            if multiplier < 0:
                pos_score += abs(multiplier)
            else:
                neg_score += multiplier
        i += 1

    # Normalize
    pos_norm = pos_score / total * 10
    neg_norm = neg_score / total * 10
    compound = (pos_norm - neg_norm) / max(pos_norm + neg_norm + 1e-6, 1) * 2 - 1
    compound = max(-1.0, min(1.0, compound))

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "pos": round(pos_norm, 3),
        "neg": round(neg_norm, 3),
        "compound": round(compound, 3),
        "label": label,
    }


# Also try VADER if available
def _try_vader(text: str):
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        scores["label"] = label
        return scores
    except Exception:
        return lexicon_sentiment(text)


def vader_sentiment(text: str) -> dict:
    return _try_vader(str(text))


# ---------------------------------------------------------------------------
# ML Classifier (Logistic Regression + TF-IDF)
# ---------------------------------------------------------------------------

def _rating_to_label(rating: float) -> str:
    if rating >= 4:   return "positive"
    elif rating <= 2: return "negative"
    return "neutral"


def train_sentiment_classifier(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    df = df[df["clean_text"].str.strip().ne("") & df["overall"].notna()].copy()
    df["sentiment_label"] = df["overall"].apply(_rating_to_label)

    X = df["clean_text"].values
    y = df["sentiment_label"].values

    unique_classes = np.unique(y)
    stratify = y if len(unique_classes) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("[sentiment] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[sentiment] Model saved to {MODEL_PATH}")
    return model, vectorizer, report


def load_sentiment_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def predict_sentiment(texts, model=None, vectorizer=None):
    if model is not None and vectorizer is not None:
        vecs = vectorizer.transform(texts)
        return list(model.predict(vecs))
    return [vader_sentiment(t)["label"] for t in texts]


def apply_sentiment(df, model=None, vectorizer=None):
    df = df.copy()

    vader_results = df["reviewText"].apply(vader_sentiment)
    df["vader_compound"] = vader_results.apply(lambda x: x["compound"])
    df["vader_label"]    = vader_results.apply(lambda x: x["label"])

    if model is not None and vectorizer is not None:
        df["ml_sentiment"] = predict_sentiment(df["clean_text"].tolist(), model, vectorizer)
        df["final_sentiment"] = df.apply(
            lambda r: r["ml_sentiment"] if r["ml_sentiment"] == r["vader_label"] else r["vader_label"],
            axis=1
        )
    else:
        df["ml_sentiment"]    = df["vader_label"]
        df["final_sentiment"] = df["vader_label"]

    return df


if __name__ == "__main__":
    samples = [
        "Absolutely love this product! Best purchase ever.",
        "Terrible. Broke after one day. Waste of money.",
        "It is okay. Does the job I suppose.",
    ]
    for s in samples:
        r = vader_sentiment(s)
        print(f"Text: {s[:60]}")
        print(f"  Score: {r['label']} (compound={r['compound']:.3f})")
        print()
