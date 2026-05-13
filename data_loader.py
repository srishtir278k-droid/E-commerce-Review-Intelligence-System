"""
data_loader.py
--------------
Loads the Amazon Reviews dataset.
If no real CSV is found, generates a realistic synthetic dataset for testing.

Expected CSV columns (Kaggle Amazon Reviews):
    reviewText, summary, overall, verified, reviewerID, asin, unixReviewTime
"""

import os
import random
import pandas as pd
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "amazon_reviews.csv")


# ---------------------------------------------------------------------------
# Synthetic data generator (used when no real dataset is present)
# ---------------------------------------------------------------------------

POSITIVE_TEXTS = [
    "This product is absolutely amazing! Works perfectly and arrived on time.",
    "Great quality item. Very happy with my purchase. Highly recommended!",
    "Exceeded my expectations. Fast shipping and excellent packaging.",
    "Perfect product, exactly as described. Will definitely buy again.",
    "Outstanding quality. My family loves it. Five stars all the way!",
    "Wonderful item, very well made. Arrived earlier than expected.",
    "Fantastic! Works like a charm. Great value for money.",
    "Really impressed with the build quality. Delivery was super fast.",
    "Love this product! It has made my daily routine so much easier.",
    "Incredible product. Customer support was also very helpful.",
]

NEGATIVE_TEXTS = [
    "Terrible quality. Broke after just two days. Total waste of money.",
    "Very disappointed. Product looks nothing like the pictures.",
    "Arrived damaged and customer service was useless. Never buying again.",
    "Poor quality materials. Falls apart easily. Do not buy.",
    "Late delivery and the item was defective. Extremely frustrating.",
    "This is garbage. Stopped working after a week. Requesting refund.",
    "Awful product. Complete waste of money. Zero stars if I could.",
    "Horrible experience. Wrong item shipped and no resolution offered.",
    "Cheaply made. The description is completely misleading.",
    "Do not buy this. It broke immediately and smells strange.",
]

NEUTRAL_TEXTS = [
    "It is okay for the price. Does what it says but nothing special.",
    "Average product. Some things work well, others not so much.",
    "Decent enough. Delivery was on time but the item is just average.",
    "Not bad, not great either. Might be useful for some people.",
    "Works as described. Nothing exceptional though.",
    "Mediocre quality but acceptable for the price point.",
    "Does the job, but there are probably better options available.",
    "Neither impressed nor disappointed. Just an ordinary product.",
]

FAKE_TEXTS = [
    "Best product ever! Amazing! Buy now! You will love it! Five stars!",
    "PERFECT PERFECT PERFECT!! Everyone should buy this immediately!!!",
    "Great great great great great! Totally love it so much!!",
    "Absolutely the best thing I have ever bought. 10/10 would recommend!!!!",
    "Wonderful wonderful! Best purchase ever! Amazing quality amazing price!",
]

ISSUE_DELIVERY_TEXTS = [
    "Package arrived three weeks late. Completely unacceptable.",
    "Shipping took forever. My order was stuck in transit for two weeks.",
    "Delivery was delayed by ten days. Very poor logistics.",
    "Item arrived damaged due to bad packaging during shipping.",
    "The courier lost my package and it arrived two weeks after expected.",
]

ISSUE_QUALITY_TEXTS = [
    "The product feels very cheap. Material is low quality plastic.",
    "Paint chipped off after one use. Very poor manufacturing.",
    "The stitching fell apart within a week. Terrible quality control.",
    "Material is not as advertised. Feels very flimsy and fragile.",
    "Motor broke after three days. Extremely poor build quality.",
]

ISSUE_SERVICE_TEXTS = [
    "Customer service refused to process my refund. Horrible experience.",
    "Contacted support five times with no resolution whatsoever.",
    "Return process is a nightmare. They keep asking for more documents.",
    "Customer care is rude and unhelpful. Will never shop here again.",
    "No response from seller after multiple complaints. Avoid.",
]


def generate_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic Amazon-like reviews dataframe."""
    random.seed(seed)
    np.random.seed(seed)

    records = []

    pools = {
        "positive": (POSITIVE_TEXTS, [4, 5], True, 0.05),
        "negative": (NEGATIVE_TEXTS, [1, 2], True, 0.05),
        "neutral":  (NEUTRAL_TEXTS,  [3],    True, 0.05),
        "fake":     (FAKE_TEXTS,     [5],    False, 0.9),
        "delivery": (ISSUE_DELIVERY_TEXTS, [1, 2], True, 0.05),
        "quality":  (ISSUE_QUALITY_TEXTS,  [1, 2], True, 0.05),
        "service":  (ISSUE_SERVICE_TEXTS,  [1, 2], True, 0.05),
    }

    category_weights = [0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05]
    categories = list(pools.keys())

    for i in range(n):
        cat = random.choices(categories, weights=category_weights, k=1)[0]
        texts, ratings, verified, fake_prob = pools[cat]
        text = random.choice(texts)
        rating = random.choice(ratings)

        # Add slight noise to text
        if random.random() < 0.3:
            text = text + " " + random.choice([
                "Bought for my wife.", "Great gift idea.", "Would buy again.",
                "Recommend to friends.", "Very useful item.", "Good value.",
            ])

        is_fake = 1 if (not verified or random.random() < fake_prob) else 0

        records.append({
            "reviewerID":      f"REVIEWER_{i:05d}",
            "asin":            f"ASIN_{random.randint(1000, 9999)}",
            "reviewText":      text,
            "summary":         text[:50],
            "overall":         float(rating),
            "verified":        verified,
            "unixReviewTime":  1609459200 + random.randint(0, 60 * 60 * 24 * 365),
            "true_fake":       is_fake,          # ground-truth label (for evaluation)
            "true_category":   cat,              # ground-truth category
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATASET_PATH, n_rows: int = None) -> pd.DataFrame:
    """
    Load the Amazon reviews CSV.
    Falls back to synthetic data if the file is missing.

    Parameters
    ----------
    path    : path to the CSV file
    n_rows  : read only the first n_rows (None = all)

    Returns
    -------
    DataFrame with at least: reviewText, summary, overall, verified
    """
    if os.path.exists(path):
        print(f"[data_loader] Loading dataset from: {path}")
        df = pd.read_csv(path, nrows=n_rows)

        # Rename common Kaggle column variants
        rename_map = {
            "review_body":    "reviewText",
            "review_title":   "summary",
            "star_rating":    "overall",
            "verified_purchase": "verified",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
                  inplace=True)

        # Ensure required columns exist
        for col in ["reviewText", "overall"]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

        if "summary" not in df.columns:
            df["summary"] = df["reviewText"].str[:50]
        if "verified" not in df.columns:
            df["verified"] = True

        df["reviewText"] = df["reviewText"].fillna("").astype(str)
        df["overall"]    = pd.to_numeric(df["overall"], errors="coerce").fillna(3)

        if n_rows:
            df = df.head(n_rows)

        print(f"[data_loader] Loaded {len(df):,} reviews.")
        return df

    else:
        print("[data_loader] Dataset not found. Generating synthetic data (500 reviews)...")
        df = generate_synthetic_dataset(n=500)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[data_loader] Synthetic dataset saved to: {path}")
        return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df.dtypes)
