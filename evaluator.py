"""
evaluator.py
------------
Computes and prints evaluation metrics for all three models.
Also exports a summary CSV to /outputs/evaluation_report.csv.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_sentiment(df: pd.DataFrame) -> dict:
    """Evaluate sentiment if ground-truth labels derivable from ratings."""
    def rating_to_label(r):
        if r >= 4: return "positive"
        if r <= 2: return "negative"
        return "neutral"

    if "final_sentiment" not in df.columns or "overall" not in df.columns:
        return {}

    y_true = df["overall"].apply(rating_to_label)
    y_pred = df["final_sentiment"]

    acc = round(accuracy_score(y_true, y_pred), 4)
    f1  = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS — EVALUATION")
    print("="*60)
    print(f"Accuracy : {acc}")
    print(f"F1 Score : {f1}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"accuracy": acc, "f1": f1, "report": report}


def evaluate_fake_detection(df: pd.DataFrame) -> dict:
    """Evaluate fake detection if true_fake labels are available."""
    if "is_fake" not in df.columns or "true_fake" not in df.columns:
        return {}

    y_true = df["true_fake"].fillna(0).astype(int)
    y_pred = df["is_fake"].astype(int)

    acc = round(accuracy_score(y_true, y_pred), 4)
    f1  = round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4)
    cm  = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=["genuine", "fake"],
                                   output_dict=True, zero_division=0)

    print("\n" + "="*60)
    print("FAKE REVIEW DETECTION — EVALUATION")
    print("="*60)
    print(f"Accuracy : {acc}")
    print(f"F1 Score : {f1}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=["genuine", "fake"], zero_division=0))
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist(), "report": report}


def evaluate_issue_classification(df: pd.DataFrame) -> dict:
    """Evaluate issue classification if true_category is available."""
    if "issue_category" not in df.columns or "true_category" not in df.columns:
        return {}

    category_map = {
        "delivery": "delivery", "quality": "quality",
        "service":  "service",  "positive": "positive",
        "negative": "quality",  "neutral": "other", "fake": "other",
    }
    y_true = df["true_category"].map(category_map).fillna("other")
    y_pred = df["issue_category"]

    common_labels = sorted(set(y_true.unique()) & set(y_pred.unique()))
    acc = round(accuracy_score(y_true, y_pred), 4)
    f1  = round(f1_score(y_true, y_pred, average="weighted",
                          labels=common_labels, zero_division=0), 4)
    report = classification_report(y_true, y_pred, labels=common_labels,
                                   output_dict=True, zero_division=0)

    print("\n" + "="*60)
    print("ISSUE CLASSIFICATION — EVALUATION")
    print("="*60)
    print(f"Accuracy : {acc}")
    print(f"F1 Score : {f1}")
    print(classification_report(y_true, y_pred, labels=common_labels, zero_division=0))
    return {"accuracy": acc, "f1": f1, "report": report}


def generate_summary_report(df: pd.DataFrame,
                             sentiment_metrics: dict,
                             fake_metrics: dict,
                             issue_metrics: dict) -> str:
    """Save a summary CSV and print a final report."""
    rows = []

    if sentiment_metrics:
        rows.append({
            "module": "Sentiment Analysis",
            "accuracy": sentiment_metrics.get("accuracy", "N/A"),
            "f1_score": sentiment_metrics.get("f1", "N/A"),
        })
    if fake_metrics:
        rows.append({
            "module": "Fake Review Detection",
            "accuracy": fake_metrics.get("accuracy", "N/A"),
            "f1_score": fake_metrics.get("f1", "N/A"),
        })
    if issue_metrics:
        rows.append({
            "module": "Issue Classification",
            "accuracy": issue_metrics.get("accuracy", "N/A"),
            "f1_score": issue_metrics.get("f1", "N/A"),
        })

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "evaluation_report.csv")
    summary_df.to_csv(out_path, index=False)

    # Also save full analyzed dataframe
    results_path = os.path.join(OUTPUT_DIR, "analyzed_reviews.csv")
    df.drop(columns=["fake_flags"], errors="ignore").to_csv(results_path, index=False)

    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - {out_path}")
    print(f"  - {results_path}")

    return out_path
