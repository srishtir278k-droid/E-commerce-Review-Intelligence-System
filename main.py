"""
main.py
-------
Entry point for the E-commerce Review Intelligence System.

Usage:
    python main.py                          # run full pipeline on synthetic data
    python main.py --csv path/to/file.csv  # run on your own CSV
    python main.py --csv data/amazon_reviews.csv --rows 2000
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="E-commerce Review Intelligence System"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to Amazon Reviews CSV. Omit to use synthetic data."
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Limit number of rows to load (default: all)"
    )
    parser.add_argument(
        "--skip-charts", action="store_true",
        help="Skip chart generation (faster runs)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path=None, n_rows=None, skip_charts=False):
    print("\n" + "="*60)
    print("  E-COMMERCE REVIEW INTELLIGENCE SYSTEM")
    print("="*60 + "\n")

    # ---- 1. Load data ----
    from data_loader import load_dataset, DATASET_PATH
    path = csv_path if csv_path else DATASET_PATH
    df = load_dataset(path=path, n_rows=n_rows)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # ---- 2. Preprocess ----
    print("STEP 1: Text Preprocessing...")
    from preprocessor import preprocess_dataframe
    df = preprocess_dataframe(df)
    print(f"Sample cleaned text: {df['clean_text'].iloc[0][:100]}\n")

    # ---- 3. Sentiment Analysis ----
    print("STEP 2: Sentiment Analysis (VADER + ML)...")
    from sentiment_analyzer import (
        train_sentiment_classifier, load_sentiment_model, apply_sentiment
    )
    model_s, vec_s = load_sentiment_model()
    if model_s is None:
        model_s, vec_s, _ = train_sentiment_classifier(df)
    df = apply_sentiment(df, model_s, vec_s)
    print(df["final_sentiment"].value_counts())
    print()

    # ---- 4. Fake Review Detection ----
    print("STEP 3: Fake Review Detection...")
    from fake_review_detector import train_fake_detector, load_fake_model, apply_fake_detection
    model_f = load_fake_model()
    if model_f is None:
        model_f, _ = train_fake_detector(df)
    df = apply_fake_detection(df, model_f)
    print(f"Fake reviews detected: {df['is_fake'].sum()} / {len(df)} "
          f"({100*df['is_fake'].mean():.1f}%)")
    print()

    # ---- 5. Issue Classification ----
    print("STEP 4: Issue Classification...")
    from issue_classifier import train_issue_classifier, load_issue_model, apply_issue_classification
    model_i, vec_i, enc_i = load_issue_model()
    if model_i is None:
        model_i, vec_i, enc_i, _ = train_issue_classifier(df)
    df = apply_issue_classification(df, model_i, vec_i, enc_i)
    print(df["issue_category"].value_counts())
    print()

    # ---- 6. Evaluation ----
    print("STEP 5: Model Evaluation...")
    from evaluator import (
        evaluate_sentiment, evaluate_fake_detection,
        evaluate_issue_classification, generate_summary_report
    )
    sm = evaluate_sentiment(df)
    fm = evaluate_fake_detection(df)
    im = evaluate_issue_classification(df)
    generate_summary_report(df, sm, fm, im)

    # ---- 7. Visualizations ----
    if not skip_charts:
        print("\nSTEP 6: Generating Charts...")
        from visualizer import generate_all_charts
        generate_all_charts(df)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Output files saved to: {os.path.abspath('outputs/')}")
    print()

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    df = run_pipeline(
        csv_path=args.csv,
        n_rows=args.rows,
        skip_charts=args.skip_charts,
    )
