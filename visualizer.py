"""
visualizer.py
-------------
Generates all charts and saves them to /outputs/.

Charts produced:
  1. Sentiment distribution (bar + pie)
  2. Rating distribution
  3. Fake vs Genuine pie
  4. Issue category breakdown
  5. Sentiment vs Rating heatmap
  6. Review length distribution
  7. WordCloud (genuine reviews)
  8. Product health scorecard
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "positive": "#2ECC71",
    "neutral":  "#95A5A6",
    "negative": "#E74C3C",
    "genuine":  "#3498DB",
    "fake":     "#E67E22",
    "delivery": "#9B59B6",
    "quality":  "#E74C3C",
    "service":  "#F39C12",
    "other":    "#BDC3C7",
}


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualizer] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame):
    counts = df["final_sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    colors = [COLORS.get(c, "#AAA") for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sentiment Distribution", fontsize=16, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_title("Review Count by Sentiment")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontweight="bold")

    # Pie chart
    non_zero = counts[counts > 0]
    axes[1].pie(
        non_zero.values,
        labels=non_zero.index,
        colors=[COLORS.get(c, "#AAA") for c in non_zero.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Sentiment Share")

    plt.tight_layout()
    return _save(fig, "1_sentiment_distribution.png")


def plot_rating_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    rating_counts = df["overall"].value_counts().sort_index()
    palette = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#27AE60"]
    bars = ax.bar(rating_counts.index.astype(int), rating_counts.values,
                  color=palette[:len(rating_counts)], edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    for bar, val in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    return _save(fig, "2_rating_distribution.png")


def plot_fake_detection(df: pd.DataFrame):
    if "is_fake" not in df.columns:
        return None
    counts = df["is_fake"].value_counts().rename({0: "Genuine", 1: "Fake"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fake Review Detection", fontsize=16, fontweight="bold")

    # Pie
    axes[0].pie(
        counts.values,
        labels=counts.index,
        colors=[COLORS["genuine"], COLORS["fake"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        explode=[0, 0.05],
    )
    axes[0].set_title("Genuine vs Fake")

    # Confidence histogram
    if "fake_confidence" in df.columns:
        genuine_conf = df[df["is_fake"] == 0]["fake_confidence"]
        fake_conf    = df[df["is_fake"] == 1]["fake_confidence"]
        axes[1].hist(genuine_conf, bins=20, alpha=0.7, color=COLORS["genuine"], label="Genuine", edgecolor="white")
        axes[1].hist(fake_conf,    bins=20, alpha=0.7, color=COLORS["fake"],    label="Fake",    edgecolor="white")
        axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold")
        axes[1].set_xlabel("Fake Confidence Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Confidence Distribution")
        axes[1].legend()

    plt.tight_layout()
    return _save(fig, "3_fake_review_detection.png")


def plot_issue_categories(df: pd.DataFrame):
    if "issue_category" not in df.columns:
        return None
    counts = df["issue_category"].value_counts()
    colors = [COLORS.get(c, "#95A5A6") for c in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Issue Category Analysis", fontsize=16, fontweight="bold")

    bars = axes[0].barh(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Number of Reviews")
    axes[0].set_title("Reviews per Category")
    axes[0].invert_yaxis()
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10)

    non_zero = counts[counts > 0]
    axes[1].pie(
        non_zero.values,
        labels=non_zero.index,
        colors=[COLORS.get(c, "#95A5A6") for c in non_zero.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Issue Share")

    plt.tight_layout()
    return _save(fig, "4_issue_categories.png")


def plot_sentiment_rating_heatmap(df: pd.DataFrame):
    if "final_sentiment" not in df.columns:
        return None
    pivot = pd.crosstab(df["final_sentiment"], df["overall"].astype(int))
    pivot = pivot.reindex(["positive", "neutral", "negative"], fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Review Count"})
    ax.set_title("Sentiment vs Star Rating", fontsize=14, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Sentiment")
    plt.tight_layout()
    return _save(fig, "5_sentiment_rating_heatmap.png")


def plot_review_length(df: pd.DataFrame):
    if "review_length" not in df.columns:
        df = df.copy()
        df["review_length"] = df["reviewText"].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Review Length Analysis", fontsize=16, fontweight="bold")

    # Histogram
    axes[0].hist(df["review_length"].clip(upper=300), bins=40,
                 color=COLORS["genuine"], edgecolor="white", linewidth=0.6)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Review Lengths")

    # Box plot by sentiment
    if "final_sentiment" in df.columns:
        order = ["positive", "neutral", "negative"]
        palette = {s: COLORS[s] for s in order}
        plot_df = df[df["final_sentiment"].isin(order)].copy()
        plot_df["review_length_clipped"] = plot_df["review_length"].clip(upper=300)
        sns.boxplot(
            data=plot_df, x="final_sentiment", y="review_length_clipped",
            order=order, palette=palette, ax=axes[1],
            showfliers=False, linewidth=0.8,
        )
        axes[1].set_title("Review Length by Sentiment")
        axes[1].set_xlabel("Sentiment")
        axes[1].set_ylabel("Word Count")

    plt.tight_layout()
    return _save(fig, "6_review_length.png")


def plot_wordcloud(df: pd.DataFrame):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("[visualizer] wordcloud not installed, skipping.")
        return None

    genuine_text = " ".join(
        df[df.get("is_fake", pd.Series(0, index=df.index)) == 0]["clean_text"].fillna("")
    ) if "is_fake" in df.columns else " ".join(df["clean_text"].fillna(""))

    if not genuine_text.strip():
        return None

    wc = WordCloud(
        width=900, height=450, background_color="white",
        colormap="viridis", max_words=150, collocations=False,
    ).generate(genuine_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Common Words in Genuine Reviews", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return _save(fig, "7_wordcloud.png")


def plot_health_scorecard(df: pd.DataFrame):
    """Summary dashboard with key metrics."""
    total    = len(df)
    pos_pct  = round(100 * (df.get("final_sentiment", pd.Series()) == "positive").sum() / max(total, 1), 1)
    neg_pct  = round(100 * (df.get("final_sentiment", pd.Series()) == "negative").sum() / max(total, 1), 1)
    fake_pct = round(100 * df.get("is_fake", pd.Series(0)).sum() / max(total, 1), 1)
    avg_rating = round(df["overall"].mean(), 2)

    issue_counts = df.get("issue_category", pd.Series()).value_counts()
    top_issue = issue_counts.index[0] if len(issue_counts) > 0 else "N/A"

    # Health score: weighted formula
    health = max(0, min(100, round(
        pos_pct * 0.4 +
        (avg_rating / 5) * 100 * 0.4 +
        max(0, 100 - fake_pct * 2) * 0.2
    , 1)))

    health_color = "#2ECC71" if health >= 70 else "#F39C12" if health >= 40 else "#E74C3C"

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    def metric_card(ax, value, label, color="#3498DB", unit=""):
        ax.set_facecolor("white")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#DEE2E6")
        ax.text(0.5, 0.65, f"{value}{unit}", ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=11, color="#6C757D", wrap=True)

    ax0 = fig.add_subplot(gs[0, 0])
    metric_card(ax0, health, "Health Score", health_color, "/100")

    ax1 = fig.add_subplot(gs[0, 1])
    metric_card(ax1, avg_rating, "Avg Rating", "#F39C12", " ★")

    ax2 = fig.add_subplot(gs[0, 2])
    metric_card(ax2, f"{pos_pct}", "Positive %", "#2ECC71", "%")

    ax3 = fig.add_subplot(gs[0, 3])
    metric_card(ax3, f"{fake_pct}", "Fake Review %", "#E67E22", "%")

    ax4 = fig.add_subplot(gs[1, 0])
    metric_card(ax4, total, "Total Reviews", "#3498DB")

    ax5 = fig.add_subplot(gs[1, 1])
    metric_card(ax5, f"{neg_pct}", "Negative %", "#E74C3C", "%")

    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.set_facecolor("white")
    ax6.axis("off")
    recommendations = []
    if fake_pct > 15:
        recommendations.append("⚠ High fake review rate detected — investigate seller activity")
    if neg_pct > 30:
        recommendations.append("⚠ High negative sentiment — check product quality issues")
    if top_issue in ["delivery", "quality", "service"]:
        recommendations.append(f"⚠ Top reported issue: {top_issue.upper()} — prioritize resolution")
    if avg_rating < 3.5:
        recommendations.append("⚠ Low average rating — product improvements needed")
    if not recommendations:
        recommendations.append("✓ Product reviews look healthy — maintain current standards")
    rec_text = "\n".join(recommendations)
    ax6.text(0.05, 0.5, f"Recommendations:\n\n{rec_text}",
             transform=ax6.transAxes, fontsize=11,
             va="center", ha="left", color="#2C3E50",
             linespacing=1.8)
    for spine in ax6.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#DEE2E6")

    fig.suptitle("Product Review Health Scorecard", fontsize=18,
                 fontweight="bold", color="#2C3E50", y=1.02)
    return _save(fig, "8_health_scorecard.png")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def generate_all_charts(df: pd.DataFrame):
    """Generate all charts. Returns list of saved paths."""
    paths = []
    fns = [
        plot_sentiment_distribution,
        plot_rating_distribution,
        plot_fake_detection,
        plot_issue_categories,
        plot_sentiment_rating_heatmap,
        plot_review_length,
        plot_wordcloud,
        plot_health_scorecard,
    ]
    for fn in fns:
        try:
            p = fn(df)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"[visualizer] Error in {fn.__name__}: {e}")
    print(f"[visualizer] Generated {len(paths)} charts in {OUTPUT_DIR}/")
    return paths
