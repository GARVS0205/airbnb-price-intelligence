#!/usr/bin/env python3
"""
review_analysis_api.py
-----------------------
Python script called by the Next.js API route via child_process.
Reads a listing_id from stdin, analyzes all reviews for that listing,
and outputs a detailed JSON summary to stdout.

Analyzes:
  - Sentiment distribution (positive / neutral / negative)
  - Review quality score
  - Key themes / topics mentioned
  - Recency and volume trends
  - Red flags (fake review detection)

Usage:
    echo '{"listing_id": 2539, "max_reviews": 200}' | python review_analysis_api.py
"""

import sys
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "reviews.csv")


# ─── Keyword themes to detect in reviews ───────────────────────────────────
THEMES = {
    "cleanliness":   ["clean", "spotless", "tidy", "fresh", "dirty", "smell", "dust", "hygiene"],
    "location":      ["location", "neighborhood", "transit", "subway", "walk", "central", "nearby"],
    "host":          ["host", "responsive", "helpful", "friendly", "communication", "welcoming", "attentive"],
    "value":         ["value", "worth", "price", "affordable", "expensive", "overpriced", "reasonable"],
    "comfort":       ["comfortable", "cozy", "spacious", "quiet", "noisy", "bed", "pillows", "sleep"],
    "amenities":     ["wifi", "kitchen", "parking", "pool", "gym", "workspace", "laundry", "tv"],
    "accuracy":      ["accurate", "as described", "exactly", "misleading", "different", "photos"],
}


def load_reviews(listing_id: int, max_reviews: int = 300):
    """Load reviews for a specific listing from the CSV."""
    import pandas as pd

    try:
        # Load only the columns we need for efficiency
        df = pd.read_csv(
            REVIEWS_PATH,
            usecols=["listing_id", "id", "date", "comments"],
            dtype={"listing_id": int, "id": int, "comments": str},
            parse_dates=["date"],
        )
    except Exception as e:
        return None, f"Could not load reviews.csv: {e}"

    reviews = df[df["listing_id"] == listing_id].copy()
    if len(reviews) == 0:
        return None, f"No reviews found for listing_id {listing_id}. Try IDs like 2539, 2595, 3176."

    reviews = reviews.sort_values("date", ascending=False)
    if max_reviews > 0:
        reviews = reviews.head(max_reviews)

    return reviews, None


def analyze_sentiment(texts: list) -> dict:
    """Run VADER sentiment on all review texts."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    scores = {"positive": 0, "neutral": 0, "negative": 0}
    compound_scores = []

    for text in texts:
        if not isinstance(text, str) or len(text.strip()) < 5:
            continue
        score = sia.polarity_scores(text[:1000])["compound"]
        compound_scores.append(score)
        if score >= 0.05:
            scores["positive"] += 1
        elif score <= -0.05:
            scores["negative"] += 1
        else:
            scores["neutral"] += 1

    total = max(1, sum(scores.values()))
    avg_compound = float(sum(compound_scores) / max(1, len(compound_scores)))

    return {
        "positive_count":  scores["positive"],
        "neutral_count":   scores["neutral"],
        "negative_count":  scores["negative"],
        "positive_pct":    round(scores["positive"] / total * 100, 1),
        "neutral_pct":     round(scores["neutral"]  / total * 100, 1),
        "negative_pct":    round(scores["negative"] / total * 100, 1),
        "avg_compound":    round(avg_compound, 4),
        "sentiment_label": (
            "Very Positive" if avg_compound >= 0.5 else
            "Positive"      if avg_compound >= 0.15 else
            "Mixed"         if avg_compound >= -0.05 else
            "Negative"
        ),
    }


def detect_themes(texts: list) -> list:
    """Count how often each theme is mentioned across all reviews."""
    theme_counts = {theme: 0 for theme in THEMES}

    for text in texts:
        if not isinstance(text, str):
            continue
        text_lower = text.lower()
        for theme, keywords in THEMES.items():
            if any(kw in text_lower for kw in keywords):
                theme_counts[theme] += 1

    total = max(1, len(texts))
    results = [
        {
            "theme":   theme,
            "count":   count,
            "pct":     round(count / total * 100, 1),
        }
        for theme, count in theme_counts.items()
    ]
    return sorted(results, key=lambda x: x["count"], reverse=True)


def compute_quality_score(
    sentiment: dict,
    themes: list,
    total_reviews: int,
    avg_length: float,
) -> dict:
    """Compute a composite review quality score (0–100)."""
    # Component 1: Sentiment score (0–40 pts)
    sentiment_score = sentiment["avg_compound"] * 40  # [-40, +40] → scaled

    # Component 2: Volume score (0–20 pts)
    volume_score = min(20, total_reviews / 5)

    # Component 3: Detail score — average review length (0–20 pts)
    detail_score = min(20, avg_length / 15)

    # Component 4: Theme diversity — how many themes appear (0–20 pts)
    theme_diversity = sum(1 for t in themes if t["pct"] >= 10)
    diversity_score = min(20, theme_diversity * 4)

    raw = max(0, sentiment_score + 40) / 80 * 40 + volume_score + detail_score + diversity_score
    score = min(100, max(0, round(raw)))

    return {
        "score": score,
        "grade": (
            "A" if score >= 85 else
            "B" if score >= 70 else
            "C" if score >= 55 else
            "D" if score >= 40 else
            "F"
        ),
        "label": (
            "Excellent"    if score >= 85 else
            "Good"         if score >= 70 else
            "Average"      if score >= 55 else
            "Below Average" if score >= 40 else
            "Poor"
        ),
        "components": {
            "sentiment":  round(max(0, sentiment_score + 40) / 80 * 40, 1),
            "volume":     round(volume_score, 1),
            "detail":     round(detail_score, 1),
            "diversity":  round(diversity_score, 1),
        }
    }


def detect_red_flags(reviews_df) -> list:
    """Basic heuristics for suspicious review patterns."""
    flags = []

    # Flag 1: Very short reviews dominate
    texts = reviews_df["comments"].dropna().tolist()
    avg_len = sum(len(t.split()) for t in texts) / max(1, len(texts))
    if avg_len < 5:
        flags.append({
            "type":     "Low Detail",
            "severity": "warning",
            "message":  f"Average review is only {avg_len:.0f} words — unusually short",
        })

    # Flag 2: Many reviews in a single month
    if "date" in reviews_df.columns:
        try:
            reviews_df["month"] = reviews_df["date"].dt.to_period("M")
            monthly = reviews_df.groupby("month").size()
            max_monthly = int(monthly.max())
            if max_monthly > 15:
                flags.append({
                    "type":     "Review Spike",
                    "severity": "warning",
                    "message":  f"{max_monthly} reviews in a single month — may indicate incentivized reviews",
                })
        except Exception:
            pass

    # Flag 3: Negative reviews present
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    negative = [
        t for t in texts[:100]
        if isinstance(t, str) and sia.polarity_scores(t[:500])["compound"] < -0.3
    ]
    if len(negative) > 3:
        flags.append({
            "type":     "Critical Reviews",
            "severity": "warning",
            "message":  f"{len(negative)} reviews with strong negative sentiment detected",
        })

    if not flags:
        flags.append({
            "type":     "No Issues Found",
            "severity": "ok",
            "message":  "No suspicious review patterns detected",
        })

    return flags


def make_timeline(reviews_df, bins: int = 8) -> list:
    """Build a review count timeline for charting."""
    if "date" not in reviews_df.columns or len(reviews_df) == 0:
        return []

    try:
        reviews_df = reviews_df.copy()
        reviews_df["year_month"] = reviews_df["date"].dt.to_period("M")
        monthly = reviews_df.groupby("year_month").size().reset_index(name="count")
        monthly["label"] = monthly["year_month"].astype(str)

        # Return last `bins` months for the chart
        recent = monthly.tail(bins)
        return [
            {"month": row["label"], "reviews": int(row["count"])}
            for _, row in recent.iterrows()
        ]
    except Exception:
        return []


def main():
    # ── Read input from stdin ──────────────────────────────────────────────
    try:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    except Exception as e:
        print(json.dumps({"error": f"Invalid input JSON: {e}"}))
        sys.exit(1)

    listing_id  = int(payload.get("listing_id", 0))
    max_reviews = int(payload.get("max_reviews", 300))

    if listing_id == 0:
        print(json.dumps({"error": "listing_id is required"}))
        sys.exit(0)

    # ── Load reviews ───────────────────────────────────────────────────────
    reviews_df, error = load_reviews(listing_id, max_reviews)
    if error:
        print(json.dumps({"error": error, "listing_id": listing_id}))
        sys.exit(0)

    texts     = reviews_df["comments"].dropna().tolist()
    total_rev = len(reviews_df)
    avg_len   = sum(len(t.split()) for t in texts) / max(1, len(texts))

    # ── Analysis ──────────────────────────────────────────────────────────
    sentiment = analyze_sentiment(texts)
    themes    = detect_themes(texts)
    quality   = compute_quality_score(sentiment, themes, total_rev, avg_len)
    flags     = detect_red_flags(reviews_df)
    timeline  = make_timeline(reviews_df)

    # ── Sample reviews (most recent 5) ────────────────────────────────────
    samples = []
    for _, row in reviews_df.head(5).iterrows():
        text = str(row.get("comments", ""))
        if text and len(text) > 5:
            samples.append({
                "date":    str(row.get("date", ""))[:10],
                "text":    text[:400] + ("…" if len(text) > 400 else ""),
            })

    return {
        "listing_id":             listing_id,
        "total_reviews":          total_rev,
        "avg_review_length_words": round(avg_len, 1),
        "sentiment":              sentiment,
        "quality_score":          quality,
        "themes":                 themes,
        "red_flags":              flags,
        "timeline":               timeline,
        "sample_reviews":         samples,
    }


# ── Callable API (used by Flask server.py) ───────────────────────────────────
def run_analysis(listing_id: int, max_reviews: int = 300) -> dict:
    """
    Run review analysis and return the result dict directly.
    Used by server.py (Flask) to avoid subprocess overhead.
    """
    reviews_df, error = load_reviews(listing_id, max_reviews)
    if error:
        return {"error": error, "listing_id": listing_id}

    texts     = reviews_df["comments"].dropna().tolist()
    total_rev = len(reviews_df)
    avg_len   = sum(len(t.split()) for t in texts) / max(1, len(texts))

    sentiment = analyze_sentiment(texts)
    themes    = detect_themes(texts)
    quality   = compute_quality_score(sentiment, themes, total_rev, avg_len)
    flags     = detect_red_flags(reviews_df)
    timeline  = make_timeline(reviews_df)

    samples = []
    for _, row in reviews_df.head(5).iterrows():
        text = str(row.get("comments", ""))
        if text and len(text) > 5:
            samples.append({
                "date": str(row.get("date", ""))[:10],
                "text": text[:400] + ("…" if len(text) > 400 else ""),
            })

    return {
        "listing_id":              listing_id,
        "total_reviews":           total_rev,
        "avg_review_length_words": round(avg_len, 1),
        "sentiment":               sentiment,
        "quality_score":           quality,
        "themes":                  themes,
        "red_flags":               flags,
        "timeline":                timeline,
        "sample_reviews":          samples,
    }


# ── Main (stdin/stdout for local subprocess use) ─────────────────────────────
if __name__ == "__main__":
    main()
