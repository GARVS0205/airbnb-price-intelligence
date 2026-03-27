#!/usr/bin/env python3
import os
import re
import sqlite3
import pandas as pd
import json
import warnings

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# Import analysis functions from the existing API script
from review_analysis_api import (
    analyze_sentiment,
    detect_themes,
    compute_quality_score,
    detect_red_flags,
    make_timeline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "reviews.csv")
DB_PATH = os.path.join(BASE_DIR, "models", "reviews_summary.db")

def compute_predict_features(rev_df, sia):
    """Replicates the logic from predict_api.py compute_review_nlp_features"""
    rev_df = rev_df.dropna(subset=["comments"])
    rev_df = rev_df[rev_df["comments"].str.strip().str.len() > 5].copy()
    
    if len(rev_df) == 0:
        return {}
        
    rev_df["compound"] = rev_df["comments"].apply(
        lambda t: sia.polarity_scores(str(t)[:1000])["compound"]
    )
    rev_df["word_count"]  = rev_df["comments"].apply(lambda t: len(str(t).split()))
    rev_df["is_positive"] = (rev_df["compound"] >= 0.05).astype(int)
    rev_df["is_negative"] = (rev_df["compound"] <= -0.05).astype(int)

    avg_compound  = float(rev_df["compound"].mean())
    positive_pct  = float(rev_df["is_positive"].mean() * 100)
    negative_pct  = float(rev_df["is_negative"].mean() * 100)
    avg_words     = float(rev_df["word_count"].mean())
    n_reviews     = len(rev_df)

    quality_score = float(((avg_compound + 1) / 2 * 40) + min(20, n_reviews / 5) + min(20, avg_words / 15))
    quality_score = max(0.0, min(80.0, quality_score))

    trend = 0.0
    if "date" in rev_df.columns:
        max_date = rev_df["date"].max()
        cutoff   = max_date - pd.Timedelta(days=180)
        recent   = rev_df[rev_df["date"] >= cutoff]
        recent_avg = float(recent["compound"].mean()) if len(recent) > 0 else avg_compound
        trend = recent_avg - avg_compound

    return {
        "review_avg_sentiment":    round(avg_compound, 4),
        "review_positive_pct":     round(positive_pct, 1),
        "review_negative_pct":     round(negative_pct, 1),
        "review_avg_length":       round(avg_words, 1),
        "review_quality_score":    round(quality_score, 1),
        "composite_review_score":  round(trend, 4),
    }

def compute_analysis_features(rev_df, listing_id):
    """Replicates the logic from review_analysis_api.py run_analysis"""
    texts = rev_df["comments"].dropna().tolist()
    total_rev = len(rev_df)
    if total_rev == 0:
        return {}
        
    avg_len = sum(len(str(t).split()) for t in texts) / max(1, len(texts))

    sentiment = analyze_sentiment(texts)
    themes    = detect_themes(texts)
    quality   = compute_quality_score(sentiment, themes, total_rev, avg_len)
    flags     = detect_red_flags(rev_df)
    timeline  = make_timeline(rev_df)

    samples = []
    for _, row in rev_df.head(5).iterrows():
        raw_text = str(row.get("comments", ""))
        # Strip HTML tags (e.g. <br/>, <br>, &amp;, etc.) baked into raw CSV
        clean_text = re.sub(r"<[^>]+>", " ", raw_text)          # remove tags
        clean_text = re.sub(r"&[a-z]+;", " ", clean_text)       # remove HTML entities
        clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()  # collapse whitespace
        if clean_text and len(clean_text) > 5:
            samples.append({
                "date": str(row.get("date", ""))[:10],
                "text": clean_text[:400] + ("…" if len(clean_text) > 400 else ""),
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

def main():
    print("Loading reviews.csv into memory...")
    try:
        df = pd.read_csv(
            REVIEWS_PATH,
            usecols=["listing_id", "id", "date", "comments"],
            dtype={"listing_id": int, "id": int, "comments": str},
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Connecting to SQLite database...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews_summary (
            listing_id INTEGER PRIMARY KEY,
            predict_features TEXT,
            analysis_data TEXT
        )
    ''')
    # Clear existing data just in case
    cursor.execute('DELETE FROM reviews_summary')
    
    sia = SentimentIntensityAnalyzer()
    
    # Sort values so that when we take head() we get the most recent
    df = df.sort_values(by=["listing_id", "date"], ascending=[True, False])
    
    grouped = df.groupby("listing_id")
    listings = list(grouped.groups.keys())
    
    print(f"Processing {len(listings)} listings...")
    
    batch_size = 500
    batch_data = []
    
    for i, listing_id in enumerate(listings):
        if i % 1000 == 0:
            print(f"Processing listing {i}/{len(listings)} ({(i/len(listings))*100:.1f}%)")
            
        # max 300 reviews per listing for performance/consistency with current API
        rev_df = grouped.get_group(listing_id).head(300)
        
        predict_features = compute_predict_features(rev_df, sia)
        analysis_data = compute_analysis_features(rev_df, listing_id)
        
        if predict_features or analysis_data:
            batch_data.append((
                int(listing_id),
                json.dumps(predict_features) if predict_features else "{}",
                json.dumps(analysis_data) if analysis_data else "{}"
            ))
            
        if len(batch_data) >= batch_size:
            cursor.executemany(
                "INSERT INTO reviews_summary (listing_id, predict_features, analysis_data) VALUES (?, ?, ?)",
                batch_data
            )
            batch_data = []
            
    if batch_data:
        cursor.executemany(
            "INSERT INTO reviews_summary (listing_id, predict_features, analysis_data) VALUES (?, ?, ?)",
            batch_data
        )
        
    conn.commit()
    conn.close()
    print("Done! SQLite database saved at:", DB_PATH)

if __name__ == "__main__":
    main()
