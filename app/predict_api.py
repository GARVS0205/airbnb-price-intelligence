#!/usr/bin/env python3
"""
predict_api.py
--------------
Python inference script called by the Next.js API route via child_process.

Architecture (Phase 1 + Phase 2 INTEGRATED):
  - The XGBoost model was trained with 6 review NLP features as part of the
    feature matrix (review_avg_sentiment, review_positive_pct, etc.).
  - At inference time, if a listing_id is provided, we read reviews.csv
    and compute the same VADER features — they go straight into the feature
    vector that the model consumes.
  - If no listing_id is given (new listing / manual entry), those 6 features
    default to neutral values (same defaults used in training).
  - There is NO post-prediction adjustment. The model itself has learned
    how review sentiment correlates with price.

Usage:
    echo '{"neighbourhood_target_encoded":5.2,"accommodates":4,...}' | python predict_api.py
    echo '{"listing_id":2539,"neighbourhood_target_encoded":5.2,...}' | python predict_api.py

Called from:
    app/app/api/predict/route.ts
"""

import sys
import json
import os
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))

# Resolve models directory: works both locally (../models) and on Vercel (./models)
_models_parent = os.path.join(SCRIPT_DIR, "..", "models")
_models_local  = os.path.join(SCRIPT_DIR, "models")
MODELS_DIR   = _models_parent if os.path.isdir(_models_parent) else _models_local

# Resolve data directory: reviews.csv only used locally (too large for Vercel)
_data_parent  = os.path.join(SCRIPT_DIR, "..", "data", "raw")
_data_local   = os.path.join(SCRIPT_DIR, "data", "raw")
DATA_DIR      = _data_parent if os.path.isdir(_data_parent) else _data_local
REVIEWS_PATH = os.path.join(DATA_DIR, "reviews.csv")

# ── NLP feature defaults (matches training-time defaults in run_pipeline.py) ──
NLP_DEFAULTS = {
    "review_avg_sentiment":    0.0,
    "review_positive_pct":     50.0,
    "review_negative_pct":     0.0,
    "review_avg_word_count":   30.0,
    "review_quality_score":    50.0,
    "review_sentiment_trend":  0.0,
}

# ── Input validation ────────────────────────────────────────────────────────
VALIDATION_RULES = {
    "accommodates":               (1, 16),
    "bedrooms":                   (0, 20),
    "bathrooms":                  (0, 20),
    "beds":                       (0, 30),
    "minimum_nights":             (1, 365),
    "availability_365":           (0, 365),
    "number_of_reviews":          (0, 10000),
    "review_scores_rating":       (0, 5),
    "review_scores_cleanliness":  (0, 5),
    "review_scores_accuracy":     (0, 5),
    "host_response_rate":         (0, 1),
    "host_acceptance_rate":       (0, 1),
    "premium_amenity_score":      (0, 9),
}


def validate_inputs(payload: dict) -> list:
    errors = []
    for field, (lo, hi) in VALIDATION_RULES.items():
        val = payload.get(field)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number, got: {val!r}")
            continue
        if not (lo <= v <= hi):
            errors.append(f"'{field}' must be between {lo} and {hi}, got: {v}")
    return errors


# ── Review NLP feature computation ─────────────────────────────────────────
def compute_review_nlp_features(listing_id: int) -> dict:
    """
    Read reviews.csv for the given listing_id and compute the same 6 NLP
    features that were computed during training in run_pipeline.py.

    Returns NLP_DEFAULTS on any failure (missing file, no reviews, import error).
    """
    features = dict(NLP_DEFAULTS)  # start with defaults

    try:
        import sqlite3
        import json
        
        db_path = os.path.join(BASE_DIR, "models", "reviews_summary.db")
        if not os.path.exists(db_path):
            return features

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT predict_features FROM reviews_summary WHERE listing_id = ?", (listing_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row[0] and row[0] != "{}":
            data = json.loads(row[0])
            features.update({
                "review_avg_sentiment":    data.get("review_avg_sentiment", 0.0),
                "review_positive_pct":     data.get("review_positive_pct", 0.0),
                "review_negative_pct":     data.get("review_negative_pct", 0.0),
                "review_avg_word_count":   data.get("review_avg_length", 0.0),
                "review_quality_score":    data.get("review_quality_score", 0.0),
                "review_sentiment_trend":  data.get("composite_review_score", 0.0),
            })
            
    except Exception as e:
        print(f"Error fetching review NLP features from DB: {e}")


    except Exception:
        pass  # silently fall back to defaults

    return features


# ── Model loading ───────────────────────────────────────────────────────────
def load_artifacts():
    try:
        import joblib
        model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
            feat_meta = json.load(f)
        return model, scaler, feat_meta["feature_names"], None
    except Exception as e:
        return None, None, None, str(e)

# ── Callable API (used by Flask server.py) ──────────────────────────────────
def run_inference(payload: dict) -> dict:
    """
    Run prediction on a feature dict. Returns the output dict directly.
    Used by server.py (Flask) to avoid subprocess overhead on Vercel alternatives.
    """
    import numpy as np

    errors = validate_inputs(payload)
    if errors:
        return {"error": "Input validation failed", "details": errors}

    model, scaler, feature_names, load_err = load_artifacts()
    if load_err:
        return {"error": f"Model loading failed: {load_err}"}

    listing_id = payload.get("listing_id")
    nlp_features = {}
    nlp_used = False

    if listing_id:
        nlp_features = compute_review_nlp_features(int(listing_id))
        nlp_used = nlp_features.get("review_avg_sentiment", 0.0) != 0.0 \
                or nlp_features.get("review_quality_score", 50.0) != 50.0

    merged = {**payload, **nlp_features}
    x = np.array(
        [float(merged.get(f, 0.0)) for f in feature_names], dtype=np.float32
    ).reshape(1, -1)

    x_scaled   = scaler.transform(x)
    log_pred   = float(model.predict(x_scaled)[0])
    price      = float(np.expm1(log_pred))
    price_low  = round(price * 0.82, 2)
    price_high = round(price * 1.18, 2)

    top_features = []
    if hasattr(model, "feature_importances_"):
        imp = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        top_features = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp]

    return {
        "predicted_price":     round(price, 2),
        "price_low":           price_low,
        "price_high":          price_high,
        "log_prediction":      round(log_pred, 4),
        "top_features":        top_features,
        "review_nlp_used":     nlp_used,
        "review_nlp_features": nlp_features if nlp_used else None,
    }


# ── Main (stdin/stdout for local subprocess use) ─────────────────────────────
def main():
    try:
        raw = sys.stdin.read().strip()
        if not raw:
            raise ValueError("Empty input")
        payload = json.loads(raw)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse input: {e}"}))
        sys.exit(1)

    result = run_inference(payload)
    print(json.dumps(result))
    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
