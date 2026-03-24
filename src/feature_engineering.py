"""
feature_engineering.py
-----------------------
Reusable feature engineering functions for the Airbnb Price Intelligence
project. Mirrors the logic in 03_feature_engineering.ipynb.
"""

import numpy as np
import pandas as pd
import json
import re
from math import radians, cos, sin, asin, sqrt
from typing import Dict, List, Optional

# ── NYC Landmark coordinates ────────────────────────────────────────────────
NYC_LANDMARKS: Dict[str, tuple] = {
    "times_square":    (40.7580, -73.9855),
    "central_park":    (40.7851, -73.9683),
    "jfk_airport":     (40.6413, -73.7781),
    "brooklyn_bridge": (40.7061, -73.9969),
    "grand_central":   (40.7527, -73.9772),
    "statue_liberty":  (40.6892, -74.0445),
    "columbia_univ":   (40.8075, -73.9626),
    "union_square":    (40.7359, -73.9906),
}

# ── Amenity keyword groups ──────────────────────────────────────────────────
PREMIUM_AMENITY_GROUPS: Dict[str, List[str]] = {
    "has_pool":       ["pool", "swimming pool"],
    "has_gym":        ["gym", "fitness", "exercise equipment"],
    "has_parking":    ["parking", "free parking", "private parking"],
    "has_doorman":    ["doorman", "security", "concierge"],
    "has_elevator":   ["elevator", "lift"],
    "has_washer":     ["washer", "washing machine"],
    "has_dishwasher": ["dishwasher"],
    "has_ac":         ["air conditioning", "central air", "a/c"],
    "has_workspace":  ["dedicated workspace", "desk", "laptop friendly"],
}

# ── Text keyword groups ─────────────────────────────────────────────────────
TEXT_KEYWORD_GROUPS: Dict[str, List[str]] = {
    "has_luxury_keywords":    ["luxury", "luxurious", "premium", "penthouse", "exclusive"],
    "has_cozy_keywords":      ["cozy", "cosy", "charming", "intimate"],
    "has_spacious_keywords":  ["spacious", "large", "huge", "roomy", "expansive"],
    "has_renovated_keywords": ["renovated", "remodeled", "modern", "updated", "brand-new"],
}


# ── Location Features ────────────────────────────────────────────────────────

def _haversine_vectorized(
    lat_arr: np.ndarray, lon_arr: np.ndarray, lm_lat: float, lm_lon: float
) -> np.ndarray:
    """Vectorized Haversine distance calculation (km)."""
    R = 6371.0
    lat1 = np.radians(lat_arr)
    lat2 = np.radians(lm_lat)
    dlon = np.radians(lon_arr - lm_lon)
    dlat = lat1 - lat2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_location_features(
    df: pd.DataFrame,
    landmarks: Optional[Dict[str, tuple]] = None,
) -> pd.DataFrame:
    """
    Add distance-to-landmark features using Haversine formula.

    Args:
        df:        DataFrame with 'latitude' and 'longitude' columns
        landmarks: Dict of {name: (lat, lon)}. Defaults to NYC_LANDMARKS.
    """
    if landmarks is None:
        landmarks = NYC_LANDMARKS
    df = df.copy()
    lat = df["latitude"].values
    lon = df["longitude"].values
    for name, (lm_lat, lm_lon) in landmarks.items():
        df[f"dist_{name}_km"] = _haversine_vectorized(lat, lon, lm_lat, lm_lon)
    return df


def add_geo_clusters(df: pd.DataFrame, kmeans_model) -> pd.DataFrame:
    """Apply pre-fitted K-Means model to add geo_cluster column."""
    df = df.copy()
    coords = df[["latitude", "longitude"]].values
    df["geo_cluster"] = kmeans_model.predict(coords)
    return df


# ── Text Features ────────────────────────────────────────────────────────────

def add_text_features(df: pd.DataFrame, use_vader: bool = True) -> pd.DataFrame:
    """
    Add NLP features from listing name and description.

    Features added:
        - desc_word_count, desc_char_count, name_word_count
        - has_luxury_keywords, has_cozy_keywords, etc.
        - desc_sentiment, name_sentiment (VADER compound score, if available)
    """
    df = df.copy()
    df["description"] = df["description"].fillna("")
    df["name"] = df["name"].fillna("")

    df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)
    df["desc_char_count"] = df["description"].str.len().fillna(0)
    df["name_word_count"] = df["name"].str.split().str.len().fillna(0)

    combined = df["description"].fillna("") + " " + df["name"].fillna("")

    for flag, keywords in TEXT_KEYWORD_GROUPS.items():
        df[flag] = combined.apply(
            lambda t: int(any(kw in t.lower() for kw in keywords))
        )

    df["desc_sentiment"] = 0.0
    df["name_sentiment"] = 0.0

    if use_vader:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            df["desc_sentiment"] = df["description"].apply(
                lambda t: sia.polarity_scores(t[:2000])["compound"] if len(t) > 5 else 0.0
            )
            df["name_sentiment"] = df["name"].apply(
                lambda t: sia.polarity_scores(t)["compound"] if len(t) > 5 else 0.0
            )
        except ImportError:
            pass  # VADER not installed — sentiment stays 0

    return df


# ── Amenity Features ─────────────────────────────────────────────────────────

def _check_amenity(amenity_list: List[str], keywords: List[str]) -> int:
    joined = " ".join(amenity_list)
    return int(any(kw in joined for kw in keywords))


def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add premium amenity binary flags and premium_amenity_score.

    Expects either:
        - 'amenities_parsed': Python list of lowercase strings, or
        - 'amenities_json':   JSON string of the list
    """
    df = df.copy()

    if "amenities_parsed" not in df.columns:
        if "amenities_json" in df.columns:
            df["amenities_parsed"] = df["amenities_json"].apply(
                lambda x: json.loads(x) if pd.notna(x) else []
            )
        else:
            df["amenities_parsed"] = [[] for _ in range(len(df))]

    for col_name, keywords in PREMIUM_AMENITY_GROUPS.items():
        df[col_name] = df["amenities_parsed"].apply(lambda lst: _check_amenity(lst, keywords))

    df["premium_amenity_score"] = df[list(PREMIUM_AMENITY_GROUPS.keys())].sum(axis=1)
    return df


# ── Host Features ─────────────────────────────────────────────────────────────

def add_host_features(
    df: pd.DataFrame,
    reference_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Add host quality and experience features.

    Features added:
        - is_superhost (int)
        - host_experience_days, host_experience_years
        - is_professional_host (has 5+ listings)
        - host_quality_score (composite)
    """
    df = df.copy()
    ref = pd.Timestamp(reference_date)

    df["is_superhost"] = df["host_is_superhost"].fillna(False).astype(int)

    if "host_since" in df.columns and "host_experience_days" not in df.columns:
        df["host_since_dt"] = pd.to_datetime(df["host_since"], errors="coerce")
        df["host_experience_days"] = (ref - df["host_since_dt"]).dt.days
        df["host_experience_days"] = df["host_experience_days"].fillna(
            df["host_experience_days"].median()
        ).clip(lower=0)

    if "host_experience_days" in df.columns:
        df["host_experience_years"] = df["host_experience_days"] / 365.25

    if "calculated_host_listings_count" in df.columns:
        df["is_professional_host"] = (
            df["calculated_host_listings_count"] >= 5
        ).astype(int)

    # Composite host quality score
    components = ["is_superhost"]
    if "host_response_rate" in df.columns:
        components.append("host_response_rate")
    if "host_acceptance_rate" in df.columns:
        components.append("host_acceptance_rate")
    df["host_quality_score"] = df[components].mean(axis=1)

    return df


# ── Review Features ──────────────────────────────────────────────────────────

def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add review-based features.

    Features added:
        - log_number_of_reviews, has_reviews
        - composite_review_score
        - review_recency_bucket
    """
    df = df.copy()

    df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"].fillna(0))
    df["has_reviews"] = (df["number_of_reviews"].fillna(0) > 0).astype(int)

    subcols = [c for c in df.columns if "review_scores_" in c and "rating" not in c]
    if subcols:
        df["composite_review_score"] = df[subcols].mean(axis=1)
    elif "review_scores_rating" in df.columns:
        df["composite_review_score"] = df["review_scores_rating"]
    else:
        df["composite_review_score"] = 0.0

    if "review_recency_days" in df.columns:
        df["review_recency_bucket"] = pd.cut(
            df["review_recency_days"],
            bins=[-1, 30, 90, 180, 365, np.inf],
            labels=[4, 3, 2, 1, 0],
        ).astype(int)

    return df


# ── Property Features ────────────────────────────────────────────────────────

def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add property-level engineered features."""
    df = df.copy()

    if "availability_365" in df.columns:
        df["availability_rate"] = df["availability_365"] / 365.0

    if "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bedrooms_per_bathroom"] = df["bedrooms"] / (df["bathrooms"] + 0.5)

    df["accommodates_sq"] = df["accommodates"] ** 2

    if "beds" in df.columns:
        df["beds_per_person"] = df["beds"] / df["accommodates"].clip(lower=1)

    return df


# ── Interaction Features ─────────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cross-feature interaction terms."""
    df = df.copy()

    if "log_number_of_reviews" in df.columns and "composite_review_score" in df.columns:
        df["reviews_x_score"] = (
            df["log_number_of_reviews"] * df["composite_review_score"].fillna(0)
        )

    if "bedrooms" in df.columns:
        df["capacity_x_bedrooms"] = df["accommodates"] * df["bedrooms"].fillna(0)

    if "is_superhost" in df.columns and "composite_review_score" in df.columns:
        df["superhost_x_score"] = df["is_superhost"] * df["composite_review_score"].fillna(0)

    if "premium_amenity_score" in df.columns:
        df["luxury_x_capacity"] = df["premium_amenity_score"] * df["accommodates"]

    return df


# ── Categorical Encoding ─────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    neighbourhood_mean_prices: Optional[Dict[str, float]] = None,
    global_mean_log_price: float = 5.0,
    fit: bool = True,
) -> pd.DataFrame:
    """
    Encode categorical features:
        - room_type → LabelEncoder (0-3)
        - borough   → LabelEncoder (0-4)
        - neighbourhood → target encoding (mean log_price per neighbourhood)

    Args:
        df:                        Input DataFrame
        neighbourhood_mean_prices: Pre-computed mapping (use at inference time)
        global_mean_log_price:     Fallback for unseen neighbourhoods
        fit:                       If True, compute encoding from df (training).
                                   If False, use provided neighbourhood_mean_prices.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    # Room type
    le_room = LabelEncoder()
    df["room_type_encoded"] = le_room.fit_transform(df["room_type"].fillna("Unknown"))

    # Borough
    le_borough = LabelEncoder()
    df["borough_encoded"] = le_borough.fit_transform(
        df["neighbourhood_group_cleansed"].fillna("Unknown")
    )

    # Neighbourhood target encoding
    if fit and "log_price" in df.columns:
        nbhd_means = df.groupby("neighbourhood_cleansed")["log_price"].mean().to_dict()
    elif neighbourhood_mean_prices is not None:
        nbhd_means = neighbourhood_mean_prices
    else:
        nbhd_means = {}

    df["neighbourhood_target_encoded"] = df["neighbourhood_cleansed"].map(nbhd_means)
    df["neighbourhood_target_encoded"] = df["neighbourhood_target_encoded"].fillna(
        global_mean_log_price
    )

    return df, {
        "room_type_encoder": le_room,
        "borough_encoder": le_borough,
        "neighbourhood_mean_prices": nbhd_means,
        "global_mean_log_price": global_mean_log_price,
    }


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_feature_engineering(
    df: pd.DataFrame,
    kmeans_model=None,
    neighbourhood_mean_prices: Optional[Dict] = None,
    global_mean_log_price: float = 5.0,
) -> pd.DataFrame:
    """
    Run all feature engineering steps in sequence.

    Args:
        df:                        Cleaned listings DataFrame
        kmeans_model:              Pre-fitted KMeans (loaded from models/)
        neighbourhood_mean_prices: Pre-computed target encoding (at inference time)
        global_mean_log_price:     Fallback for target encoding

    Returns:
        Feature-enriched DataFrame
    """
    df = add_location_features(df)
    if kmeans_model is not None:
        df = add_geo_clusters(df, kmeans_model)
    df = add_text_features(df)
    df = add_amenity_features(df)
    df = add_host_features(df)
    df = add_review_features(df)
    df = add_property_features(df)
    df = add_interaction_features(df)

    fit_encoding = neighbourhood_mean_prices is None
    df, _ = encode_categoricals(
        df,
        neighbourhood_mean_prices=neighbourhood_mean_prices,
        global_mean_log_price=global_mean_log_price,
        fit=fit_encoding,
    )

    return df
