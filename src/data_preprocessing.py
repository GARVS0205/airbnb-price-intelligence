"""
data_preprocessing.py
--------------------
Reusable data cleaning and preprocessing functions for the Airbnb Price
Intelligence project. These functions encapsulate every cleaning step
documented in 02_data_cleaning.ipynb.
"""

import numpy as np
import pandas as pd
import re
import ast
import json
from typing import Tuple, Dict, List, Optional


def load_raw_data(listings_path: str, reviews_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load raw Airbnb CSV files (supports .gz compression automatically).

    Args:
        listings_path: Path to listings.csv or listings.csv.gz
        reviews_path:  Optional path to reviews.csv or reviews.csv.gz

    Returns:
        Tuple of (listings_df, reviews_df or None)
    """
    listings_df = pd.read_csv(listings_path, low_memory=False)
    print(f"[load] Listings: {listings_df.shape}")

    reviews_df = None
    if reviews_path:
        reviews_df = pd.read_csv(reviews_path, low_memory=False)
        print(f"[load] Reviews:  {reviews_df.shape}")

    return listings_df, reviews_df


def clean_price(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Convert price from string (e.g. '$1,200.00') to float.
    Drops rows where price is NaN or zero after conversion.

    Args:
        df:        Input DataFrame
        price_col: Name of the price column

    Returns:
        DataFrame with price as float, invalid rows removed
    """
    df = df.copy()
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )
    n_before = len(df)
    df = df[df[price_col].notna() & (df[price_col] > 0)]
    print(f"[clean_price] Removed {n_before - len(df):,} rows with null/zero price")
    return df


def remove_duplicates(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """Remove duplicate listing rows by ID."""
    n_before = len(df)
    df = df.drop_duplicates(subset=id_col, keep="first")
    print(f"[remove_duplicates] Removed {n_before - len(df):,} duplicates")
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    col: str = "price",
    domain_lower: float = 10.0,
    domain_upper: float = 1000.0,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers using IQR method with optional domain knowledge bounds.

    Strategy:
      1. Compute Q1, Q3, IQR from the data
      2. Compute IQR bounds (Q1 - k*IQR, Q3 + k*IQR)
      3. Apply the more conservative of IQR bounds and domain bounds

    Args:
        df:              Input DataFrame
        col:             Column to filter on
        domain_lower:    Hard minimum (domain knowledge)
        domain_upper:    Hard maximum (domain knowledge)
        iqr_multiplier:  k in Q ± k*IQR (default=1.5, use 3.0 for lighter trimming)
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    iqr_lower = Q1 - iqr_multiplier * IQR
    iqr_upper = Q3 + iqr_multiplier * IQR

    final_lower = max(iqr_lower, domain_lower)
    final_upper = min(iqr_upper * 1.5, domain_upper)

    n_before = len(df)
    df = df[(df[col] >= final_lower) & (df[col] <= final_upper)]
    print(f"[remove_outliers_iqr] {col}: kept ${final_lower:.0f}–${final_upper:.0f}, "
          f"removed {n_before - len(df):,} rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply context-aware missing value imputation.

    Strategy:
        - bedrooms/beds  → median per room_type
        - bathrooms      → 1.0 (most common)
        - review_scores  → global median
        - reviews_per_month → 0.0
        - host_response/acceptance_rate → median
        - host_is_superhost → False
    """
    df = df.copy()

    # Bedrooms: median per room_type
    for col in ["bedrooms", "beds"]:
        if col in df.columns:
            df[col] = df.groupby("room_type")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # Bathrooms
    if "bathrooms" in df.columns:
        df["bathrooms"] = df["bathrooms"].fillna(1.0)

    # Review scores
    for col in [c for c in df.columns if "review_scores" in c]:
        df[col] = df[col].fillna(df[col].median())

    # Reviews per month
    if "reviews_per_month" in df.columns:
        df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)

    # Host rates
    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Superhost
    if "host_is_superhost" in df.columns:
        df["host_is_superhost"] = df["host_is_superhost"].fillna(False)

    print(f"[handle_missing] Imputation complete. Remaining NaNs: {df.isnull().sum().sum()}")
    return df


def encode_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Airbnb 't'/'f' boolean columns to Python True/False."""
    bool_cols = [
        "host_is_superhost", "host_has_profile_pic", "host_identity_verified",
        "has_availability", "instant_bookable",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"t": True, "f": False, True: True, False: False})
    return df


def parse_amenities(amenity_str) -> List[str]:
    """
    Parse the amenities column from its JSON-like string format.

    Examples:
        '["Wifi", "Kitchen"]'  → ['wifi', 'kitchen']
        '[TV, Heating, ...]'   → ['tv', 'heating', ...]
    """
    if pd.isna(amenity_str) or str(amenity_str).strip() == "":
        return []
    try:
        result = ast.literal_eval(str(amenity_str))
        if isinstance(result, list):
            return [a.strip().lower() for a in result if isinstance(a, str)]
        return []
    except Exception:
        clean = re.sub(r'[\[\]"\\]', "", str(amenity_str))
        return [a.strip().lower() for a in clean.split(",") if a.strip()]


def parse_bathrooms(text) -> float:
    """
    Parse bathroom count from strings like '1.5 baths' or 'Shared half-bath'.

    Returns:
        Float number of bathrooms (0.5 for half-bath, 1.0 default if unparseable)
    """
    if pd.isna(text):
        return np.nan
    text_str = str(text).lower()
    if "half" in text_str:
        return 0.5
    numbers = re.findall(r"\d+\.?\d*", text_str)
    return float(numbers[0]) if numbers else np.nan


def parse_rate(rate_str) -> float:
    """Convert '97%' to 0.97."""
    if pd.isna(rate_str):
        return np.nan
    cleaned = str(rate_str).replace("%", "").strip()
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return np.nan


def drop_high_missingness_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns where more than `threshold` fraction of values are missing."""
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    print(f"[drop_high_missingness] Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    return df.drop(columns=drop_cols, errors="ignore")


def full_cleaning_pipeline(
    listings_path: str,
    output_path: Optional[str] = None,
    price_lower: float = 10.0,
    price_upper: float = 1000.0,
) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline end-to-end.

    Args:
        listings_path: Path to raw listings CSV/gz file
        output_path:   If provided, save cleaned CSV here
        price_lower:   Minimum valid price
        price_upper:   Maximum valid price

    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    df, _ = load_raw_data(listings_path)
    df = clean_price(df)
    df = remove_duplicates(df)
    df = remove_outliers_iqr(df, domain_lower=price_lower, domain_upper=price_upper)
    df = drop_high_missingness_columns(df, threshold=0.5)
    df = encode_booleans(df)

    # Parse complex columns
    if "bathrooms_text" in df.columns:
        df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)
    for rate_col in ["host_response_rate", "host_acceptance_rate"]:
        if rate_col in df.columns:
            df[rate_col] = df[rate_col].apply(parse_rate)
    if "amenities" in df.columns:
        df["amenities_parsed"] = df["amenities"].apply(parse_amenities)
        df["amenity_count"] = df["amenities_parsed"].apply(len)

    df = handle_missing(df)

    if output_path:
        # Save with amenities as JSON string
        df_save = df.copy()
        if "amenities_parsed" in df_save.columns:
            df_save["amenities_json"] = df_save["amenities_parsed"].apply(json.dumps)
            df_save = df_save.drop(columns=["amenities_parsed"])
        df_save.to_csv(output_path, index=False)
        print(f"[pipeline] Saved cleaned data → {output_path}")

    print(f"[pipeline] Final shape: {df.shape}")
    return df
