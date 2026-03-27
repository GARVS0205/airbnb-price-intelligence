#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Runs the complete Airbnb Price Intelligence ML pipeline:
  1. Data loading & cleaning
  2. Feature engineering
  3. Model training (7 algorithms) + hyperparameter tuning
  4. Saves model artifacts to models/

Auto-detects .csv vs .csv.gz files in data/raw/
"""

import os, sys, json, time, warnings, joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

STEP = 0
def log(msg):
    global STEP; STEP += 1
    print(f"[{STEP:02d}] {msg}", flush=True)

def find_file(name_stems):
    """Find a data file by trying multiple name patterns."""
    for stem in name_stems:
        for ext in [".csv", ".csv.gz"]:
            path = os.path.join(RAW_DIR, stem + ext)
            if os.path.exists(path):
                return path
    return None

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
log("Loading raw data files...")
listings_path = find_file(["listings"])
reviews_path  = find_file(["reviews"])

if not listings_path:
    print("ERROR: listings.csv (or listings.csv.gz) not found in data/raw/")
    sys.exit(1)

log(f"  listings -> {os.path.basename(listings_path)}")
df = pd.read_csv(listings_path, low_memory=False)
log(f"  Loaded {df.shape[0]:,} listings, {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# STEP 2: CLEAN PRICE
# ─────────────────────────────────────────────────────────────
log("Cleaning price column...")
df["price"] = (
    df["price"].astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
n_before = len(df)
df = df[df["price"].notna() & (df["price"] > 0)]
log(f"  Removed {n_before - len(df):,} rows with null/zero price. Remaining: {len(df):,}")

# Remove duplicates
df = df.drop_duplicates(subset="id", keep="first")

# Outlier removal: IQR + domain bounds
Q1, Q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
IQR = Q3 - Q1
price_upper = min(Q3 + 3 * IQR, 1000)
price_lower = max(Q1 - 1.5 * IQR, 10)
n_before = len(df)
df = df[(df["price"] >= price_lower) & (df["price"] <= price_upper)]
log(f"  Outlier removal: kept ${price_lower:.0f}–${price_upper:.0f}/night. Removed {n_before - len(df):,} rows.")

# Log-transform target
df["log_price"] = np.log1p(df["price"])

# ─────────────────────────────────────────────────────────────
# STEP 3: DROP HIGH-MISSINGNESS COLUMNS
# ─────────────────────────────────────────────────────────────
log("Dropping high-missingness columns (>50% missing)...")
miss_pct = df.isnull().mean()
drop_cols = miss_pct[miss_pct > 0.5].index.tolist()
df = df.drop(columns=drop_cols, errors="ignore")
log(f"  Dropped {len(drop_cols)} columns. Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4: PARSE COMPLEX COLUMNS
# ─────────────────────────────────────────────────────────────
log("Parsing complex columns...")

import re, ast

# Bathrooms
def parse_bathrooms(text):
    if pd.isna(text): return np.nan
    text = str(text).lower()
    if "half" in text: return 0.5
    nums = re.findall(r"\d+\.?\d*", text)
    return float(nums[0]) if nums else np.nan

if "bathrooms_text" in df.columns:
    df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)

# Boolean columns
for col in ["host_is_superhost", "host_has_profile_pic",
            "host_identity_verified", "instant_bookable"]:
    if col in df.columns:
        df[col] = df[col].map({"t": True, "f": False, True: True, False: False})

# Host rates
for col in ["host_response_rate", "host_acceptance_rate"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace("%", "").pipe(pd.to_numeric, errors="coerce") / 100

# Amenities
def parse_amenities(s):
    if pd.isna(s): return []
    try:
        r = ast.literal_eval(str(s))
        return [x.strip().lower() for x in r if isinstance(x, str)] if isinstance(r, list) else []
    except Exception:
        return [x.strip().lower() for x in re.sub(r'[\[\]"\\]', "", str(s)).split(",") if x.strip()]

if "amenities" in df.columns:
    df["amenities_parsed"] = df["amenities"].apply(parse_amenities)
    df["amenity_count"] = df["amenities_parsed"].apply(len)
else:
    df["amenities_parsed"] = [[] for _ in range(len(df))]
    df["amenity_count"] = 0

# Dates
ref = pd.Timestamp("2024-01-01")
for col, new_col in [("host_since", "host_experience_days"), ("last_review", "review_recency_days")]:
    if col in df.columns:
        df[new_col] = (ref - pd.to_datetime(df[col], errors="coerce")).dt.days.clip(lower=0)
        df[new_col] = df[new_col].fillna(df[new_col].median())

log("  Complex columns parsed.")

# ─────────────────────────────────────────────────────────────
# STEP 5: MISSING VALUE IMPUTATION
# ─────────────────────────────────────────────────────────────
log("Imputing missing values...")

for col in ["bedrooms", "beds"]:
    if col in df.columns:
        if "room_type" in df.columns:
            df[col] = df.groupby("room_type")[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

if "bathrooms" in df.columns:
    df["bathrooms"] = df["bathrooms"].fillna(1.0)

for col in [c for c in df.columns if "review_scores" in c]:
    df[col] = df[col].fillna(df[col].median())

if "reviews_per_month" in df.columns:
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
if "host_response_rate" in df.columns:
    df["host_response_rate"] = df["host_response_rate"].fillna(df["host_response_rate"].median())
if "host_acceptance_rate" in df.columns:
    df["host_acceptance_rate"] = df["host_acceptance_rate"].fillna(df["host_acceptance_rate"].median())
if "host_is_superhost" in df.columns:
    df["host_is_superhost"] = df["host_is_superhost"].fillna(False)

log(f"  Remaining NaNs: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 6: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
log("Engineering location features (Haversine distances)...")

LANDMARKS = {
    "times_square":    (40.7580, -73.9855),
    "central_park":    (40.7851, -73.9683),
    "jfk_airport":     (40.6413, -73.7781),
    "brooklyn_bridge": (40.7061, -73.9969),
    "grand_central":   (40.7527, -73.9772),
}

lat = df["latitude"].values
lon = df["longitude"].values

for name, (lm_lat, lm_lon) in LANDMARKS.items():
    R = 6371.0
    lat1 = np.radians(lat); lat2 = np.radians(lm_lat)
    dlon = np.radians(lon - lm_lon); dlat = lat1 - lat2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df[f"dist_{name}_km"] = 2 * R * np.arcsin(np.sqrt(a))

# Geo clustering
log("Geo-clustering (K-Means k=8)...")
from sklearn.cluster import KMeans
coords = df[["latitude", "longitude"]].fillna(0).values
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df["geo_cluster"] = kmeans.fit_predict(coords)
joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_geo.pkl"))

# Text features
log("Computing text features...")
df["description"] = df["description"].fillna("")
df["name"] = df["name"].fillna("") if "name" in df.columns else ""
df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)
df["name_word_count"] = df["name"].str.split().str.len().fillna(0) if "name" in df.columns else 0
combined = df["description"] + " " + (df["name"] if "name" in df.columns else "")
for flag, kws in [
    ("has_luxury_keywords",    ["luxury","luxurious","premium","penthouse"]),
    ("has_cozy_keywords",      ["cozy","cosy","charming","intimate"]),
    ("has_spacious_keywords",  ["spacious","large","huge","roomy"]),
    ("has_renovated_keywords", ["renovated","remodeled","modern","updated"]),
]:
    df[flag] = combined.apply(lambda t: int(any(k in t.lower() for k in kws)))

# VADER sentiment
df["desc_sentiment"] = 0.0
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    df["desc_sentiment"] = df["description"].apply(
        lambda t: sia.polarity_scores(t[:2000])["compound"] if len(t) > 5 else 0.0
    )
    log("  VADER sentiment computed.")
except Exception as e:
    log(f"  VADER skipped: {e}")

# Amenity flags
log("Engineering amenity features...")
AMENITY_FLAGS = {
    "has_pool":     ["pool","swimming pool"],
    "has_gym":      ["gym","fitness"],
    "has_parking":  ["parking"],
    "has_elevator": ["elevator","lift"],
    "has_washer":   ["washer","washing machine"],
    "has_ac":       ["air conditioning","a/c"],
    "has_workspace":["dedicated workspace","desk"],
    "has_doorman":  ["doorman","concierge"],
}
for col, kws in AMENITY_FLAGS.items():
    df[col] = df["amenities_parsed"].apply(lambda lst: int(any(k in " ".join(lst) for k in kws)))
df["premium_amenity_score"] = df[list(AMENITY_FLAGS.keys())].sum(axis=1)

# Host features
log("Engineering host features...")
df["is_superhost"] = df["host_is_superhost"].fillna(False).astype(int)
if "host_experience_days" in df.columns:
    df["host_experience_years"] = df["host_experience_days"] / 365.25
if "calculated_host_listings_count" in df.columns:
    df["is_professional_host"] = (df["calculated_host_listings_count"] >= 5).astype(int)
host_components = ["is_superhost"]
if "host_response_rate" in df.columns: host_components.append("host_response_rate")
if "host_acceptance_rate" in df.columns: host_components.append("host_acceptance_rate")
df["host_quality_score"] = df[host_components].mean(axis=1)

# Review features (from listings metadata)
log("Engineering review features...")
df["log_number_of_reviews"] = np.log1p(df["number_of_reviews"].fillna(0))
df["has_reviews"] = (df["number_of_reviews"].fillna(0) > 0).astype(int)
subcols = [c for c in df.columns if "review_scores_" in c and "rating" not in c]
if subcols:
    df["composite_review_score"] = df[subcols].mean(axis=1)
elif "review_scores_rating" in df.columns:
    df["composite_review_score"] = df["review_scores_rating"]
else:
    df["composite_review_score"] = 4.5

if "review_recency_days" in df.columns:
    df["review_recency_bucket"] = pd.cut(
        df["review_recency_days"], bins=[-1, 30, 90, 180, 365, np.inf],
        labels=[4, 3, 2, 1, 0]
    ).astype(float).fillna(0).astype(int)
else:
    df["review_recency_bucket"] = 2

# ─────────────────────────────────────────────────────────────
# STEP 6b: REVIEW NLP SENTIMENT FEATURES  (Phase 2 Integration)
# Compute VADER sentiment aggregates per listing from reviews.csv
# and merge them as model features — so XGBoost learns directly
# from review text quality, not as a post-prediction adjustment.
# ─────────────────────────────────────────────────────────────
NLP_DEFAULTS = {
    "review_avg_sentiment":     0.0,   # VADER compound: -1..+1
    "review_positive_pct":      50.0,  # % positive reviews
    "review_negative_pct":      0.0,
    "review_avg_word_count":    30.0,  # avg words per review
    "review_quality_score":     50.0,  # composite 0-100
    "review_sentiment_trend":   0.0,   # recent vs overall sentiment delta
}
# Initialize with defaults (filled for listings with no reviews)
for col, default in NLP_DEFAULTS.items():
    df[col] = default

if reviews_path:
    log("Computing per-listing NLP sentiment from reviews.csv (Phase 2 integration)...")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia_rev = SentimentIntensityAnalyzer()

        # Load only the columns we need
        rev_df = pd.read_csv(
            reviews_path,
            usecols=["listing_id", "date", "comments"],
            dtype={"listing_id": int, "comments": str},
            parse_dates=["date"],
            low_memory=False,
        )
        rev_df = rev_df.dropna(subset=["comments"])
        rev_df = rev_df[rev_df["comments"].str.strip().str.len() > 5]

        # Compute VADER score per review (cap at 1000 chars for speed)
        rev_df["compound"] = rev_df["comments"].apply(
            lambda t: sia_rev.polarity_scores(str(t)[:1000])["compound"]
        )
        rev_df["word_count"] = rev_df["comments"].apply(
            lambda t: len(str(t).split())
        )
        rev_df["is_positive"] = (rev_df["compound"] >= 0.05).astype(int)
        rev_df["is_negative"] = (rev_df["compound"] <= -0.05).astype(int)

        # Sentinel for "recent" = last 180 days relative to most recent review date
        max_date = rev_df["date"].max()
        cutoff   = max_date - pd.Timedelta(days=180)
        recent   = rev_df[rev_df["date"] >= cutoff]

        # Aggregate per listing
        agg = rev_df.groupby("listing_id").agg(
            review_avg_sentiment   =("compound",    "mean"),
            review_positive_pct    =("is_positive",  lambda x: x.mean() * 100),
            review_negative_pct    =("is_negative",  lambda x: x.mean() * 100),
            review_avg_word_count  =("word_count",   "mean"),
            total_reviews_nlp      =("compound",    "count"),
        ).reset_index()

        # Review quality score (mirrors review_analysis_api.py formula)
        agg["review_quality_score"] = (
            # Sentiment component 0-40 pts
            ((agg["review_avg_sentiment"] + 1) / 2 * 40)
            # Volume component 0-20 pts
            + np.minimum(20, agg["total_reviews_nlp"] / 5)
            # Detail component 0-20 pts
            + np.minimum(20, agg["review_avg_word_count"] / 15)
        ).clip(0, 80)  # max 80 without diversity component

        # Sentiment trend: recent avg - overall avg (positive = improving)
        if len(recent) > 0:
            recent_agg = recent.groupby("listing_id")["compound"].mean().rename("recent_sentiment")
            agg = agg.merge(recent_agg, on="listing_id", how="left")
            agg["review_sentiment_trend"] = (
                agg["recent_sentiment"].fillna(agg["review_avg_sentiment"])
                - agg["review_avg_sentiment"]
            )
        else:
            agg["review_sentiment_trend"] = 0.0

        nlp_cols = list(NLP_DEFAULTS.keys())
        agg_subset = agg[["listing_id"] + [c for c in nlp_cols if c in agg.columns]]

        # Use map() via listing_id -> value dicts, avoiding DataFrame merge
        # collision with pre-initialized columns.
        if "id" in df.columns:
            listing_id_series = df["id"]
            for col, default in NLP_DEFAULTS.items():
                if col in agg_subset.columns:
                    col_map = agg_subset.set_index("listing_id")[col].to_dict()
                    df[col] = listing_id_series.map(col_map).fillna(default)

        n_matched = int((df["review_avg_sentiment"] != NLP_DEFAULTS["review_avg_sentiment"]).sum())
        log(f"  NLP features computed. Listings with review sentiment: {n_matched:,} / {len(df):,}")
        log(f"  Avg sentiment: {df['review_avg_sentiment'].mean():.4f}")
        log(f"  Avg quality score: {df['review_quality_score'].mean():.1f}")

    except ImportError:
        log("  vaderSentiment not installed - NLP features default to 0. Run: pip install vaderSentiment")
    except Exception as e:
        log(f"  Review NLP failed (non-fatal): {e}. Defaults used.")
else:
    log("  reviews.csv not found - NLP sentiment features set to defaults.")

# Property features
log("Engineering property features...")
if "availability_365" in df.columns:
    df["availability_rate"] = df["availability_365"] / 365.0
df["accommodates_sq"] = df["accommodates"] ** 2
if "bedrooms" in df.columns and "bathrooms" in df.columns:
    df["bedrooms_per_bathroom"] = df["bedrooms"] / (df["bathrooms"] + 0.5)
if "beds" in df.columns:
    df["beds_per_person"] = df["beds"] / df["accommodates"].clip(lower=1)

# Interaction features
df["reviews_x_score"] = df["log_number_of_reviews"] * df["composite_review_score"].fillna(4.5)
if "bedrooms" in df.columns:
    df["capacity_x_bedrooms"] = df["accommodates"] * df["bedrooms"].fillna(1)
df["luxury_x_capacity"] = df["premium_amenity_score"] * df["accommodates"]

# Categorical encoding
log("Encoding categorical features...")

# Room type — target encoding using mean log_price per type
# This gives XGBoost real price-correlated numeric values instead of arbitrary 0/1/2/3
rt_means = df.groupby("room_type")["log_price"].mean()
rt_means_dict = rt_means.to_dict()
global_mean_log_price_rt = float(df["log_price"].mean())
df["room_type_encoded"] = df["room_type"].map(rt_means_dict).fillna(global_mean_log_price_rt)

if "neighbourhood_group_cleansed" in df.columns:
    from sklearn.preprocessing import LabelEncoder
    le_borough = LabelEncoder()
    df["borough_encoded"] = le_borough.fit_transform(df["neighbourhood_group_cleansed"].fillna("Unknown"))
else:
    df["borough_encoded"] = 0

# Neighbourhood target encoding
if "neighbourhood_cleansed" in df.columns:
    nbhd_means = df.groupby("neighbourhood_cleansed")["log_price"].mean()
    df["neighbourhood_target_encoded"] = df["neighbourhood_cleansed"].map(nbhd_means).fillna(df["log_price"].mean())
    global_mean_log_price = float(df["log_price"].mean())
    nbhd_means_dict = nbhd_means.to_dict()
else:
    df["neighbourhood_target_encoded"] = df["log_price"].mean()
    nbhd_means_dict = {}
    global_mean_log_price = float(df["log_price"].mean())


log("Feature engineering complete.")

# ─────────────────────────────────────────────────────────────
# STEP 7: DEFINE FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────
CANDIDATE_FEATURES = [
    "neighbourhood_target_encoded", "borough_encoded", "latitude", "longitude",
    "room_type_encoded", "accommodates", "accommodates_sq", "bedrooms", "bathrooms",
    "beds", "beds_per_person", "availability_rate",
    "dist_times_square_km", "dist_central_park_km", "dist_jfk_airport_km",
    "dist_brooklyn_bridge_km", "dist_grand_central_km", "geo_cluster",
    "amenity_count", "premium_amenity_score",
    "has_pool", "has_gym", "has_parking", "has_elevator",
    "has_washer", "has_ac", "has_workspace", "has_doorman",
    "is_superhost", "host_quality_score", "host_experience_years",
    "is_professional_host",
    "log_number_of_reviews", "has_reviews", "reviews_per_month",
    "composite_review_score", "review_recency_bucket",
    "reviews_x_score", "capacity_x_bedrooms", "luxury_x_capacity",
    "desc_word_count", "desc_sentiment",
    "has_luxury_keywords", "has_cozy_keywords",
    "has_spacious_keywords", "has_renovated_keywords",
    # Phase 2 NLP features — per-listing sentiment from reviews.csv
    "review_avg_sentiment", "review_positive_pct", "review_negative_pct",
    "review_avg_word_count", "review_quality_score", "review_sentiment_trend",
]

FEATURE_COLS = [c for c in CANDIDATE_FEATURES if c in df.columns]
log(f"Feature columns: {len(FEATURE_COLS)}")

# Save engineered dataset
df[FEATURE_COLS + ["log_price", "price"]].to_csv(
    os.path.join(PROC_DIR, "listings_features.csv"), index=False
)
log(f"Saved listings_features.csv ({len(df):,} rows)")

# ─────────────────────────────────────────────────────────────
# STEP 8: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
log("Train/test split (80/20)...")
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

X = df[FEATURE_COLS].fillna(0)
y = df["log_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

scaler = RobustScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# STEP 9: TRAIN MODELS
# ─────────────────────────────────────────────────────────────
def mape(yt, yp):
    yt2, yp2 = np.expm1(yt), np.expm1(yp)
    return 100*float(np.mean(np.abs((yt2-yp2)/(yt2+1e-9))))

def eval_model(model, Xtr, Xte, ytr, yte, name):
    t0 = time.time()
    model.fit(Xtr, ytr)
    yp_tr = model.predict(Xtr); yp_te = model.predict(Xte)
    r2_tr = r2_score(ytr, yp_tr); r2_te = r2_score(yte, yp_te)
    rmse = np.sqrt(mean_squared_error(yte, yp_te))
    mp = mape(yte, yp_te)
    t = time.time()-t0
    flag = "[WARN]" if (r2_tr-r2_te)>0.1 else "[OK]  "
    log(f"  {flag} {name}: R2={r2_te:.4f}, RMSE={rmse:.4f}, MAPE={mp:.1f}%, {t:.1f}s")
    return {"Model": name, "R2_Test": r2_te, "RMSE": rmse, "MAPE": mp, "Time": t}, model

results = []

log("Training Ridge (baseline)...")
r, m = eval_model(Ridge(alpha=10.0), X_train_sc, X_test_sc, y_train, y_test, "Ridge")
results.append(r)

log("Training Random Forest...")
r, m = eval_model(
    RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                          max_features=0.5, n_jobs=-1, random_state=42),
    X_train, X_test, y_train, y_test, "Random Forest"
)
results.append(r)

log("Training XGBoost (initial)...")
xgb_init = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0
)
r, m = eval_model(xgb_init, X_train, X_test, y_train, y_test, "XGBoost (initial)")
results.append(r)

# LightGBM
try:
    import lightgbm as lgb
    log("Training LightGBM...")
    r, m_lgbm = eval_model(
        lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.7, n_jobs=-1,
                          random_state=42, verbosity=-1),
        X_train, X_test, y_train, y_test, "LightGBM"
    )
    results.append(r)
except ImportError:
    log("  LightGBM not available — skipping")

# ─────────────────────────────────────────────────────────────
# STEP 10: HYPERPARAMETER TUNING (XGBoost)
# ─────────────────────────────────────────────────────────────
log("Tuning XGBoost (20 iterations × 5-fold CV)...")
param_dist = {
    "n_estimators":     [300, 500, 700],
    "max_depth":        [5, 6, 7],
    "learning_rate":    [0.03, 0.05, 0.08],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "min_child_weight": [3, 5, 7],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [0.5, 1.0, 2.0],
}
search = RandomizedSearchCV(
    xgb.XGBRegressor(n_jobs=-1, random_state=42, verbosity=0),
    param_distributions=param_dist, n_iter=20, cv=5,
    scoring="r2", n_jobs=-1, random_state=42, verbose=0
)
search.fit(X_train, y_train)
best_model = search.best_estimator_
yp = best_model.predict(X_test)
r2_final = r2_score(y_test, yp)
mape_final = mape(y_test, yp)
rmse_final = np.sqrt(mean_squared_error(y_test, yp))
log(f"  [OK] XGBoost (Tuned): R2={r2_final:.4f}, RMSE={rmse_final:.4f}, MAPE={mape_final:.1f}%")
log(f"  Best params: {search.best_params_}")

# ─────────────────────────────────────────────────────────────
# STEP 11: SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
log("Saving model artifacts...")

model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, model_path, compress=3)
size_mb = os.path.getsize(model_path)/1e6
log(f"  Saved best_model.pkl ({size_mb:.1f} MB)")

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
log("  Saved scaler.pkl")

# Encoders
label_encoders = {
    "room_type_mean_prices": rt_means_dict,
    "neighbourhood_mean_prices": nbhd_means_dict,
    "global_mean_log_price": global_mean_log_price,
}
joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
log("  Saved label_encoders.pkl")

# Feature names
feat_meta = {
    "feature_names": FEATURE_COLS,
    "n_features": len(FEATURE_COLS),
    "target": "log_price",
    "transform": "log1p/expm1",
    "room_type_mean_prices": rt_means_dict,
}
with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
    json.dump(feat_meta, f, indent=2)
log("  Saved feature_names.json")

# Model metadata
metadata = {
    "model_type": "XGBoostRegressor",
    "best_params": search.best_params_,
    "r2_test": float(r2_final),
    "rmse_log": float(rmse_final),
    "mape_pct": float(mape_final),
    "feature_names": FEATURE_COLS,
    "target": "log_price",
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "all_results": results,
}
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
log("  Saved model_metadata.json")

# Save cleaned listings
df_save = df[[c for c in df.columns if c != "amenities_parsed"]].copy()
df_save.to_csv(os.path.join(PROC_DIR, "listings_clean.csv"), index=False)
log("  Saved listings_clean.csv")

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("PIPELINE COMPLETE!")
print("="*55)
print(f"  Best model:  XGBoost (Tuned)")
print(f"  R2 Test:     {r2_final:.4f}")
print(f"  MAPE:        {mape_final:.1f}%")
print(f"  RMSE (log):  {rmse_final:.4f}")
print(f"\n  Model saved to: models/best_model.pkl ({size_mb:.1f} MB)")
print(f"  Features:       {len(FEATURE_COLS)}")
print(f"  Training rows:  {len(X_train):,}")
print("\nNext: cd app && npm run dev  to launch the web app!")
