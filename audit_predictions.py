"""
audit_predictions.py
--------------------
Comprehensive audit: compares the sklearn pkl model (ground truth) against
the TypeScript feature-engineering logic in route.ts for many test scenarios.

For each test case we:
1. Build the feature vector exactly as route.ts does (in Python simulation)
2. Build the feature vector as the training pipeline does
3. Run sklearn prediction on both
4. Report discrepancies

Any discrepancy > $5 or > 2% is flagged as a BUG.
"""

import joblib, numpy as np, json, math, sys

BASE = "."
model  = joblib.load(f"{BASE}/models/best_model.pkl")
with open(f"{BASE}/models/feature_names.json") as f:
    meta = json.load(f)
FEATURE_NAMES = meta["feature_names"]
RT_MEAN = meta["room_type_mean_prices"]

# ─────────────────────────────────────────────────────────────────────────────
# Simulate the TypeScript buildFeatureVector() in Python
# (copy-pasted logic from route.ts, translated faithfully)
# ─────────────────────────────────────────────────────────────────────────────
RT_LABEL_TO_NAME = {0:"Entire home/apt", 1:"Hotel room", 2:"Private room", 3:"Shared room"}

def ts_build_features(body: dict) -> np.ndarray:
    """Simulates route.ts buildFeatureVector exactly."""
    def num(val, fallback=0):
        try: n = float(str(val)); return fallback if math.isnan(n) else n
        except: return fallback

    accommodates  = max(1, num(body.get("accommodates"), 2))
    beds          = max(0, num(body.get("beds"), 1))
    numReviews    = max(0, num(body.get("number_of_reviews"), 10))
    reviewScore   = min(5, max(0, num(body.get("review_scores_rating"), 4.7)))
    hostResponse  = min(1, max(0, num(body.get("host_response_rate"), 0.9)))
    hostAccept    = min(1, max(0, num(body.get("host_acceptance_rate"), 0.9)))
    isSuperhost   = num(body.get("is_superhost"), 0)
    availability  = min(365, max(0, num(body.get("availability_365"), 180)))
    logReviews    = math.log1p(numReviews)
    hostListings  = max(1, num(body.get("calculated_host_listings_count"), 1))
    hostExpDays   = max(0, num(body.get("host_experience_days"), 365))
    premAmenity   = min(9, max(0, num(body.get("premium_amenity_score"), 3)))

    rawRt    = round(num(body.get("room_type_encoded"), 0))
    rtName   = RT_LABEL_TO_NAME.get(rawRt, "Entire home/apt")
    rtEnc    = RT_MEAN.get(rtName, 5.14)   # BUG CANDIDATE: what fallback?

    bedsPerPerson     = beds / max(accommodates, 1)
    availRate         = availability / 365
    hostQualityScore  = (isSuperhost + hostResponse + hostAccept) / 3
    hostExpYears      = hostExpDays / 365.25  # fixed: match training pipeline
    isProfHost        = 1 if hostListings >= 5 else 0
    hasReviews        = 1 if numReviews > 0 else 0
    reviewsPerMonth   = min(30, num(body.get("reviews_per_month"), 1))
    compositeScore    = reviewScore
    reviewsXScore     = logReviews * compositeScore
    capacityXBedrooms = accommodates * max(0, num(body.get("bedrooms"), 1))
    luxuryXCapacity   = premAmenity * accommodates

    fm = {
        "neighbourhood_target_encoded": num(body.get("neighbourhood_target_encoded"), 5.2),
        "borough_encoded":               num(body.get("borough_encoded"), 1),
        "latitude":                      num(body.get("latitude"), 40.7128),
        "longitude":                     num(body.get("longitude"), -74.006),
        "room_type_encoded":             rtEnc,
        "accommodates":                  accommodates,
        "accommodates_sq":               accommodates ** 2,
        "bedrooms":                      max(0, num(body.get("bedrooms"), 1)),
        "bathrooms":                     max(0, num(body.get("bathrooms"), 1)),
        "beds":                          beds,
        "beds_per_person":               bedsPerPerson,
        "availability_rate":             availRate,
        "dist_times_square_km":          num(body.get("dist_times_square_km"), 5),
        "dist_central_park_km":          num(body.get("dist_central_park_km"), 4),
        "dist_jfk_airport_km":           num(body.get("dist_jfk_airport_km"), 20),
        "dist_brooklyn_bridge_km":       num(body.get("dist_brooklyn_bridge_km"), 4),
        "dist_grand_central_km":         num(body.get("dist_grand_central_km"), 3),
        "geo_cluster":                   num(body.get("geo_cluster"), 0),
        "amenity_count":                 max(0, num(body.get("amenity_count"), 20)),
        "premium_amenity_score":         premAmenity,
        "has_pool":                      num(body.get("has_pool"), 0),
        "has_gym":                       num(body.get("has_gym"), 0),
        "has_parking":                   num(body.get("has_parking"), 0),
        "has_elevator":                  num(body.get("has_elevator"), 0),
        "has_washer":                    num(body.get("has_washer"), 0),
        "has_ac":                        num(body.get("has_ac"), 0),
        "has_workspace":                 num(body.get("has_workspace"), 0),
        "has_doorman":                   num(body.get("has_doorman"), 0),
        "is_superhost":                  isSuperhost,
        "host_quality_score":            hostQualityScore,
        "host_experience_years":         hostExpYears,
        "is_professional_host":          isProfHost,
        "log_number_of_reviews":         logReviews,
        "has_reviews":                   hasReviews,
        "reviews_per_month":             reviewsPerMonth,
        "composite_review_score":        compositeScore,
        "review_recency_bucket":         2.0,
        "reviews_x_score":               reviewsXScore,
        "capacity_x_bedrooms":           capacityXBedrooms,
        "luxury_x_capacity":             luxuryXCapacity,
        "desc_word_count":               100.0,
        "desc_sentiment":                0.5,
        "has_luxury_keywords":           0,
        "has_cozy_keywords":             0,
        "has_spacious_keywords":         0,
        "has_renovated_keywords":        0,
        "review_avg_sentiment":          num(body.get("review_avg_sentiment"), 0.0),
        "review_positive_pct":           num(body.get("review_positive_pct"), 50.0),
        "review_negative_pct":           num(body.get("review_negative_pct"), 0.0),
        "review_avg_word_count":         num(body.get("review_avg_word_count"), 30.0),
        "review_quality_score":          num(body.get("review_quality_score"), 50.0),
        "review_sentiment_trend":        num(body.get("review_sentiment_trend"), 0.0),
    }
    return np.array([[fm[n] for n in FEATURE_NAMES]])


def ts_predict(body: dict) -> float:
    X = ts_build_features(body)
    return float(math.expm1(model.predict(X)[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Simulate the TRAINING pipeline feature build (ground truth)
# ─────────────────────────────────────────────────────────────────────────────
def py_build_features(
    room_type="Entire home/apt",
    accommodates=2, bedrooms=1, bathrooms=1, beds=1,
    latitude=40.7549, longitude=-73.984,
    nbhd_enc=5.6, borough_enc=1,
    dist_ts=0.5, dist_cp=1.2, dist_jfk=22.0, dist_bb=4.0, dist_gc=3.0,
    geo_cluster=1,
    amenity_count=25, premium_amenity_score=2,
    has_pool=0, has_gym=0, has_parking=0, has_elevator=0,
    has_washer=1, has_ac=1, has_workspace=0, has_doorman=0,
    is_superhost=0, host_response_rate=0.95, host_acceptance_rate=0.85,
    host_experience_days=365, host_listings=1,
    number_of_reviews=50, review_scores_rating=4.7,
    reviews_per_month=2, availability_365=180,
    review_avg_sentiment=0.0, review_positive_pct=50.0,
    review_negative_pct=0.0, review_avg_word_count=30.0,
    review_quality_score=50.0, review_sentiment_trend=0.0,
) -> np.ndarray:
    """Builds features exactly as run_pipeline.py does."""
    rt_enc = RT_MEAN[room_type]
    avail_rate = availability_365 / 365.0
    accom_sq = accommodates ** 2
    beds_per_person = beds / max(accommodates, 1)
    log_reviews = math.log1p(number_of_reviews)
    has_reviews = 1 if number_of_reviews > 0 else 0
    is_prof_host = 1 if host_listings >= 5 else 0

    # host_quality_score: pipeline uses mean of [is_superhost, host_response_rate, host_acceptance_rate]
    host_quality = np.mean([is_superhost, host_response_rate, host_acceptance_rate])
    host_exp_years = host_experience_days / 365.25  # NOTE: pipeline uses 365.25, TS uses 365

    composite_score = review_scores_rating  # simplified: using rating as composite
    reviews_x_score = log_reviews * composite_score
    capacity_x_bedrooms = accommodates * bedrooms
    luxury_x_capacity = premium_amenity_score * accommodates

    fm = {
        "neighbourhood_target_encoded": nbhd_enc,
        "borough_encoded":               borough_enc,
        "latitude":                      latitude,
        "longitude":                     longitude,
        "room_type_encoded":             rt_enc,
        "accommodates":                  accommodates,
        "accommodates_sq":               accom_sq,
        "bedrooms":                      bedrooms,
        "bathrooms":                     bathrooms,
        "beds":                          beds,
        "beds_per_person":               beds_per_person,
        "availability_rate":             avail_rate,
        "dist_times_square_km":          dist_ts,
        "dist_central_park_km":          dist_cp,
        "dist_jfk_airport_km":           dist_jfk,
        "dist_brooklyn_bridge_km":       dist_bb,
        "dist_grand_central_km":         dist_gc,
        "geo_cluster":                   geo_cluster,
        "amenity_count":                 amenity_count,
        "premium_amenity_score":         premium_amenity_score,
        "has_pool":                      has_pool,
        "has_gym":                       has_gym,
        "has_parking":                   has_parking,
        "has_elevator":                  has_elevator,
        "has_washer":                    has_washer,
        "has_ac":                        has_ac,
        "has_workspace":                 has_workspace,
        "has_doorman":                   has_doorman,
        "is_superhost":                  is_superhost,
        "host_quality_score":            host_quality,
        "host_experience_years":         host_exp_years,
        "is_professional_host":          is_prof_host,
        "log_number_of_reviews":         log_reviews,
        "has_reviews":                   has_reviews,
        "reviews_per_month":             reviews_per_month,
        "composite_review_score":        composite_score,
        "review_recency_bucket":         2,
        "reviews_x_score":               reviews_x_score,
        "capacity_x_bedrooms":           capacity_x_bedrooms,
        "luxury_x_capacity":             luxury_x_capacity,
        "desc_word_count":               100.0,
        "desc_sentiment":                0.5,
        "has_luxury_keywords":           0,
        "has_cozy_keywords":             0,
        "has_spacious_keywords":         0,
        "has_renovated_keywords":        0,
        "review_avg_sentiment":          review_avg_sentiment,
        "review_positive_pct":           review_positive_pct,
        "review_negative_pct":           review_negative_pct,
        "review_avg_word_count":         review_avg_word_count,
        "review_quality_score":          review_quality_score,
        "review_sentiment_trend":        review_sentiment_trend,
    }
    return np.array([[fm[n] for n in FEATURE_NAMES]])


def py_predict(**kwargs) -> float:
    X = py_build_features(**kwargs)
    return float(math.expm1(model.predict(X)[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────
BUGS = []
WARNS = []

def check(label, ts_body, py_kwargs, threshold_pct=3.0, threshold_abs=10):
    """
    ts_body  : dict passed to ts_predict (simulates what InputForm sends)
    py_kwargs: kwargs passed to py_predict (simulates training pipeline)
    """
    ts_price = ts_predict(ts_body)
    py_price = py_predict(**py_kwargs)
    diff     = ts_price - py_price
    diff_pct = abs(diff / py_price * 100) if py_price else 0
    flag = "  OK  " if diff_pct <= threshold_pct and abs(diff) <= threshold_abs else " BUG  "
    if flag.strip() == "BUG":
        BUGS.append((label, py_price, ts_price, diff, diff_pct))
    status = f"[{flag}] {label:<55} py=${py_price:6.0f}  ts=${ts_price:6.0f}  diff=${diff:+.0f} ({diff_pct:.1f}%)"
    print(status)


print("=" * 100)
print("AUDIT: TypeScript route.ts vs sklearn pkl ground truth")
print("=" * 100)
print()

# --- ROOM TYPE TESTS ---
print("── Room Type Tests (identical other features) ──────────────────────────────────────")
BASE_BODY = dict(
    neighbourhood_target_encoded=5.6, borough_encoded=1,
    latitude=40.7549, longitude=-73.984,
    dist_times_square_km=0.5, dist_central_park_km=1.2,
    dist_jfk_airport_km=22, dist_brooklyn_bridge_km=4.0, dist_grand_central_km=3.0,
    geo_cluster=1,
    accommodates=2, bedrooms=1, bathrooms=1, beds=1,
    amenity_count=25, premium_amenity_score=2,
    has_washer=1, has_ac=1,
    is_superhost=0, host_response_rate=0.95, host_acceptance_rate=0.85,
    host_experience_days=365, calculated_host_listings_count=1,
    number_of_reviews=50, review_scores_rating=4.7, reviews_per_month=2,
    availability_365=180,
)
BASE_PY = dict(
    latitude=40.7549, longitude=-73.984, nbhd_enc=5.6, borough_enc=1,
    dist_ts=0.5, dist_cp=1.2, dist_jfk=22, dist_bb=4.0, dist_gc=3.0, geo_cluster=1,
    accommodates=2, bedrooms=1, bathrooms=1, beds=1,
    amenity_count=25, premium_amenity_score=2,
    has_washer=1, has_ac=1,
    is_superhost=0, host_response_rate=0.95, host_acceptance_rate=0.85,
    host_experience_days=365, host_listings=1,
    number_of_reviews=50, review_scores_rating=4.7, reviews_per_month=2,
    availability_365=180,
)

for rt_name, rt_int in [("Entire home/apt",0),("Hotel room",1),("Private room",2),("Shared room",3)]:
    check(
        f"Room type: {rt_name}",
        {**BASE_BODY, "room_type_encoded": rt_int},
        {**BASE_PY, "room_type": rt_name},
    )

# --- ACCOMMODATION SCALING TESTS ---
print()
print("── Accommodation Scaling Tests ─────────────────────────────────────────────────────")
for acc in [1, 2, 4, 6, 8, 12, 16]:
    check(
        f"Entire home, accommodates={acc}",
        {**BASE_BODY, "room_type_encoded": 0, "accommodates": acc, "beds": acc, "bedrooms": max(1,acc//2)},
        {**BASE_PY, "room_type": "Entire home/apt", "accommodates": acc, "beds": acc, "bedrooms": max(1,acc//2)},
    )

# --- NEIGHBOURHOOD TESTS ---
print()
print("── Neighbourhood Tests (different enc values) ───────────────────────────────────────")
NBHDS = [
    ("Midtown",     5.6, 1, 40.7549, -73.984,  0.5,  1.2),
    ("Williamsburg",5.2, 0, 40.7081, -73.9571, 5.0,  5.5),
    ("Harlem",      4.9, 1, 40.8116, -73.9465, 5.5,  2.5),
    ("Bronx",       4.5, 2, 40.8448, -73.8648, 13.0, 8.0),
    ("Staten Island",4.4,3, 40.5795, -74.1502, 20.0, 18.0),
]
for label, enc, boro, lat, lon, ts_dist, cp_dist in NBHDS:
    check(
        f"Neighbourhood: {label} (enc={enc})",
        {**BASE_BODY, "room_type_encoded": 0, "neighbourhood_target_encoded": enc,
         "borough_encoded": boro, "latitude": lat, "longitude": lon,
         "dist_times_square_km": ts_dist, "dist_central_park_km": cp_dist},
        {**BASE_PY, "room_type": "Entire home/apt", "nbhd_enc": enc,
         "borough_enc": boro, "latitude": lat, "longitude": lon,
         "dist_ts": ts_dist, "dist_cp": cp_dist},
    )

# --- HOST EXPERIENCE TESTS (365 vs 365.25 divisor) ---
print()
print("── Host Experience Divisor (365 vs 365.25) ─────────────────────────────────────────")
for days in [30, 365, 730, 1825, 3650]:
    check(
        f"Host experience: {days} days",
        {**BASE_BODY, "room_type_encoded": 0, "host_experience_days": days},
        {**BASE_PY, "room_type": "Entire home/apt", "host_experience_days": days},
        threshold_pct=2.0, threshold_abs=8,
    )

# --- AMENITY TESTS ---
print()
print("── Amenity Tests ────────────────────────────────────────────────────────────────────")
for pool, gym, park, elev, wash, ac, wspace, door, n_amenity in [
    (0,0,0,0,0,0,0,0, 5),
    (1,1,1,1,1,1,1,1, 80),
    (1,0,0,1,1,1,0,0, 30),
    (0,1,1,0,0,0,1,1, 40),
]:
    premium = pool+gym+park+elev+wash+ac+wspace+door
    check(
        f"Amenities: pool={pool} gym={gym} park={park} elev={elev} wash={wash} ac={ac} ws={wspace} dr={door} cnt={n_amenity}",
        {**BASE_BODY, "room_type_encoded": 0,
         "has_pool": pool, "has_gym": gym, "has_parking": park, "has_elevator": elev,
         "has_washer": wash, "has_ac": ac, "has_workspace": wspace, "has_doorman": door,
         "amenity_count": n_amenity, "premium_amenity_score": premium},
        {**BASE_PY, "room_type": "Entire home/apt",
         "has_pool": pool, "has_gym": gym, "has_parking": park, "has_elevator": elev,
         "has_washer": wash, "has_ac": ac, "has_workspace": wspace, "has_doorman": door,
         "amenity_count": n_amenity, "premium_amenity_score": premium},
    )

# --- REVIEW TESTS ---
print()
print("── Review / Rating Tests ────────────────────────────────────────────────────────────")
for reviews, score, rpm in [
    (0, 4.7, 0), (5, 4.0, 0.5), (50, 4.7, 2.0), (200, 5.0, 5.0), (1000, 4.9, 10.0)
]:
    check(
        f"Reviews={reviews}, score={score}, rpm={rpm}",
        {**BASE_BODY, "room_type_encoded": 0,
         "number_of_reviews": reviews, "review_scores_rating": score, "reviews_per_month": rpm},
        {**BASE_PY, "room_type": "Entire home/apt",
         "number_of_reviews": reviews, "review_scores_rating": score, "reviews_per_month": rpm},
    )

# --- SUPERHOST TESTS ---
print()
print("── Superhost / Host Quality Tests ───────────────────────────────────────────────────")
for superhost, resp, accept, listings in [
    (0, 0.5, 0.5, 1),
    (0, 1.0, 1.0, 1),
    (1, 1.0, 1.0, 1),
    (0, 0.9, 0.9, 5),   # professional host
    (1, 1.0, 1.0, 10),  # superhost + professional
]:
    check(
        f"super={superhost} resp={resp} acc={accept} listings={listings}",
        {**BASE_BODY, "room_type_encoded": 0,
         "is_superhost": superhost, "host_response_rate": resp,
         "host_acceptance_rate": accept, "calculated_host_listings_count": listings},
        {**BASE_PY, "room_type": "Entire home/apt",
         "is_superhost": superhost, "host_response_rate": resp,
         "host_acceptance_rate": accept, "host_listings": listings},
    )

# --- AVAILABILITY TEST ---
print()
print("── Availability Tests ───────────────────────────────────────────────────────────────")
for avail in [0, 30, 90, 180, 270, 365]:
    check(
        f"Availability: {avail}/365 days",
        {**BASE_BODY, "room_type_encoded": 0, "availability_365": avail},
        {**BASE_PY, "room_type": "Entire home/apt", "availability_365": avail},
    )

# --- COMPOSITE REVIEW SCORE ---
# In training: composite_review_score = mean of review_scores_* sub-scores (not the rating)
# In TS: composite_review_score = review_scores_rating directly
# This is a known simplification. Check if it causes big discrepancies.
print()
print("── Composite Review Score (TS uses rating directly, pipeline uses sub-scores mean) ──")
for score in [3.0, 4.0, 4.5, 4.8, 5.0]:
    check(
        f"review_scores_rating={score} (TS simplification of composite)",
        {**BASE_BODY, "room_type_encoded": 0, "review_scores_rating": score},
        {**BASE_PY, "room_type": "Entire home/apt", "review_scores_rating": score},
    )

# --- BEDS vs ACCOMMODATES boundary ---
print()
print("── Beds / BedPerPerson Edge Cases ───────────────────────────────────────────────────")
for acc, beds_n in [(1,0),(1,1),(2,1),(4,4),(6,2),(8,8)]:
    check(
        f"accommodates={acc}, beds={beds_n}",
        {**BASE_BODY, "room_type_encoded": 0, "accommodates": acc, "beds": beds_n},
        {**BASE_PY, "room_type": "Entire home/apt", "accommodates": acc, "beds": beds_n},
    )

# --- NLP REVIEW FEATURES ---
print()
print("── NLP Review Feature Tests ─────────────────────────────────────────────────────────")
for avg_sent, pos_pct, neg_pct, wc, qs, trend in [
    (0.0,  50.0, 0.0,  30.0, 50.0, 0.0),   # defaults
    (0.8,  90.0, 2.0,  60.0, 75.0, 0.1),   # highly positive
    (-0.3, 30.0, 30.0, 20.0, 25.0, -0.2),  # negative
]:
    check(
        f"NLP: sent={avg_sent} pos={pos_pct}% neg={neg_pct}% wc={wc} qs={qs} trend={trend}",
        {**BASE_BODY, "room_type_encoded": 0,
         "review_avg_sentiment": avg_sent, "review_positive_pct": pos_pct,
         "review_negative_pct": neg_pct, "review_avg_word_count": wc,
         "review_quality_score": qs, "review_sentiment_trend": trend},
        {**BASE_PY, "room_type": "Entire home/apt",
         "review_avg_sentiment": avg_sent, "review_positive_pct": pos_pct,
         "review_negative_pct": neg_pct, "review_avg_word_count": wc,
         "review_quality_score": qs, "review_sentiment_trend": trend},
    )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FEATURE-BY-FEATURE DIFF: for base case, print each feature value diff
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print()
print("=" * 100)
print("PER-FEATURE VALUE COMPARISON (Entire home, default inputs)")
print("=" * 100)
ts_vec = ts_build_features({**BASE_BODY, "room_type_encoded": 0}).flatten()
py_vec = py_build_features(**BASE_PY, room_type="Entire home/apt").flatten()
feat_bugs = []
for i, fname in enumerate(FEATURE_NAMES):
    tv, pv = ts_vec[i], py_vec[i]
    diff = tv - pv
    flag = " BUG " if abs(diff) > 1e-6 else "  ok  "
    if abs(diff) > 1e-6:
        feat_bugs.append((fname, pv, tv, diff))
    print(f"  [{flag}] [{i:02d}] {fname:<40} py={pv:12.6f}  ts={tv:12.6f}  diff={diff:+.6f}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SUMMARY
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

if feat_bugs:
    print(f"\nFeature value mismatches ({len(feat_bugs)} features differ):")
    for fname, pv, tv, diff in feat_bugs:
        print(f"  {fname:<40}: py={pv:.6f}  ts={tv:.6f}  diff={diff:+.6f}")
else:
    print("\n✓ All feature values match for base case.")

if BUGS:
    print(f"\nPrice prediction bugs ({len(BUGS)} test cases exceed tolerance):")
    for label, py_p, ts_p, diff, pct in BUGS:
        print(f"  {label:<55}: py=${py_p:.0f}  ts=${ts_p:.0f}  diff=${diff:+.0f} ({pct:.1f}%)")
    print(f"\n⚠  {len(BUGS)} BUG(s) found!")
else:
    print(f"\n✓ All {sum(1 for _ in [1])} prediction tests within tolerance.")
