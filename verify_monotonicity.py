"""
Verify monotonicity: does reducing reviews from 100 to 70 now lower (or hold equal) the price?
"""
import json, re
import numpy as np

with open('app/models/model.json') as f:
    data = json.load(f)

trees = data['learner']['gradient_booster']['model']['trees']
bs_raw = data['learner']['learner_model_param']['base_score']
base_score = float(re.findall(r'[\d.E+\-]+', str(bs_raw))[0])

FEATURE_NAMES = [
    "neighbourhood_target_encoded","borough_encoded","latitude","longitude",
    "room_type_encoded","accommodates","accommodates_sq","bedrooms","bathrooms",
    "beds","beds_per_person","availability_rate","dist_times_square_km",
    "dist_central_park_km","dist_jfk_airport_km","dist_brooklyn_bridge_km",
    "dist_grand_central_km","geo_cluster","amenity_count","premium_amenity_score",
    "has_pool","has_gym","has_parking","has_elevator","has_washer","has_ac",
    "has_workspace","has_doorman","is_superhost","host_quality_score",
    "host_experience_years","is_professional_host","log_number_of_reviews",
    "has_reviews","reviews_per_month","composite_review_score","review_recency_bucket",
    "reviews_x_score","capacity_x_bedrooms","luxury_x_capacity","desc_word_count",
    "desc_sentiment","has_luxury_keywords","has_cozy_keywords","has_spacious_keywords",
    "has_renovated_keywords","review_avg_sentiment","review_positive_pct",
    "review_negative_pct","review_avg_word_count","review_quality_score","review_sentiment_trend",
]
feat_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}

def build_features(num_reviews, review_score=4.7, neighbourhood_te=5.2,
                   accommodates=2, bedrooms=1, bathrooms=1, beds=1, amenity_count=25):
    log_reviews = np.log1p(num_reviews)
    reviews_x_score = log_reviews * review_score
    host_quality = (0 + 0.95 + 0.9) / 3
    f = [0.0] * 52
    f[feat_idx["neighbourhood_target_encoded"]] = neighbourhood_te
    f[feat_idx["borough_encoded"]] = 1
    f[feat_idx["latitude"]] = 40.7549
    f[feat_idx["longitude"]] = -73.9840
    f[feat_idx["room_type_encoded"]] = 5.14   # Entire home/apt
    f[feat_idx["accommodates"]] = accommodates
    f[feat_idx["accommodates_sq"]] = accommodates ** 2
    f[feat_idx["bedrooms"]] = bedrooms
    f[feat_idx["bathrooms"]] = bathrooms
    f[feat_idx["beds"]] = beds
    f[feat_idx["beds_per_person"]] = beds / accommodates
    f[feat_idx["availability_rate"]] = 180 / 365
    f[feat_idx["dist_times_square_km"]] = 0.5
    f[feat_idx["dist_central_park_km"]] = 2.5
    f[feat_idx["dist_jfk_airport_km"]] = 18.0
    f[feat_idx["dist_brooklyn_bridge_km"]] = 4.0
    f[feat_idx["dist_grand_central_km"]] = 0.3
    f[feat_idx["amenity_count"]] = amenity_count
    f[feat_idx["premium_amenity_score"]] = 3
    f[feat_idx["host_quality_score"]] = host_quality
    f[feat_idx["host_experience_years"]] = 1.0
    f[feat_idx["log_number_of_reviews"]] = log_reviews
    f[feat_idx["has_reviews"]] = 1
    f[feat_idx["reviews_per_month"]] = 1.0
    f[feat_idx["composite_review_score"]] = review_score
    f[feat_idx["review_recency_bucket"]] = 2.0
    f[feat_idx["reviews_x_score"]] = reviews_x_score
    f[feat_idx["capacity_x_bedrooms"]] = accommodates * bedrooms
    f[feat_idx["luxury_x_capacity"]] = 3 * accommodates
    f[feat_idx["desc_word_count"]] = 100.0
    f[feat_idx["desc_sentiment"]] = 0.5
    f[feat_idx["review_positive_pct"]] = 50.0
    f[feat_idx["review_quality_score"]] = 50.0
    return f

def predict(features):
    score = base_score
    for tree in trees:
        lc, rc, si, sc, bw = (tree['left_children'], tree['right_children'],
                               tree['split_indices'], tree['split_conditions'], tree['base_weights'])
        node = 0
        while lc[node] != -1:
            node = lc[node] if features[si[node]] < sc[node] else rc[node]
        score += bw[node]
    return np.expm1(score)

print("=== Monotonicity Verification (Your Exact Scenario) ===")
print(f"{'Reviews':>10} | {'Price':>10} | {'Change':>10}")
print("-" * 38)
prev = None
for n in [50, 60, 70, 80, 90, 100, 110, 120]:
    price = predict(build_features(n, review_score=4.7))
    change = f"+${price-prev:.2f}" if prev is not None and price >= prev else (f"-${prev-price:.2f}" if prev is not None else "—")
    flag = " ← VIOLATION" if prev is not None and price < prev - 0.5 else ""
    print(f"{n:>10} | ${price:>9.2f} | {change:>10}{flag}")
    prev = price

print()
print("=== Review Score Monotonicity ===")
print(f"{'Score':>8} | {'Price':>10} | {'Change':>10}")
print("-" * 35)
prev = None
for score in [3.0, 3.5, 4.0, 4.5, 4.7, 4.9, 5.0]:
    price = predict(build_features(100, review_score=score))
    change = f"+${price-prev:.2f}" if prev is not None and price >= prev else (f"-${prev-price:.2f}" if prev is not None else "—")
    flag = " ← VIOLATION" if prev is not None and price < prev - 0.5 else ""
    print(f"{score:>8.1f} | ${price:>9.2f} | {change:>10}{flag}")
    prev = price
