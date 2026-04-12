import { NextRequest, NextResponse } from "next/server";
import path from "path";

export const maxDuration = 60;

/**
 * POST /api/predict
 *
 * Runs XGBoost inference directly in Node.js via the ONNX Runtime.
 * No Python, no Flask, no external backend — everything runs on Vercel.
 *
 * Model: app/models/model.onnx  (3.6 MB, converted from best_model.pkl)
 * Feature order must match app/models/feature_names.json exactly.
 */

// ── Feature names in the EXACT order the model was trained ─────────────────
const FEATURE_NAMES = [
  "neighbourhood_target_encoded",
  "borough_encoded",
  "latitude",
  "longitude",
  "room_type_encoded",
  "accommodates",
  "accommodates_sq",
  "bedrooms",
  "bathrooms",
  "beds",
  "beds_per_person",
  "availability_rate",
  "dist_times_square_km",
  "dist_central_park_km",
  "dist_jfk_airport_km",
  "dist_brooklyn_bridge_km",
  "dist_grand_central_km",
  "geo_cluster",
  "amenity_count",
  "premium_amenity_score",
  "has_pool",
  "has_gym",
  "has_parking",
  "has_elevator",
  "has_washer",
  "has_ac",
  "has_workspace",
  "has_doorman",
  "is_superhost",
  "host_quality_score",
  "host_experience_years",
  "is_professional_host",
  "log_number_of_reviews",
  "has_reviews",
  "reviews_per_month",
  "composite_review_score",
  "review_recency_bucket",
  "reviews_x_score",
  "capacity_x_bedrooms",
  "luxury_x_capacity",
  "desc_word_count",
  "desc_sentiment",
  "has_luxury_keywords",
  "has_cozy_keywords",
  "has_spacious_keywords",
  "has_renovated_keywords",
  "review_avg_sentiment",
  "review_positive_pct",
  "review_negative_pct",
  "review_avg_word_count",
  "review_quality_score",
  "review_sentiment_trend",
];

// Room type target-encoded means (from training — room_type_mean_prices in feature_names.json)
const RT_LABEL_TO_NAME: Record<number, string> = {
  0: "Entire home/apt",
  1: "Hotel room",
  2: "Private room",
  3: "Shared room",
};
const RT_MEAN_LOG_PRICE: Record<string, number> = {
  "Entire home/apt": 5.14,
  "Hotel room":      4.98,
  "Private room":    4.48,
  "Shared room":     4.20,
};

// ── Helper ─────────────────────────────────────────────────────────────────
const num = (val: unknown, fallback = 0): number => {
  const n = parseFloat(String(val));
  return isNaN(n) ? fallback : n;
};

// ── Build the full 52-feature vector from the request body ─────────────────
function buildFeatureVector(body: Record<string, unknown>): Float32Array {
  const accommodates   = Math.max(1, num(body.accommodates, 2));
  const beds           = Math.max(0, num(body.beds, 1));
  const numReviews     = Math.max(0, num(body.number_of_reviews, 10));
  const reviewScore    = Math.min(5, Math.max(0, num(body.review_scores_rating, 4.7)));
  const hostResponse   = Math.min(1, Math.max(0, num(body.host_response_rate, 0.9)));
  const hostAccept     = Math.min(1, Math.max(0, num(body.host_acceptance_rate, 0.9)));
  const isSuperhost    = num(body.is_superhost, 0);
  const availability   = Math.min(365, Math.max(0, num(body.availability_365, 180)));
  const logReviews     = Math.log1p(numReviews);
  const hostListings   = Math.max(1, num(body.calculated_host_listings_count, 1));
  const hostExpDays    = Math.max(0, num(body.host_experience_days, 365));
  const premAmenity    = Math.min(9, Math.max(0, num(body.premium_amenity_score, 3)));

  // Room type target-encoding (frontend sends 0-3, model wants log-price mean)
  const rawRt          = Math.round(num(body.room_type_encoded, 0));
  const rtName         = RT_LABEL_TO_NAME[rawRt] ?? "Entire home/apt";
  const rtEncoded      = RT_MEAN_LOG_PRICE[rtName] ?? 5.14;

  // Derived / engineered features
  const bedsPerPerson        = beds / Math.max(accommodates, 1);
  const availRate            = availability / 365;
  const hostQualityScore     = (isSuperhost + hostResponse + hostAccept) / 3;
  const hostExpYears         = hostExpDays / 365;
  const isProfessionalHost   = hostListings >= 5 ? 1 : 0;
  const hasReviews           = numReviews > 0 ? 1 : 0;
  const reviewsPerMonth      = Math.min(30, num(body.reviews_per_month, 1));
  const compositeReviewScore = reviewScore;
  const reviewsXScore        = logReviews * compositeReviewScore;
  const capacityXBedrooms    = accommodates * Math.max(0, num(body.bedrooms, 1));
  const luxuryXCapacity      = premAmenity * accommodates;

  const featureMap: Record<string, number> = {
    neighbourhood_target_encoded: num(body.neighbourhood_target_encoded, 5.2),
    borough_encoded:              num(body.borough_encoded, 1),
    latitude:                     num(body.latitude, 40.7128),
    longitude:                    num(body.longitude, -74.006),
    room_type_encoded:            rtEncoded,
    accommodates,
    accommodates_sq:              accommodates ** 2,
    bedrooms:                     Math.max(0, num(body.bedrooms, 1)),
    bathrooms:                    Math.max(0, num(body.bathrooms, 1)),
    beds,
    beds_per_person:              bedsPerPerson,
    availability_rate:            availRate,
    dist_times_square_km:         num(body.dist_times_square_km, 5),
    dist_central_park_km:         num(body.dist_central_park_km, 4),
    dist_jfk_airport_km:          num(body.dist_jfk_airport_km, 20),
    dist_brooklyn_bridge_km:      num(body.dist_brooklyn_bridge_km, 4),
    dist_grand_central_km:        num(body.dist_grand_central_km, 3),
    geo_cluster:                  num(body.geo_cluster, 0),
    amenity_count:                Math.max(0, num(body.amenity_count, 20)),
    premium_amenity_score:        premAmenity,
    has_pool:                     num(body.has_pool, 0),
    has_gym:                      num(body.has_gym, 0),
    has_parking:                  num(body.has_parking, 0),
    has_elevator:                 num(body.has_elevator, 0),
    has_washer:                   num(body.has_washer, 0),
    has_ac:                       num(body.has_ac, 0),
    has_workspace:                num(body.has_workspace, 0),
    has_doorman:                  num(body.has_doorman, 0),
    is_superhost:                 isSuperhost,
    host_quality_score:           hostQualityScore,
    host_experience_years:        hostExpYears,
    is_professional_host:         isProfessionalHost,
    log_number_of_reviews:        logReviews,
    has_reviews:                  hasReviews,
    reviews_per_month:            reviewsPerMonth,
    composite_review_score:       compositeReviewScore,
    review_recency_bucket:        2.0,  // default
    reviews_x_score:              reviewsXScore,
    capacity_x_bedrooms:          capacityXBedrooms,
    luxury_x_capacity:            luxuryXCapacity,
    desc_word_count:              100.0,
    desc_sentiment:               0.5,
    has_luxury_keywords:          0,
    has_cozy_keywords:            0,
    has_spacious_keywords:        0,
    has_renovated_keywords:       0,
    // Review NLP features (default neutral — used if listing_id provided and found)
    review_avg_sentiment:         num(body.review_avg_sentiment, 0.0),
    review_positive_pct:          num(body.review_positive_pct, 50.0),
    review_negative_pct:          num(body.review_negative_pct, 0.0),
    review_avg_word_count:        num(body.review_avg_word_count, 30.0),
    review_quality_score:         num(body.review_quality_score, 50.0),
    review_sentiment_trend:       num(body.review_sentiment_trend, 0.0),
  };

  // Build Float32Array in exact feature order
  return new Float32Array(FEATURE_NAMES.map(name => featureMap[name] ?? 0.0));
}

// ── ONNX inference ──────────────────────────────────────────────────────────
async function runOnnxInference(featureVector: Float32Array): Promise<number> {
  // Dynamic import — onnxruntime-node is a native module
  const ort = await import("onnxruntime-node");

  const modelPath = path.join(process.cwd(), "models", "model.onnx");
  const session   = await ort.InferenceSession.create(modelPath);

  const inputName = session.inputNames[0];
  const tensor    = new ort.Tensor("float32", featureVector, [1, FEATURE_NAMES.length]);
  const results   = await session.run({ [inputName]: tensor });

  const outputName = session.outputNames[0];
  const logPred    = (results[outputName].data as Float32Array)[0];
  return logPred;
}

// ── POST handler ────────────────────────────────────────────────────────────
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    if (!body || typeof body !== "object") {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }

    // Validation
    const errors: string[] = [];
    const checkRange = (field: string, val: unknown, lo: number, hi: number, label?: string) => {
      const v = parseFloat(String(val));
      if (!isNaN(v) && val != null && (v < lo || v > hi)) {
        errors.push(`${label ?? field} must be between ${lo} and ${hi} (got ${v})`);
      }
    };
    checkRange("accommodates",         body.accommodates,         1,  16, "Guests");
    checkRange("bedrooms",             body.bedrooms,             0,  20, "Bedrooms");
    checkRange("bathrooms",            body.bathrooms,            0,  20, "Bathrooms");
    checkRange("beds",                 body.beds,                 0,  30, "Beds");
    checkRange("review_scores_rating", body.review_scores_rating, 0,  5,  "Review score");
    checkRange("host_response_rate",   body.host_response_rate,   0,  1,  "Host response rate");
    checkRange("number_of_reviews",    body.number_of_reviews,    0,  10000, "Number of reviews");
    if (errors.length > 0) {
      return NextResponse.json({ error: "Input validation failed", details: errors }, { status: 422 });
    }

    const featureVector = buildFeatureVector(body as Record<string, unknown>);
    const logPred       = await runOnnxInference(featureVector);
    const price         = Math.exp(logPred) - 1;  // expm1

    const priceRounded = Math.round(price * 100) / 100;
    const priceLow     = Math.round(price * 0.82 * 100) / 100;
    const priceHigh    = Math.round(price * 1.18 * 100) / 100;

    // Feature importances are static (from training) — top 5 most impactful
    const topFeatures = [
      { feature: "neighbourhood_target_encoded", importance: 0.1823 },
      { feature: "room_type_encoded",            importance: 0.1512 },
      { feature: "accommodates",                 importance: 0.1104 },
      { feature: "reviews_x_score",              importance: 0.0723 },
      { feature: "dist_times_square_km",         importance: 0.0612 },
    ];

    return NextResponse.json({
      predicted_price:     priceRounded,
      price_low:           priceLow,
      price_high:          priceHigh,
      log_prediction:      Math.round(logPred * 10000) / 10000,
      top_features:        topFeatures,
      review_nlp_used:     false,
      review_nlp_features: null,
    });

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/predict] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
