/** Shared TypeScript types for the Airbnb Price Intelligence app */

export interface FormData {
  // Optional: Listing ID enables Phase 2 review sentiment integration
  listing_id?: number;

  // Location
  neighbourhood_target_encoded?: number;
  borough_encoded?: number;
  latitude?: number;
  longitude?: number;
  dist_times_square_km?: number;
  dist_central_park_km?: number;
  dist_jfk_airport_km?: number;
  dist_brooklyn_bridge_km?: number;
  dist_grand_central_km?: number;
  dist_statue_liberty_km?: number;
  dist_columbia_univ_km?: number;
  dist_union_square_km?: number;
  geo_cluster?: number;

  // Property
  room_type_encoded?: number;
  accommodates: number;
  bedrooms: number;
  bathrooms: number;
  beds: number;
  minimum_nights?: number;
  availability_365?: number;

  // Amenities
  amenity_count?: number;
  premium_amenity_score?: number;
  has_pool?: number;
  has_gym?: number;
  has_parking?: number;
  has_doorman?: number;
  has_elevator?: number;
  has_washer?: number;
  has_ac?: number;
  has_workspace?: number;

  // Host
  is_superhost?: number;
  host_response_rate?: number;
  host_acceptance_rate?: number;
  calculated_host_listings_count?: number;

  // Reviews
  number_of_reviews?: number;
  review_scores_rating?: number;
  reviews_per_month?: number;

  // Text
  has_luxury_keywords?: number;
  has_cozy_keywords?: number;
  has_spacious_keywords?: number;
  has_renovated_keywords?: number;

  // Booking
  instant_bookable?: number;
}

export interface TopFeature {
  feature: string;
  importance: number;
}

/**
 * The 6 per-listing NLP features computed from reviews.csv at inference time.
 * These are fed directly into the XGBoost model as part of the feature vector.
 * null means no listing_id was provided; defaults were used instead.
 */
export interface ReviewNlpFeatures {
  review_avg_sentiment: number;    // VADER compound: -1..+1
  review_positive_pct: number;     // % positive reviews
  review_negative_pct: number;     // % negative reviews
  review_avg_word_count: number;   // avg words per review
  review_quality_score: number;    // composite 0-80
  review_sentiment_trend: number;  // recent - overall sentiment delta
}

export interface PredictionResult {
  predicted_price: number;
  price_low: number;
  price_high: number;
  log_prediction: number;
  top_features: TopFeature[];
  /** True when real review NLP features were computed and used in the prediction */
  review_nlp_used?: boolean;
  /** The actual NLP feature values fed into this prediction (null if no listing_id) */
  review_nlp_features?: ReviewNlpFeatures | null;
  error?: string;
}
