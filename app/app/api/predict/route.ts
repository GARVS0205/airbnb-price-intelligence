import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

/**
 * POST /api/predict
 * 
 * Calls the Python inference script via child_process,
 * passing the listing features as JSON through stdin.
 * 
 * This approach avoids needing a separate Python server —
 * the model runs in a serverless function on Vercel.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Basic input validation
    if (!body || typeof body !== "object") {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }

    // Clamp and validate numeric inputs
    const features = sanitizeFeatures(body);

    // Pass optional listing_id for Phase 2 review sentiment integration
    if (body.listing_id != null) {
      const id = parseInt(String(body.listing_id), 10);
      if (!isNaN(id) && id > 0) {
        (features as Record<string, unknown>).listing_id = id;
      }
    }

    // Path to Python inference script — located in the app/ root
    const scriptPath = path.join(process.cwd(), "predict_api.py");

    const prediction = await runPythonInference(scriptPath, features);

    // Python returns structured validation errors with details array
    if ((prediction as Record<string, unknown>).details) {
      return NextResponse.json(prediction, { status: 422 });
    }

    return NextResponse.json(prediction, { status: 200 });

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/predict] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/**
 * Sanitize and prepare feature values from frontend form data.
 * Maps user-friendly form fields to the feature names expected by the model.
 */
function sanitizeFeatures(body: Record<string, unknown>): Record<string, number> {
  const num = (val: unknown, fallback = 0): number => {
    const n = parseFloat(String(val));
    return isNaN(n) ? fallback : n;
  };

  return {
    // Location
    latitude:                       num(body.latitude, 40.7128),
    longitude:                      num(body.longitude, -74.006),
    neighbourhood_target_encoded:   num(body.neighbourhood_target_encoded, 5.2),
    borough_encoded:                num(body.borough_encoded, 1),
    dist_times_square_km:           num(body.dist_times_square_km, 5),
    dist_central_park_km:           num(body.dist_central_park_km, 4),
    dist_jfk_airport_km:            num(body.dist_jfk_airport_km, 20),
    dist_brooklyn_bridge_km:        num(body.dist_brooklyn_bridge_km, 4),
    dist_grand_central_km:          num(body.dist_grand_central_km, 3),
    geo_cluster:                    num(body.geo_cluster, 0),

    // Property
    room_type_encoded:              num(body.room_type_encoded, 0),
    accommodates:                   Math.max(1, num(body.accommodates, 2)),
    accommodates_sq:                Math.max(1, num(body.accommodates, 2)) ** 2,
    bedrooms:                       Math.max(0, num(body.bedrooms, 1)),
    bathrooms:                      Math.max(0.5, num(body.bathrooms, 1)),
    beds:                           Math.max(1, num(body.beds, 1)),
    beds_per_person:                Math.max(1, num(body.beds, 1)) / Math.max(1, num(body.accommodates, 2)),
    availability_365:               num(body.availability_365, 180),
    availability_rate:              num(body.availability_365, 180) / 365,
    minimum_nights:                 Math.max(1, num(body.minimum_nights, 2)),

    // Amenities
    amenity_count:                  num(body.amenity_count, 20),
    premium_amenity_score:          num(body.premium_amenity_score, 2),
    has_pool:                       num(body.has_pool, 0),
    has_gym:                        num(body.has_gym, 0),
    has_parking:                    num(body.has_parking, 0),
    has_doorman:                    num(body.has_doorman, 0),
    has_elevator:                   num(body.has_elevator, 0),
    has_washer:                     num(body.has_washer, 0),
    has_ac:                         num(body.has_ac, 1),
    has_workspace:                  num(body.has_workspace, 0),

    // Host
    is_superhost:                   num(body.is_superhost, 0),
    host_experience_days:           num(body.host_experience_days, 1000),
    host_response_rate:             Math.min(1, num(body.host_response_rate, 0.9)),
    host_acceptance_rate:           Math.min(1, num(body.host_acceptance_rate, 0.85)),
    calculated_host_listings_count: num(body.calculated_host_listings_count, 1),
    is_professional_host:           num(body.is_professional_host, 0),
    host_quality_score:             num(body.host_quality_score, 0.7),

    // Reviews
    number_of_reviews:              num(body.number_of_reviews, 50),
    log_number_of_reviews:          Math.log1p(num(body.number_of_reviews, 50)),
    reviews_per_month:              num(body.reviews_per_month, 2),
    review_recency_days:            num(body.review_recency_days, 60),
    review_scores_rating:           Math.min(5, num(body.review_scores_rating, 4.7)),
    composite_review_score:         Math.min(5, num(body.composite_review_score, 4.6)),
    review_recency_bucket:          num(body.review_recency_bucket, 3),
    has_reviews:                    1,

    // Text features
    desc_word_count:                num(body.desc_word_count, 120),
    desc_sentiment:                 num(body.desc_sentiment, 0.2),
    has_luxury_keywords:            num(body.has_luxury_keywords, 0),
    has_cozy_keywords:              num(body.has_cozy_keywords, 0),
    has_spacious_keywords:          num(body.has_spacious_keywords, 0),
    has_renovated_keywords:         num(body.has_renovated_keywords, 0),

    // Interaction features
    reviews_x_score:                Math.log1p(num(body.number_of_reviews, 50)) * Math.min(5, num(body.composite_review_score, 4.6)),
    luxury_x_capacity:              num(body.premium_amenity_score, 2) * Math.max(1, num(body.accommodates, 2)),
    capacity_x_bedrooms:            Math.max(1, num(body.accommodates, 2)) * Math.max(0, num(body.bedrooms, 1)),

    // Booking
    instant_bookable:               num(body.instant_bookable, 0),
  };
}

/**
 * Spawn Python subprocess, feed features via stdin, capture stdout JSON.
 */
function runPythonInference(
  scriptPath: string,
  features: Record<string, number>
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = spawn("python", [scriptPath]);
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    python.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

    python.on("close", (code: number) => {
      if (code !== 0) {
        reject(new Error(`Python script failed (code ${code}): ${stderr}`));
        return;
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result);
        }
      } catch {
        reject(new Error(`Failed to parse Python output: ${stdout}`));
      }
    });

    python.on("error", (err: Error) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    // Send features JSON to Python stdin
    python.stdin.write(JSON.stringify(features));
    python.stdin.end();
  });
}
