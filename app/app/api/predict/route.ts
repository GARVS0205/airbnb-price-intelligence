import { NextRequest, NextResponse } from "next/server";
import path from "path";
import { spawn } from "child_process";

export const maxDuration = 80; // Allow enough time for Render cold-starts

/**
 * POST /api/predict
 *
 * In production (Vercel): forwards request to PYTHON_API_URL (Flask backend on Railway/Render)
 * In development (local): spawns Python subprocess directly
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    if (!body || typeof body !== "object") {
      return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
    }

    const features = sanitizeFeatures(body);
    if (body.listing_id != null) {
      const id = parseInt(String(body.listing_id), 10);
      if (!isNaN(id) && id > 0) {
        (features as Record<string, unknown>).listing_id = id;
      }
    }

    const pythonApiUrl = process.env.PYTHON_API_URL;

    if (pythonApiUrl) {
      // ── Production: call Flask backend via HTTP ──────────────────────────
      const res = await fetch(`${pythonApiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
        signal: AbortSignal.timeout(70000),  // 70s — Render cold-start can take ~60s
      });
      const data = await res.json();
      return NextResponse.json(data, { status: res.ok ? 200 : 500 });
    } else {
      // ── Development: spawn Python subprocess ────────────────────────────
      const scriptPath = path.join(process.cwd(), "predict_api.py");
      const pythonCmd = resolvePythonCommand();
      const prediction = await runPythonInference(scriptPath, features, pythonCmd);
      if ((prediction as Record<string, unknown>).details) {
        return NextResponse.json(prediction, { status: 422 });
      }
      return NextResponse.json(prediction, { status: 200 });
    }

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/predict] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// ── Feature sanitization ─────────────────────────────────────────────────────
function sanitizeFeatures(body: Record<string, unknown>): Record<string, number> {
  const num = (val: unknown, fallback = 0): number => {
    const n = parseFloat(String(val));
    return isNaN(n) ? fallback : n;
  };

  return {
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
    room_type_encoded:              num(body.room_type_encoded, 0),
    accommodates:                   Math.max(1, num(body.accommodates, 2)),
    accommodates_sq:                Math.max(1, num(body.accommodates, 2)) ** 2,
    bedrooms:                       Math.max(0, num(body.bedrooms, 1)),
    bathrooms:                      Math.max(0, num(body.bathrooms, 1)),
    beds:                           Math.max(0, num(body.beds, 1)),
    amenity_count:                  Math.max(0, num(body.amenity_count, 20)),
    has_ac:                         num(body.has_ac, 0),
    has_workspace:                  num(body.has_workspace, 0),
    has_pool:                       num(body.has_pool, 0),
    has_gym:                        num(body.has_gym, 0),
    has_washer:                     num(body.has_washer, 0),
    has_parking:                    num(body.has_parking, 0),
    has_elevator:                   num(body.has_elevator, 0),
    review_scores_rating:           Math.min(5, Math.max(0, num(body.review_scores_rating, 4.7))),
    number_of_reviews:              Math.max(0, num(body.number_of_reviews, 10)),
    log_number_of_reviews:          Math.log1p(Math.max(0, num(body.number_of_reviews, 10))),
    is_superhost:                   num(body.is_superhost, 0),
    host_response_rate:             Math.min(1, Math.max(0, num(body.host_response_rate, 0.9))),
    host_experience_days:           Math.max(0, num(body.host_experience_days, 365)),
    review_avg_sentiment:           num(body.review_avg_sentiment, 0),
    review_positive_pct:            num(body.review_positive_pct, 50),
    review_negative_pct:            num(body.review_negative_pct, 10),
    review_avg_length:              num(body.review_avg_length, 200),
    review_quality_score:           num(body.review_quality_score, 50),
    composite_review_score:         num(body.composite_review_score, 0),
  };
}

// ── Cross-platform Python resolution (Windows friendly) ─────────────────────
function resolvePythonCommand(): string {
  if (process.env.PYTHON_CMD && process.env.PYTHON_CMD.trim()) {
    return process.env.PYTHON_CMD.trim();
  }
  if (process.platform === "win32") {
    return "python"; // Windows images typically expose `python` and `py`
  }
  return "python3"; // Unix-like default
}

// ── Local dev: Python subprocess ─────────────────────────────────────────────
function runPythonInference(
  scriptPath: string,
  features: Record<string, unknown>,
  pythonCmd: string
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath], {
      shell: process.platform === "win32", // allow python.cmd / py.exe
    });
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    python.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

    python.on("close", (code: number) => {
      if (code !== 0) { reject(new Error(`Python script failed (code ${code}): ${stderr}`)); return; }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) { reject(new Error(result.error)); } else { resolve(result); }
      } catch { reject(new Error(`Failed to parse Python output: ${stdout}`)); }
    });

    python.on("error", (err: Error) => { reject(new Error(`Failed to spawn Python: ${err.message}`)); });
    python.stdin.write(JSON.stringify(features));
    python.stdin.end();
  });
}
