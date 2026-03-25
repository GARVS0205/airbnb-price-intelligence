# Python Flask API for ListingLens ML inference
# Deploy this separately on Railway or Render.
# Next.js calls these endpoints via HTTP instead of subprocess.

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the app directory to path so predict_api and review_analysis_api can import their deps
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

app = Flask(__name__)
CORS(app)  # Allow requests from Vercel frontend

# ── Lazy-load inference modules to avoid cold start penalty ──
_predictor = None
_reviewer  = None

def get_predictor():
    global _predictor
    if _predictor is None:
        import predict_api as pa
        _predictor = pa
    return _predictor

def get_reviewer():
    global _reviewer
    if _reviewer is None:
        import review_analysis_api as ra
        _reviewer = ra
    return _reviewer


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ListingLens ML API"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = request.get_json(force=True)
        if not features or not isinstance(features, dict):
            return jsonify({"error": "Invalid request body"}), 400

        pa = get_predictor()
        result = pa.run_inference(features)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-reviews", methods=["POST"])
def analyze_reviews():
    try:
        body = request.get_json(force=True)
        listing_id  = int(body.get("listing_id", 0))
        max_reviews = int(body.get("max_reviews", 300))

        if not listing_id or listing_id <= 0:
            return jsonify({"error": "A valid listing_id is required."}), 400

        ra = get_reviewer()
        result = ra.run_analysis(listing_id, max_reviews)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
