# ListingLens

> NYC Airbnb nightly price estimator — powered by a machine learning model trained on 20,700+ real listings with NLP review sentiment analysis.

**Live demo:** [https://listinglens.vercel.app](https://listinglens.vercel.app)

---

## What It Does

ListingLens takes Airbnb listing details — neighbourhood, room type, amenities, host status, and optionally real guest review sentiment — and returns a data-driven nightly price estimate with:

- Predicted price range
- Borough comparison
- Top price drivers
- Personalised optimisation tips

The **Review Analysis** tool lets you inspect the review quality of any NYC Airbnb listing — sentiment breakdown, quality score, topic themes, and plain-language insights.

---

## Tech Stack & Architecture

This application uses a **split-stack architecture** to bypass serverless environment limitations:

| Layer | Technology | Deployment |
|---|---|---|
| **Frontend** | Next.js 15 (App Router), TypeScript | Vercel |
| **Backend API** | Python, Flask, Gunicorn | Render |
| **ML Model** | Local XGBoost inference (`predict_api.py`) | Render |
| **Database** | Precomputed SQLite DB (`reviews_summary.db`) | Render |
| **NLP** | VaderSentiment (Pre-processing only) | Local |

---

## Project Structure

```
airbnb-price-intelligence/
├── app/                  # Next.js 15 frontend
│   ├── app/              # Pages (/, /predict, /reviews)
│   ├── components/       # InputForm, PredictionDisplay, ReviewsDashboard
│   ├── lib/              # listingsDirectory.json (25k+ listings)
│   └── predict_api.py    # Python ML inference script
├── src/                  # ML pipeline source
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model_utils.py
├── models/               # Trained model artifacts (~2MB total)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── kmeans_geo.pkl
│   ├── feature_names.json
│   └── model_metadata.json
├── data/
│   └── raw/              # Raw CSVs excluded from git (see .gitignore)
├── requirements.txt
└── vercel.json
```

---

## Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and run Next.js frontend
cd app
npm install
npm run dev
```

App runs at `http://localhost:3000`

### Windows shell tips
- If PowerShell blocks npm or venv scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` in the current shell.
- Prefer `python` (or set `PYTHON_CMD=python`) because `python3` is not available on Windows by default.
- Use `npm.cmd` if `npm` is blocked by execution policy.

---

## Data

Data sourced from [Inside Airbnb](http://insideairbnb.com) — NYC snapshot, November 2025.  
Raw CSV files are excluded from the repository due to size (400MB+). Model artifacts are included and sufficient to run inference.

---

## Deployment Architecture

Due to Vercel's serverless environment constraints (50MB size limit, missing Python runtime) the application uses a **distributed architecture** in production.

1. **Frontend (Vercel)** `https://listinglens.vercel.app`
    - Next.js application handling UI and routing.
    - API routes (`/api/predict` and `/api/analyze-reviews`) check for the `PYTHON_API_URL` environment variable.
2. **Backend (Render)** `https://listinglens-ru9r.onrender.com`
    - A lightweight Flask server (`app/server.py`) serving as a REST API.
    - Executes the ML inference using the trained XGBoost `.pkl` models.
    - Queries `models/reviews_summary.db` for instant NLP insights, avoiding the need to process the 314MB raw CSV in production.

**Local Fallback:** In development (when `PYTHON_API_URL` is undefined), the Next.js app automatically falls back to spawning a local `python3` subprocess to execute `predict_api.py`.
