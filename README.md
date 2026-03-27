# ListingLens — NYC Airbnb Price Intelligence

> An end-to-end machine learning application that estimates nightly Airbnb prices for any NYC property using XGBoost, NLP review sentiment analysis, and 52 engineered features trained on 20,700+ real listings.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-listinglens--phi.vercel.app-blue?style=flat-square)](https://listinglens-phi.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-Render-orange?style=flat-square)](https://listinglens-ru9r.onrender.com/health)
[![Model Accuracy](https://img.shields.io/badge/Model%20R²-0.809-green?style=flat-square)](https://listinglens-phi.vercel.app)

---

## What It Does

**ListingLens** is a full-stack AI tool for NYC Airbnb hosts and investors. Enter your listing details and get:

- **Predicted nightly price** with confidence interval
- **Borough comparison** — see how you stack up against similar listings
- **Top price drivers** — understand which features most influence your price
- **NLP review sentiment** — enter any Listing ID to have real guest reviews factored into the prediction
- **Review Analysis** — inspect the sentiment, quality score, and topic themes of any NYC listing's reviews

---

## Architecture

This application uses a **split-stack architecture** because Vercel's serverless environment cannot run Python ML pipelines directly.

```
User Browser
     │
     ▼
┌─────────────────────────────────┐
│  Vercel (Next.js 15)            │  ← Frontend + API Proxy
│  listinglens-phi.vercel.app     │
│                                 │
│  /api/predict      ─────────────┼──┐
│  /api/analyze-reviews ──────────┼──┤
│  /api/ping (keep-alive) ────────┼──┤
└─────────────────────────────────┘  │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Render (Flask / Gunicorn)       │  ← Python ML Backend
                    │  listinglens-ru9r.onrender.com  │
                    │                                  │
                    │  /predict → XGBoost inference    │
                    │  /analyze-reviews → SQLite NLP   │
                    │  /health  → uptime probe         │
                    └─────────────────────────────────┘
```

| Layer | Technology | Platform |
|---|---|---|
| Frontend | Next.js 15, TypeScript, Recharts | Vercel |
| Backend API | Python 3.12, Flask, Gunicorn | Render |
| ML Model | XGBoost (R² = 0.809, 52 features) | Render |
| NLP Database | SQLite precomputed from VaderSentiment | Render (82MB) |
| Data Source | Inside Airbnb — NYC Nov 2025 (20,700+ listings) | — |

---

## ML Pipeline

The model was trained using a full, reproducible pipeline (`run_pipeline.py`):

1. **Data Preprocessing** — clean prices, handle nulls, parse amenities
2. **Feature Engineering** — 52 features across 7 categories:
   - Location: neighbourhood target encoding, distances to landmarks, geo clusters
   - Property: room type, capacity, beds/bathrooms, amenities
   - Host: superhost status, response rate, experience, quality score
   - Reviews: VADER sentiment, quality score, volume, recency
   - Interaction terms: `luxury_x_capacity`, `capacity_x_bedrooms`, `reviews_x_score`
   - NLP: precomputed from 500K+ reviews via `precompute_reviews.py`
3. **Model Selection** — XGBoost vs Linear Regression vs Random Forest (XGBoost wins, R²=0.809)
4. **Hyperparameter Tuning** — GridSearchCV on key XGBoost params

---

## Project Structure

```
airbnb-price-intelligence/
│
├── app/                        # Next.js 15 Frontend + Python Backend
│   ├── app/                    # App Router pages (/, /predict, /reviews)
│   │   └── api/                # API proxy routes
│   │       ├── predict/        # Forwards to Render /predict
│   │       ├── analyze-reviews/# Forwards to Render /analyze-reviews
│   │       └── ping/           # Keep-alive ping to prevent cold starts
│   ├── components/             # React components
│   │   ├── InputForm.tsx       # 37-feature listing form
│   │   ├── PredictionDisplay.tsx # Results + borough chart + tips
│   │   └── ReviewsDashboard.tsx  # Review sentiment explorer
│   ├── models/                 # Trained model artifacts (tracked in git)
│   │   ├── best_model.pkl      # XGBoost model (~1.5MB)
│   │   ├── scaler.pkl          # StandardScaler
│   │   ├── feature_names.json  # 52 feature names in order
│   │   └── reviews_summary.db  # Precomputed VADER NLP SQLite (82MB)
│   ├── server.py               # Flask API (3 routes: health/predict/analyze-reviews)
│   ├── predict_api.py          # XGBoost inference + engineered feature reconstruction
│   ├── review_analysis_api.py  # SQLite-based review NLP
│   ├── precompute_reviews.py   # One-off script: CSV → SQLite DB
│   └── vercel.json             # Vercel function timeout config (80s)
│
├── src/                        # ML pipeline source code
│   ├── data_preprocessing.py   # Data cleaning + price parsing
│   ├── feature_engineering.py  # 52 feature computation
│   └── model_utils.py          # Training, evaluation utilities
│
├── data/                       # Data directory (raw CSVs excluded from git)
│   └── raw/                    # listings.csv, reviews.csv (.gitignored, 400MB+)
│
├── run_pipeline.py             # End-to-end training pipeline
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/GARVS0205/airbnb-price-intelligence.git
cd airbnb-price-intelligence

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the Next.js frontend (development mode — uses local Python subprocess)
cd app
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

> **Note:** In development mode, the Next.js API routes spawn a local Python subprocess, so no separate Flask server is needed. The `PYTHON_API_URL` env variable is only required in production.

### Environment Variables (Production / Vercel)

| Variable | Value |
|---|---|
| `PYTHON_API_URL` | `https://listinglens-ru9r.onrender.com` |

---

## Deployment

### Frontend → Vercel

1. Connect your GitHub repo to Vercel.
2. Set **Root Directory** to `app/`.
3. Add the `PYTHON_API_URL` environment variable.

### Backend → Render

1. Create a new **Web Service** on Render.
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn -w 2 -b 0.0.0.0:$PORT server:app --timeout 120`
4. Root directory: `app/`

### Keeping Render Alive (Recommended)

Render's free tier spins down after 15 minutes of inactivity, causing a ~50s cold start.

**To eliminate cold starts permanently:**
1. Go to [cron-job.org](https://cron-job.org) (free account)
2. Create a cron job: `GET https://listinglens-ru9r.onrender.com/health`
3. Set interval: **every 5 minutes**

This keeps the Render server always warm. No cold starts, no timeouts.

---

## Model Performance

| Metric | Value |
|---|---|
| R² Score | **0.809** |
| RMSE | ~$38 |
| Training samples | 20,700+ listings |
| Features | 52 (including 6 NLP features) |
| Algorithm | XGBoost (with StandardScaler) |

**Top Price Drivers (XGBoost gain importance):**

1. Room Type (43.5%)
2. Guest Capacity (6.6%)
3. Neighbourhood (5.0%)
4. Luxury × Capacity interaction (4.9%)
5. Accommodates² (4.5%)

---

## Data

- Source: [Inside Airbnb](http://insideairbnb.com) — NYC snapshot, November 2025
- Raw CSV files are **excluded from git** (400MB+, see `.gitignore`)
- Model artifacts and the precomputed SQLite DB are tracked and sufficient to run inference

---

## Tech Stack

**Backend Pipeline:** Python · XGBoost · scikit-learn · VaderSentiment · SQLite · Flask · Gunicorn  
**Frontend:** Next.js 15 · TypeScript · Recharts · CSS Variables  
**Infrastructure:** Vercel (frontend) · Render (Python ML API)
