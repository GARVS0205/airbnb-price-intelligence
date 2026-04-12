<div align="center">
  <h1>ListingLens — NYC Airbnb Price Intelligence</h1>
  <p><strong>End-to-end Machine Learning Application for Short-Term Rental Pricing & NLP Review Analysis</strong></p>

  [![Live Demo](https://img.shields.io/badge/Live%20Demo-listinglens--phi.vercel.app-blue?style=for-the-badge)](https://listinglens-phi.vercel.app)
  [![Deployed on](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com)
  [![Model R²](https://img.shields.io/badge/Model%20R²-0.809-success?style=for-the-badge)](#model-performance)

  <p>
    An intelligent pricing tool that estimates nightly Airbnb rates for any NYC property using <b>XGBoost</b>, <b>NLP sentiment analysis</b>, and <b>52 engineered features</b> trained on 20,700+ real listings.
  </p>
</div>

---

## 🌟 Key Features

- 🎯 **Accurate Nightly Pricing**: Predict prices with an XGBoost model (R² = 0.809) factoring in room type, location, amenities, and host quality.
- 📊 **Feature Explainability**: Understand *why* a price was predicted. See the top features driving the price via SHAP-inspired impact weights.
- 🧠 **NLP Guest Sentiment**: Integrate actual guest reviews. Enter an Airbnb Listing ID and the model will analyze review language (via VADER) to adjust the price estimate based on real quality signals.
- 📈 **Review Dashboard**: A dedicated tool to explore sentiment distribution, review quality scores (0-100), and topic themes (e.g., cleanliness, location, host) for any NYC listing.
- 📱 **Fully Responsive UI**: A modern, mobile-friendly interface built with Next.js 15, Recharts, and custom CSS variables.
- ⚡ **Instant Predictions**: ML inference runs directly on Vercel via a pure TypeScript XGBoost tree-walker — no cold starts, no external backend.

---

## 🏗️ Architecture

ListingLens runs **entirely on Vercel** with zero external backend dependencies. The XGBoost model is converted to its native JSON format and inference is performed by a pure TypeScript decision-tree walker — no Python, no native binaries, no cold starts.

```mermaid
graph TD
    Client([User Browser]) -->|HTTP| Vercel[Vercel Edge]

    subgraph Vercel ["Vercel — Single Platform"]
        Vercel --> NextJS[Next.js Frontend]
        NextJS -->|/api/predict| PredictFn["TypeScript XGBoost\n(tree walker, model.json)"]
        NextJS -->|/api/analyze-reviews| ReviewsFn["better-sqlite3\n(reviews_summary.db)"]
    end
```

| Layer | Technology | Platform |
|---|---|---|
| **Frontend** | Next.js 15, TypeScript, React, Recharts | Vercel |
| **ML Inference** | Pure TypeScript XGBoost tree-walker | Vercel (serverless function) |
| **Model** | XGBoost JSON (700 trees, 52 features, R²=0.809) | Bundled in deployment |
| **Review Data** | Precomputed SQLite DB (better-sqlite3) | Bundled in deployment |
| **Data Source** | Inside Airbnb — NYC Nov 2025 (20.7k listings) | — |

---

## 🔬 Machine Learning Pipeline

The core ML engine was built with a reproducible pipeline (`run_pipeline.py`) structured in three phases:

### 1. Feature Engineering (52 total features)
- **Geographic**: Target encoding for neighbourhoods, Haversine distances to 5 major NYC landmarks, and K-Means spatial clustering (8 clusters).
- **Listing Details**: Room type, capacity, bedrooms, bathrooms, and boolean flags for premium amenities.
- **NLP Sentiment**: Precomputed VADER sentiment metrics (positive/negative %) and a proprietary 100-point Review Quality Score.
- **Interactions**: Engineered terms like `luxury_x_capacity` and `capacity_x_bedrooms`.

### 2. Model Selection & Tuning
- Tested Ridge Regression (Baseline), Random Forest, LightGBM, and XGBoost.
- **XGBoost** performed best. Hyperparameters were optimized using `RandomizedSearchCV` across 100 fits (20 configs × 5-fold CV).
- Prevented data leakage by splitting the train/test sets *before* applying neighborhood target encoding.

### 3. Model Performance
| Metric | Result | Context |
|---|---|---|
| **R² Score** | `0.809` | The model explains ~81% of pricing variance. |
| **MAPE** | `22.8%` | Average error margin (competitive for volatile real estate pricing). |
| **Top Driver** | `Room Type` | Accounts for 43.5% of model gain importance. |

---

## 🛠️ Local Setup & Development

### Prerequisites
- Node.js 18+
- Python 3.10+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GARVS0205/airbnb-price-intelligence.git
   cd airbnb-price-intelligence
   ```

2. **Install Python dependencies** *(for training/experimentation only — not needed to run the app):*
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Frontend dependencies & Run:**
   ```bash
   cd app
   npm install
   npm run dev
   ```

4. **View the app:**
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🚀 Deployment

The entire application deploys to **Vercel** from a single push. No external services required.

1. Import the repo into [Vercel](https://vercel.com)
2. Set **Root Directory** to `app/`
3. Deploy — that's it. No environment variables needed.

> **How ML works on Vercel:** The XGBoost model is stored as `app/models/model.json` (native XGBoost JSON, 6.2 MB). A pure TypeScript function in `app/api/predict/route.ts` walks the 700 decision trees at request time — achieving ~50ms predictions with zero native dependencies.

---

## 📁 Repository Structure

```text
airbnb-price-intelligence/
├── app/                        # Next.js Application (deployed to Vercel)
│   ├── app/                    # React Pages & API Routes
│   │   ├── api/predict/        # TypeScript XGBoost inference (no Python)
│   │   ├── api/analyze-reviews/# SQLite review lookup (better-sqlite3)
│   │   ├── predict/            # Price Estimator page
│   │   └── reviews/            # Review Analysis page
│   ├── components/             # Reusable UI Components
│   └── models/                 # ML artifacts
│       ├── model.json          # XGBoost (700 trees, 52 features) — 6.2 MB
│       ├── feature_names.json  # Feature order & metadata
│       └── reviews_summary.db  # Precomputed review analysis — 86 MB
├── src/                        # ML Training Source Code
│   ├── data_preprocessing.py   # Cleaning & imputation
│   └── feature_engineering.py  # Geographic & sentiment feature creation
├── run_pipeline.py             # End-to-end model training script
├── convert_model_to_onnx.py    # Model export utility
└── requirements.txt            # Python dependencies (training only)
```

---

<div align="center">
  <p>Built with ❤️ for data science and web engineering.</p>
</div>
