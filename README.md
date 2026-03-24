# ListingLens

> NYC Airbnb nightly price estimator — powered by a machine learning model trained on 20,700+ real listings with NLP review sentiment analysis.

**Live demo:** _add Vercel URL here after deploy_

---

## What It Does

ListingLens takes Airbnb listing details — neighbourhood, room type, amenities, host status, and optionally real guest review sentiment — and returns a data-driven nightly price estimate with:

- Predicted price range
- Borough comparison
- Top price drivers
- Personalised optimisation tips

The **Review Analysis** tool lets you inspect the review quality of any NYC Airbnb listing — sentiment breakdown, quality score, topic themes, and plain-language insights.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15 (App Router), TypeScript |
| Styling | Vanilla CSS with custom design system |
| ML Model | XGBoost (trained offline, loaded at inference time) |
| NLP | VADER sentiment analysis on 700k+ reviews |
| API | Next.js API routes spawn Python subprocesses |
| Deployment | Vercel |

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

---

## Data

Data sourced from [Inside Airbnb](http://insideairbnb.com) — NYC snapshot, November 2025.  
Raw CSV files are excluded from the repository due to size (400MB+). Model artifacts are included and sufficient to run inference.

---

## Deployment

This project is configured for **Vercel**. The `vercel.json` at the root handles the monorepo layout (Next.js app in `app/` subdirectory).

> **Note:** Vercel's serverless functions spawn Python subprocesses for ML inference. Python and required packages must be available in the Vercel build environment. See `requirements.txt`.
