# ListingLens — Project Health Report

## Architecture Overview

```
User Browser
    │
    ▼
Vercel (Next.js 15)           ← Frontend + UI
  /predict → /api/predict     → Render Flask (Python)
  /reviews → /api/analyze-reviews → Render Flask
  /api/ping                   → Render /health (wake-up)
          │
          ▼
  Render (Flask / Gunicorn)   ← Python ML Backend
    /health        (wake-up probe)
    /predict       → predict_api.run_inference()
    /analyze-reviews → review_analysis_api.run_analysis()
```

---

## ✅ All Systems Verified

### Backend (Render Flask)
| File | Status | Notes |
|------|--------|-------|
| `app/server.py` | ✅ Correct | Flask API with 3 clean routes |
| `app/predict_api.py` | ✅ Correct | XGBoost model + engineered features + SQLite NLP |
| `app/review_analysis_api.py` | ✅ Correct | Reads precomputed VADER/SQLite data |
| `app/precompute_reviews.py` | ✅ Run once | Generates `models/reviews_summary.db` (82MB) |
| `app/models/reviews_summary.db` | ✅ In git | Tracked, 82MB, queried at runtime |

### Frontend (Vercel Next.js)
| File | Status | Notes |
|------|--------|-------|
| `app/app/api/predict/route.ts` | ✅ Correct | `maxDuration=80`, proxies to Render |
| `app/app/api/analyze-reviews/route.ts` | ✅ Correct | `maxDuration=80`, proxies to Render |
| `app/app/api/ping/route.ts` | ✅ Correct | `maxDuration=80`, wake-up ping |
| `app/components/InputForm.tsx` | ✅ Correct | Passes all 37 features + listing_id |
| `app/components/ReviewsDashboard.tsx` | ✅ Correct | Fires ping on mount, queries `/api/analyze-reviews` |
| `app/app/predict/page.tsx` | ✅ Correct | Fires ping on mount, shows per-field validation errors |

---

## ✅ Local & Production Test Results

### Price Prediction (XGBoost, R² = 0.809)
* **Bug Fixed:** Previously, the `predict_api` ran features through a `RobustScaler` before passing them to the XGBoost model. However, XGBoost was trained on raw DataFrames. This mismatch crushed the target-encoded `room_type_encoded` values down to zero or negative margins, preventing the model from recognizing room types. 
* Removing the scaler transformation inside `predict_api.py` perfectly restored differentiation.

**Production QA Verification (listinglens-phi.vercel.app):**
| Config | Price |
|--------|-------|
| Entire home / apartment, 2 guests, 1 bed | **$189** |
| Private room, 2 guests, 1 bed | **$87** |

![Prediction Entire Home](C:\Users\garvs\.gemini\antigravity\brain\f74dde5a-0282-4c37-9f6d-73fb2a72168d\entire_home.png)
![Prediction Private Room](C:\Users\garvs\.gemini\antigravity\brain\f74dde5a-0282-4c37-9f6d-73fb2a72168d\private_room.png)

### Review Analysis & HTML Sanitization
* **Bug Fixed:** Review excerpts previously rendered literal `<br/>` HTML tags in the UI. 
* Added regex tag stripping to `precompute_reviews.py` to clean all strings before inserting them into `reviews_summary.db`.

**Production QA Verification:**
- Verified live on "Untitled at 3 Freeman" (25,007 rows processed).
- Text snippets render cleanly without HTML injection.
![Review Analysis Clean](C:\Users\garvs\.gemini\antigravity\brain\f74dde5a-0282-4c37-9f6d-73fb2a72168d\clean_reviews.png)

---

## ✅ All Commits Pushed

| Commit | What |
|--------|------|
| `c16cf84` | fix(ml): Use raw dataframe for XGBoost inference to fix scaler flattening encoded room types<br/>feat(data): Strip HTML tags from review database<br/>chor(pipeline): Retrain models to match encoding strategies end-to-end |
| `2e30151` | Wake-up ping on page load (battle Render cold starts) |
| `dfaa04f` | Compute 17 missing interaction features in `predict_api.py` |
| `40be51a` | User-friendly validation error messages |
| `60f3ff6` | Fix SQLite DB path in `predict_api.py` |
| `a16e65a` | Increase `maxDuration` to 60s |
| `ba4d19e` | Precomputed SQLite DB (`reviews_summary.db`) |

---

## Known Limitations

- **Render Free Tier Cold Start:** First request after 15min idle may take ~50s to respond. The wake-up ping mitigates this — as long as the user stays on the page for ~15s before clicking Predict, Render will be warm.
- **maxDuration = 80s now set** on all 3 Vercel API routes. This is the Vercel Pro tier limit.

---

## Key Env Variables Required

| Variable | Set On | Value |
|---------|--------|-------|
| `PYTHON_API_URL` | Vercel | `https://listinglens-ru9r.onrender.com` |
