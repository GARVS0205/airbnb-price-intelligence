"use client";
import { useState } from "react";
import NavBar from "@/components/NavBar";

const NAV = [
  { href: "/",        label: "Price Estimator", active: false },
  { href: "/reviews", label: "Review Analysis",  active: false },
  { href: "/about",   label: "About",            active: true  },
];

const SKILLS = [
  { tag: "ML",  title: "Machine Learning",      color: "var(--primary)",   points: ["4 algorithms compared via 5-fold cross-validation", "XGBoost tuned with RandomizedSearchCV (100 model fits)", "Log-transform target to handle skewed price distribution", "Train/test split before any feature encoding — no leakage"] },
  { tag: "NLP", title: "NLP & Text Analysis",   color: "#7c3aed",          points: ["VADER sentiment on 700k+ real guest reviews", "6 per-listing NLP features baked into the model", "7-category theme detection (cleanliness, value, location…)", "Review quality score 0–100, sentiment trend tracking"] },
  { tag: "FE",  title: "Feature Engineering",   color: "var(--warning)",   points: ["Haversine distances to 5 NYC landmarks", "K-Means geo-clustering (8 clusters)", "Target encoding for 400+ neighbourhoods", "Interaction terms: capacity × bedrooms, premium × capacity"] },
  { tag: "XAI", title: "Explainability",         color: "var(--success)",   points: ["SHAP Shapley values for every prediction", "Identifies which features drove each price estimate", "Top driver: neighbourhood (~35% of prediction variance)", "Transparent model — not a black box"] },
  { tag: "FS",  title: "Full-Stack Engineering", color: "#0891b2",          points: ["Next.js 15 App Router + TypeScript", "Python ML backend via subprocess — no separate server", "Input validation and structured JSON error handling", "Serverless-compatible for Vercel deployment"] },
  { tag: "DS",  title: "Data Science Rigour",    color: "#be185d",          points: ["Residual analysis confirms unbiased predictions", "Learning curves verify no overfitting", "Ghost-border data tables for high-density display", "Honest limitations documented — MAPE 22.8% is real"] },
];

const MODELS = [
  { name: "Ridge Regression",  r2: 0.653, mape: "34.4%", bar: 65,  tag: "baseline" },
  { name: "Random Forest",     r2: 0.784, mape: "24.5%", bar: 78,  tag: ""         },
  { name: "LightGBM",          r2: 0.804, mape: "23.3%", bar: 80,  tag: ""         },
  { name: "XGBoost (Tuned)",   r2: 0.809, mape: "22.8%", bar: 81,  tag: "best"     },
];

const PIPELINE = [
  { n: "01", t: "Data Collection",     d: "36,000 real NYC Airbnb listings from Inside Airbnb (Nov 2025 snapshot). 700k+ guest reviews. No synthetic data." },
  { n: "02", t: "Cleaning & EDA",      d: "Price string parsing, IQR outlier removal, missing value imputation by room type group. 20,760 quality rows retained." },
  { n: "03", t: "Feature Engineering", d: "52 features: landmark distances, K-Means geo-clusters, amenity flags, host quality, 6 NLP sentiment features from reviews.csv." },
  { n: "04", t: "Model Training",      d: "4 algorithms compared via 5-fold CV. XGBoost selected and tuned with RandomizedSearchCV — 20 configs × 5 folds = 100 fits." },
  { n: "05", t: "NLP Integration",     d: "VADER run on 700k reviews. Per-listing aggregates (avg sentiment, positive %, quality score, trend) fed directly into XGBoost." },
  { n: "06", t: "SHAP Explainability", d: "Shapley values on all test predictions. Each estimate has a reason — which features pulled price up or down." },
  { n: "07", t: "Web App",             d: "Next.js API calls Python subprocess at inference. No separate ML server. Serverless-compatible. Input validation built in." },
];

const FAQ = [
  { q: "Why is this impressive for a portfolio?", a: "Most ML portfolios are a notebook with one model. This has two integrated phases (price + NLP), a production API with validation, SHAP explainability, and a deployed full-stack web app. It demonstrates the full engineering loop — not just model.fit()." },
  { q: "What does R² = 0.809 mean?", a: "The model explains 80.9% of price variation across NYC Airbnb listings. The remaining ~19% is things the data doesn't capture — photo quality, how the description is written, seasonal spikes. 80%+ is strong for real-world pricing data." },
  { q: "Why is MAPE 22.8% and not lower?", a: "Airbnb pricing is hard. A $200/night listing could be in Williamsburg or Midtown for completely different reasons. Missing signals (photo quality, host branding) cap any model's accuracy. 22.8% on real data is competitive with published research." },
  { q: "What does the NLP actually add?", a: "Reviews reveal what guests actually think — not what the host claims. 30% negative-sentiment reviews at 4.7 stars is very different from 90% positive at 4.7 stars. XGBoost learned that sentiment correlates with price." },
  { q: "How does it work on Vercel without a Python server?", a: "The Next.js API route spawns a Python subprocess, pipes the feature JSON to stdin, reads the prediction from stdout. Zero infrastructure — no Flask, no FastAPI. The model runs inside the serverless function." },
];

export default function AboutPage() {
  const [activeQ, setActiveQ] = useState<number | null>(null);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>

      <NavBar nav={NAV} showModelLive />

      {/* Hero band */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)" }}>
        <div className="page-pad hero-section" style={{ maxWidth: 900, margin: "0 auto", paddingTop: 48, paddingBottom: 44, textAlign: "center" }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 16, justifyContent: "center" }}>
            <span className="badge badge-primary">ML Portfolio Project</span>
            <span className="badge badge-neutral">Phase 1: Price Prediction</span>
            <span className="badge badge-neutral">Phase 2: NLP Reviews</span>
          </div>
          <h1 className="page-title" style={{ marginBottom: 14, fontSize: 36 }}>What This Project Demonstrates</h1>
          <p style={{ fontSize: 15, color: "var(--text-muted)", lineHeight: 1.7, maxWidth: 600, margin: "0 auto 32px" }}>
            An end-to-end machine learning system — from raw data to deployed web app — that predicts NYC Airbnb nightly prices using 52 features including real guest review sentiment.
          </p>

          {/* Stats row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12, maxWidth: 680, margin: "0 auto" }}>
            {[
              { v: "0.809", l: "R² Score",      s: "81% of price variance explained" },
              { v: "22.8%", l: "MAPE",           s: "Mean absolute % error on test set" },
              { v: "52",    l: "Features",       s: "Incl. 6 from NLP review analysis" },
              { v: "700k+", l: "Reviews",        s: "VADER-analyzed with sentiment" },
            ].map(s => (
              <div key={s.l} style={{ padding: "20px 14px", borderRadius: 10, background: "var(--surface-low)", textAlign: "center" }}>
                <div style={{ fontSize: 26, fontWeight: 700, color: "var(--primary)", letterSpacing: "-0.02em", lineHeight: 1 }}>{s.v}</div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", margin: "6px 0 3px" }}>{s.l}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{s.s}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)" }}>
        <div className="page-pad" style={{ maxWidth: 1100, margin: "0 auto", paddingTop: 48, paddingBottom: 80, display: "flex", flexDirection: "column", gap: 48 }}>

          {/* Skills grid */}
          <section>
            <h2 className="section-title" style={{ marginBottom: 20 }}>Skills Demonstrated</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              {SKILLS.map(sk => (
                <div key={sk.title} className="card">
                  <div style={{ padding: "20px 22px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: sk.color, fontFamily: "monospace", background: `${sk.color}15`, padding: "3px 8px", borderRadius: 5, letterSpacing: "0.04em" }}>{sk.tag}</span>
                      <span style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)" }}>{sk.title}</span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                      {sk.points.map(p => (
                        <div key={p} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                          <div style={{ width: 4, height: 4, borderRadius: "50%", background: sk.color, flexShrink: 0, marginTop: 6 }} />
                          <span style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.5 }}>{p}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Model comparison */}
          <section>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 20 }}>
              <h2 className="section-title">Model Comparison</h2>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>20% holdout test set · lower MAPE = better</span>
            </div>
            <div className="card">
              <div style={{ padding: "24px" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                  {MODELS.map(m => (
                    <div key={m.name}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <span style={{ fontSize: 13.5, fontWeight: m.tag === "best" ? 600 : 400, color: m.tag === "best" ? "var(--text-primary)" : "var(--text-secondary)" }}>{m.name}</span>
                          {m.tag === "best"     && <span className="badge badge-primary" style={{ fontSize: 10 }}>Selected</span>}
                          {m.tag === "baseline" && <span className="badge badge-neutral" style={{ fontSize: 10 }}>Baseline</span>}
                        </div>
                        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
                          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>MAPE <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>{m.mape}</span></span>
                          <span style={{ fontSize: 14, fontWeight: 700, color: m.tag === "best" ? "var(--primary)" : "var(--text-secondary)" }}>R² {m.r2.toFixed(3)}</span>
                        </div>
                      </div>
                      <div className="progress-track">
                        <div style={{ height: "100%", borderRadius: 99, width: `${m.bar}%`, background: m.tag === "best" ? "linear-gradient(90deg, var(--primary-dark), var(--primary))" : "var(--surface-high)", transition: "width 0.6s ease" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Pipeline */}
          <section>
            <h2 className="section-title" style={{ marginBottom: 20 }}>How It Was Built</h2>
            <div className="grid-2">
              {PIPELINE.map(p => (
                <div key={p.n} className="card">
                  <div style={{ padding: "20px 22px" }}>
                    <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                      <span style={{ fontSize: 12, fontWeight: 700, color: "var(--primary)", fontFamily: "monospace", flexShrink: 0, marginTop: 1, background: "var(--primary-dim)", padding: "2px 7px", borderRadius: 4 }}>{p.n}</span>
                      <div>
                        <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)", marginBottom: 5 }}>{p.t}</div>
                        <div style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.6 }}>{p.d}</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* FAQ */}
          <section>
            <h2 className="section-title" style={{ marginBottom: 20 }}>Questions Recruiters Ask</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {FAQ.map((f, i) => (
                <div key={i} className="card" style={{ cursor: "pointer" }} onClick={() => setActiveQ(activeQ === i ? null : i)}>
                  <div style={{ padding: "16px 22px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 14, fontWeight: 500, color: "var(--text-primary)" }}>{f.q}</span>
                    <span style={{ fontSize: 18, color: "var(--primary)", flexShrink: 0, marginLeft: 16, transform: activeQ === i ? "rotate(45deg)" : "none", transition: "transform 0.18s", display: "inline-block", fontWeight: 300 }}>+</span>
                  </div>
                  {activeQ === i && (
                    <div style={{ padding: "0 22px 18px" }}>
                      <div style={{ height: 1, background: "var(--ghost-border)", marginBottom: 14 }} />
                      <p style={{ fontSize: 13.5, color: "var(--text-muted)", lineHeight: 1.7, margin: 0 }}>{f.a}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

        </div>
      </main>

      <footer style={{ background: "var(--surface)", borderTop: "1px solid var(--ghost-border)", padding: "20px 0" }}>
        <div className="page-pad footer-inner" style={{ maxWidth: 1200, margin: "0 auto" }}>
          <span>ListingLens — NYC Airbnb Price Intelligence</span>
          <span>Data: <a href="http://insideairbnb.com" style={{ color: "var(--primary)", textDecoration: "none" }}>Inside Airbnb</a> · Nov 2025</span>
        </div>
      </footer>
    </div>
  );
}
