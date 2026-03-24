"use client";
import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const MODELS = [
  { name: "Ridge Regression",  r2: 0.653, mape: 34.4, best: false },
  { name: "Random Forest",     r2: 0.785, mape: 24.4, best: false },
  { name: "LightGBM",         r2: 0.806, mape: 23.2, best: false },
  { name: "XGBoost (Tuned)",  r2: 0.812, mape: 22.6, best: true  },
];

const KEY_METRICS = [
  { value: "0.812", label: "R² Score",     sub: "Explains 81.2% of price variance" },
  { value: "22.6%", label: "MAPE",         sub: "Mean absolute % error on test set" },
  { value: "20.7k", label: "Training rows", sub: "Clean NYC listings Nov 2025" },
  { value: "46",    label: "Features",     sub: "Engineered from raw listing data"  },
];

const TT = { background: "#0c1120", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, color: "#e2e8f0", fontSize: 12 };

export default function ModelMetrics() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Headline ── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <p className="section-label" style={{ marginBottom: 4 }}>Model Performance</p>
          <h3 style={{ fontSize: 16, fontWeight: 600, color: "var(--text-primary)" }}>
            Algorithm Benchmark · NYC Test Set (20% holdout)
          </h3>
        </div>
        <span className="badge badge-emerald">XGBoost Selected</span>
      </div>

      {/* ── Key stat cards ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
        {KEY_METRICS.map(m => (
          <div key={m.label} className="stat-card">
            <div className="mono stat-value gradient-text">{m.value}</div>
            <div className="stat-label">{m.label}</div>
            <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 3 }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* ── Model comparison ── */}
      <div className="card">
        <div className="card-header">
          <span className="card-header-title">R² Score Comparison</span>
          <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>Higher is better · Range: 0–1</span>
        </div>
        <div className="card-body">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {MODELS.map(m => (
              <div key={m.name} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 130, fontSize: 12, color: m.best ? "var(--text-primary)" : "var(--text-secondary)", fontWeight: m.best ? 600 : 400, flexShrink: 0 }}>
                  {m.name}
                </div>
                <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 10 }}>
                  <div className="progress-track" style={{ flex: 1 }}>
                    <div
                      style={{
                        height: "100%", borderRadius: 99,
                        width: `${m.r2 * 100}%`,
                        background: m.best
                          ? "linear-gradient(90deg, var(--emerald), var(--emerald-light))"
                          : "rgba(255,255,255,0.1)",
                        transition: "width 0.8s ease",
                        boxShadow: m.best ? "0 0 8px rgba(16,185,129,0.4)" : "none",
                      }}
                    />
                  </div>
                  <span className="mono" style={{ fontSize: 11, color: m.best ? "var(--emerald)" : "var(--text-muted)", width: 40, textAlign: "right" }}>
                    {m.r2}
                  </span>
                </div>
                <div style={{ width: 60, fontSize: 11, color: "var(--text-muted)", textAlign: "right" }}>
                  MAPE {m.mape}%
                </div>
                {m.best && <span className="badge badge-emerald" style={{ fontSize: 9.5, padding: "2px 7px" }}>Best</span>}
              </div>
            ))}
          </div>
          <p style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
            XGBoost selected after RandomizedSearchCV tuning: 20 iterations × 5-fold CV.
            Best params: n_estimators=700 · max_depth=7 · learning_rate=0.05
          </p>
        </div>
      </div>

    </div>
  );
}
