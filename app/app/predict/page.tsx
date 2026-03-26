"use client";
import { useState } from "react";
import InputForm from "@/components/InputForm";
import PredictionDisplay from "@/components/PredictionDisplay";
import { PredictionResult, FormData } from "@/types";

const NAV = [
  { href: "/",        label: "About",            active: false },
  { href: "/predict", label: "Price Estimator",  active: true  },
  { href: "/reviews", label: "Review Analysis",  active: false },
];

export default function PredictPage() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string[]>([]);
  const [formData, setFormData]     = useState<FormData | null>(null);

  const handlePredict = async (data: FormData) => {
    setLoading(true); setError(null); setErrorDetails([]); setFormData(data);
    try {
      const res  = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
      const json = await res.json();
      if (!res.ok || json.error) {
        if (json.details && Array.isArray(json.details)) {
          setErrorDetails(json.details);
        }
        throw new Error(json.error || "Prediction failed");
      }
      setPrediction(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally { setLoading(false); }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <a href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(180deg, var(--primary) 0%, var(--primary-dark) 100%)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em", boxShadow: "0 1px 4px rgba(37,99,235,0.25)" }}>LL</div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>ListingLens</span>
          </a>
          <nav style={{ display: "flex", alignItems: "center", gap: 2 }}>
            {NAV.map(n => <a key={n.href} href={n.href} className={`nav-link ${n.active ? "active" : ""}`}>{n.label}</a>)}
            <div style={{ width: 1, height: 16, background: "var(--ghost-border)", margin: "0 10px" }} />
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="nav-link">GitHub</a>
          </nav>
          <div style={{ display: "flex", alignItems: "center", gap: 7, background: "var(--success-dim)", borderRadius: 99, padding: "5px 12px" }}>
            <span className="live-dot" />
            <span style={{ fontSize: 12, fontWeight: 500, color: "var(--success)" }}>Model live</span>
          </div>
        </div>
      </header>

      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 32px 36px" }}>
          <h1 className="page-title" style={{ marginBottom: 10, maxWidth: 560 }}>NYC Airbnb Price Estimator</h1>
          <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.6, maxWidth: 500 }}>
            Fill in your listing details and get an accurate nightly price estimate for any NYC property in seconds.
            Add a Listing ID to factor in your real guest review quality.
          </p>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 32px 64px" }}>
          <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24, alignItems: "start" }}>
            <div className="card">
              <div className="card-header">
                <span className="card-header-title">Listing Details</span>
              </div>
              <InputForm onSubmit={handlePredict} loading={loading} />
            </div>

            <div>
              {error && (
                <div style={{ padding: "14px 18px", borderRadius: 10, background: "var(--error-dim)", marginBottom: 16 }}>
                  <p style={{ fontSize: 13, color: "var(--error)", fontWeight: 500 }}>{error}</p>
                  {errorDetails.length > 0 && (
                    <ul style={{ fontSize: 12, color: "var(--error)", marginTop: 6, opacity: 0.8, paddingLeft: 18, marginBottom: 0 }}>
                      {errorDetails.map((detail, idx) => (
                        <li key={idx} style={{ marginBottom: 4 }}>{detail}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {prediction ? (
                <PredictionDisplay prediction={prediction} formData={formData} />
              ) : (
                <div className="card" style={{ padding: "64px 32px", textAlign: "center" }}>
                  <div style={{ width: 48, height: 48, borderRadius: 12, background: "var(--surface-low)", margin: "0 auto 20px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                      <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
                    </svg>
                  </div>
                  <p style={{ fontSize: 15, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>Configure your listing to see a price estimate</p>
                  <p style={{ fontSize: 13, color: "var(--text-muted)", maxWidth: 320, margin: "0 auto", lineHeight: 1.6 }}>
                    Results include a price range, neighbourhood comparison, and tips to increase your nightly rate.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer style={{ background: "var(--surface)", borderTop: "1px solid var(--ghost-border)", padding: "20px 32px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)" }}>
          <span>ListingLens — NYC Airbnb Price Intelligence</span>
          <span>Data: <a href="http://insideairbnb.com" style={{ color: "var(--primary)", textDecoration: "none" }}>Inside Airbnb</a> · Nov 2025</span>
        </div>
      </footer>
    </div>
  );
}
