"use client";
import { useState, useRef } from "react";
import InputForm from "@/components/InputForm";
import PredictionDisplay from "@/components/PredictionDisplay";
import NavBar from "@/components/NavBar";
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

  const resultsRef = useRef<HTMLDivElement>(null);

  const handlePredict = async (data: FormData) => {
    setLoading(true);
    setError(null);
    setErrorDetails([]);
    setFormData(data);
    try {
      const res  = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (!res.ok || json.error) {
        if (json.details && Array.isArray(json.details)) setErrorDetails(json.details);
        throw new Error(json.error || "Prediction failed");
      }
      setPrediction(json);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 80);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <NavBar nav={NAV} showModelLive />

      {/* Page hero */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)" }}>
        <div className="page-pad" style={{ maxWidth: 1200, margin: "0 auto", paddingTop: 40, paddingBottom: 36 }}>
          <h1 className="page-title" style={{ marginBottom: 10, maxWidth: 560 }}>NYC Airbnb Price Estimator</h1>
          <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.6, maxWidth: 500 }}>
            Fill in your listing details and get an accurate nightly price estimate powered by machine learning.
            Add a Listing ID to factor in your real guest review quality.
          </p>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)" }}>
        <div className="page-pad" style={{ maxWidth: 1200, margin: "0 auto", paddingTop: 32, paddingBottom: 64 }}>

          {/* Loading indicator */}
          {loading && (
            <div style={{ marginBottom: 16, padding: "10px 16px", borderRadius: 8, background: "var(--primary-dim)", fontSize: 12, color: "var(--primary)", display: "flex", alignItems: "center", gap: 8 }}>
              <span className="spinner" style={{ borderColor: "rgba(37,99,235,0.3)", borderTopColor: "var(--primary)" }} />
              <span>Running ML inference…</span>
            </div>
          )}

          <div className="sidebar-layout">
            <div className="card">
              <div className="card-header">
                <span className="card-header-title">Listing Details</span>
                <span style={{ fontSize: 11, color: "var(--success)", fontWeight: 500, display: "flex", alignItems: "center", gap: 4 }}>
                  <span className="live-dot" style={{ width: 6, height: 6 }} /> Ready
                </span>
              </div>
              <InputForm onSubmit={handlePredict} loading={loading} />
            </div>

            <div ref={resultsRef}>
              {error && (
                <div style={{ padding: "14px 18px", borderRadius: 10, background: "var(--error-dim)", marginBottom: 16 }}>
                  <p style={{ fontSize: 13, color: "var(--error)", fontWeight: 500 }}>{error}</p>
                  {errorDetails.length > 0 && (
                    <ul style={{ fontSize: 12, color: "var(--error)", marginTop: 6, opacity: 0.8, paddingLeft: 18, marginBottom: 0 }}>
                      {errorDetails.map((d, i) => <li key={i} style={{ marginBottom: 4 }}>{d}</li>)}
                    </ul>
                  )}
                </div>
              )}
              {prediction ? (
                <PredictionDisplay prediction={prediction} formData={formData} />
              ) : (
                <div className="card" style={{ padding: "48px 24px", textAlign: "center" }}>
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

      <footer style={{ background: "var(--surface)", borderTop: "1px solid var(--ghost-border)", padding: "20px 0" }}>
        <div className="page-pad footer-inner" style={{ maxWidth: 1200, margin: "0 auto" }}>
          <span>ListingLens — NYC Airbnb Price Intelligence</span>
          <span>Data: <a href="http://insideairbnb.com" style={{ color: "var(--primary)", textDecoration: "none" }}>Inside Airbnb</a> · Nov 2025</span>
        </div>
      </footer>
    </div>
  );
}
