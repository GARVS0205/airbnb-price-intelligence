"use client";
import ReviewsDashboard from "../../components/ReviewsDashboard";
import NavBar from "@/components/NavBar";

const NAV = [
  { href: "/",        label: "About",           active: false },
  { href: "/predict", label: "Price Estimator", active: false },
  { href: "/reviews", label: "Review Analysis",  active: true  },
];

export default function ReviewsPage() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <NavBar nav={NAV} showModelLive />

      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)" }}>
        <div className="page-pad" style={{ maxWidth: 1200, margin: "0 auto", paddingTop: 40, paddingBottom: 36 }}>
          <h1 className="page-title" style={{ marginBottom: 10 }}>Review Analysis</h1>
          <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.6, maxWidth: 500 }}>
            Select any NYC Airbnb listing to see a breakdown of what guests say: overall sentiment,
            common topics, quality score, and recent review excerpts.
          </p>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)" }}>
        <div className="page-pad" style={{ maxWidth: 1200, margin: "0 auto", paddingTop: 32, paddingBottom: 64 }}>
          <ReviewsDashboard />
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
