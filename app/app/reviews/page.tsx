import ReviewsDashboard from "../../components/ReviewsDashboard";

export const metadata = {
  title: "Review Analysis | ListingLens",
  description: "NLP sentiment analysis of guest reviews for NYC Airbnb listings.",
};

const NAV = [
  { href: "/",        label: "About",           active: false },
  { href: "/predict", label: "Price Estimator", active: false },
  { href: "/reviews", label: "Review Analysis",  active: true  },
];

export default function ReviewsPage() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>

      <header style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <a href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(180deg, var(--primary) 0%, var(--primary-dark) 100%)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em", boxShadow: "0 1px 4px rgba(37,99,235,0.25)" }}>LL</div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>ListingLens</span>
          </a>
          <nav style={{ display: "flex", alignItems: "center", gap: 2 }}>
            {NAV.map(n => (
              <a key={n.href} href={n.href} className={`nav-link ${n.active ? "active" : ""}`}>{n.label}</a>
            ))}
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
          <h1 className="page-title" style={{ marginBottom: 10 }}>Review Analysis</h1>
          <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.6, maxWidth: 500 }}>
            Select any NYC Airbnb listing to see a breakdown of what guests say: overall sentiment,
            common topics, quality score, and recent review excerpts.
          </p>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)", padding: "32px 32px 64px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <ReviewsDashboard />
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
