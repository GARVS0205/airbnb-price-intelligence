"use client";
import { useState } from "react";
import NavBar from "@/components/NavBar";

const NAV = [
  { href: "/",        label: "About",           active: true  },
  { href: "/predict", label: "Price Estimator", active: false },
  { href: "/reviews", label: "Review Analysis", active: false },
];

const FOR_WHOM = [
  {
    audience: "Airbnb Hosts",
    icon: "🏠",
    desc: "Find out if your listing is priced too high, too low, or just right. Discover what specific upgrades, like adding a workspace or improving your reviews, will actually move the needle on your price.",
  },
  {
    audience: "Prospective Hosts",
    icon: "📋",
    desc: "Planning to list a property? Get an accurate price estimate before you go live. Compare your expected rate against similar listings in your neighbourhood to set realistic expectations.",
  },
  {
    audience: "Short-term Rental Investors",
    icon: "📈",
    desc: "Evaluate NYC neighbourhoods by average nightly rates, demand signals, and guest review quality before committing to a property. Make data-backed decisions, not guesses.",
  },
];

const WHAT_IT_DOES = [
  {
    title: "Predict the right nightly price",
    desc: "Enter your listing details, location, room type, amenities, host status, and get an accurate nightly price estimate. See whether your listing should be priced above or below the neighbourhood average, and by how much.",
  },
  {
    title: "Factor in guest reviews automatically",
    desc: "Have an existing listing? Add your Listing ID and ListingLens will read your real guest reviews, detecting language patterns that guests use when they love a place, and incorporate that quality signal into the price estimate.",
  },
  {
    title: "Understand what's driving your price",
    desc: "The results don't just give you a number. They show you which factors contributed most to your price, so you know exactly what to improve to charge more.",
  },
  {
    title: "Analyse any listing's reviews",
    desc: "Use the Review Analysis tool to inspect the sentiment and topics in any NYC Airbnb listing's guest reviews. See quality scores, common themes, and health flags, useful for competitive research or self-assessment.",
  },
];

const USER_FLOW = [
  { n: "01", title: "Open Price Estimator",         desc: "Navigate to Price Estimator from the top navigation." },
  { n: "02", title: "Choose your neighbourhood",    desc: "Select one of 12 NYC neighbourhoods across Manhattan, Brooklyn, Queens, Bronx, and Staten Island." },
  { n: "03", title: "Set your property details",    desc: "Pick your room type, number of guests, beds, bedrooms, and bathrooms." },
  { n: "04", title: "Select your amenities",        desc: "Check which amenities your listing offers: pool, gym, A/C, workspace, washer, and more." },
  { n: "05", title: "Optionally add a Listing ID",  desc: "If you have a live Airbnb listing, enter its ID to include your real guest review quality in the prediction." },
  { n: "06", title: "Run the estimate",             desc: "Click 'Get Price Estimate'. Results are ready in seconds." },
  { n: "07", title: "Explore your results",         desc: "See your nightly price range, how it compares to your borough's average, and personalised tips to increase your rate." },
  { n: "08", title: "Try Review Analysis",          desc: "Switch to Review Analysis to deep-dive into guest reviews for any NYC listing. Get quality scores, topic breakdown, and improvement flags." },
];

export default function AboutPage() {
  const [flowOpen, setFlowOpen] = useState(true);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <NavBar nav={NAV} />

      {/* Hero */}
      <div style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)" }}>
        <div className="page-pad hero-section" style={{ maxWidth: 800, margin: "0 auto", paddingTop: 60, paddingBottom: 52, textAlign: "center" }}>
          <h1 className="page-title" style={{ marginBottom: 16 }}>
            Know exactly what your Airbnb listing should cost tonight.
          </h1>
          <p style={{ fontSize: 16, color: "var(--text-muted)", lineHeight: 1.75, maxWidth: 560, margin: "0 auto 36px" }}>
            ListingLens analyses thousands of real NYC Airbnb listings, including guest reviews, to give you a smart, data-driven nightly price estimate for any property in New York City.
          </p>
          <div className="hero-btns">
            <a href="/predict" className="btn btn-primary" style={{ padding: "11px 26px", fontSize: 14 }}>Try the Price Estimator</a>
            <a href="/reviews" className="btn btn-secondary" style={{ padding: "11px 26px", fontSize: 14 }}>Explore Review Analysis</a>
          </div>
        </div>
      </div>

      <main style={{ flex: 1, background: "var(--bg)" }}>
        <div className="page-pad" style={{ maxWidth: 1100, margin: "0 auto", paddingTop: 52, paddingBottom: 80, display: "flex", flexDirection: "column", gap: 56 }}>

          {/* What it does */}
          <section>
            <h2 className="section-title" style={{ marginBottom: 6 }}>What ListingLens Does</h2>
            <p style={{ fontSize: 13.5, color: "var(--text-muted)", marginBottom: 24, lineHeight: 1.65 }}>
              Two tools. One goal: help you price smarter.
            </p>
            <div className="grid-2">
              {WHAT_IT_DOES.map(w => (
                <div key={w.title} className="card">
                  <div style={{ padding: "22px 26px" }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 8 }}>{w.title}</div>
                    <div style={{ fontSize: 13.5, color: "var(--text-muted)", lineHeight: 1.7 }}>{w.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Who it's for */}
          <section>
            <h2 className="section-title" style={{ marginBottom: 6 }}>Who Is It For?</h2>
            <p style={{ fontSize: 13.5, color: "var(--text-muted)", marginBottom: 24, lineHeight: 1.65 }}>
              ListingLens is built for anyone making decisions about NYC short-term rental pricing.
            </p>
            <div className="grid-2">
              {FOR_WHOM.map(f => (
                <div key={f.audience} className="card">
                  <div style={{ padding: "22px 26px" }}>
                    <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 10 }}>
                      <span style={{ fontSize: 22 }}>{f.icon}</span>
                      <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{f.audience}</span>
                    </div>
                    <div style={{ fontSize: 13.5, color: "var(--text-muted)", lineHeight: 1.7 }}>{f.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* How to use */}
          <section>
            <div
              style={{ display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", marginBottom: flowOpen ? 20 : 0 }}
              onClick={() => setFlowOpen(o => !o)}
            >
              <div>
                <h2 className="section-title" style={{ marginBottom: 4 }}>How to Use It</h2>
                {!flowOpen && <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Click to expand the step-by-step guide</p>}
              </div>
              <span style={{ fontSize: 20, color: "var(--primary)", fontWeight: 300, transform: flowOpen ? "rotate(45deg)" : "none", transition: "transform 0.18s", display: "inline-block" }}>+</span>
            </div>

            {flowOpen && (
              <div className="grid-2">
                {USER_FLOW.map(step => (
                  <div key={step.n} className="card">
                    <div style={{ padding: "18px 22px", display: "flex", gap: 14 }}>
                      <span style={{ fontSize: 11, fontWeight: 700, color: "var(--primary)", background: "var(--primary-dim)", padding: "3px 8px", borderRadius: 5, flexShrink: 0, alignSelf: "flex-start", marginTop: 2 }}>{step.n}</span>
                      <div>
                        <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)", marginBottom: 5 }}>{step.title}</div>
                        <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.65 }}>{step.desc}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
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
