"use client";
import { useState, useRef } from "react";

interface NavItem {
  href: string;
  label: string;
  active: boolean;
}

interface NavBarProps {
  nav: NavItem[];
  showModelLive?: boolean;
}

const GITHUB_URL = "https://github.com/GARVS0205/airbnb-price-intelligence";

/** Pages that use the ML backend — hovering these triggers a pre-warm ping */
const ML_PAGES = ["/predict", "/reviews"];

export default function NavBar({ nav, showModelLive = false }: NavBarProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const pingFiredRef = useRef(false); // fire at most once per NavBar mount

  const handleNavHover = (href: string) => {
    if (!ML_PAGES.includes(href)) return;
    if (pingFiredRef.current) return; // already pinged
    pingFiredRef.current = true;
    // Fire-and-forget — we don't need the result here
    fetch("/api/ping", { method: "GET" }).catch(() => {});
  };

  return (
    <>
      <header style={{ background: "var(--surface)", borderBottom: "1px solid var(--ghost-border)", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          {/* Top row */}
          <div className="page-pad" style={{ height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            {/* Logo */}
            <a href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
              <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(180deg, var(--primary) 0%, var(--primary-dark) 100%)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em", boxShadow: "0 1px 4px rgba(37,99,235,0.25)", flexShrink: 0 }}>LL</div>
              <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>ListingLens</span>
            </a>

            {/* Desktop nav */}
            <nav className="nav-menu">
              {nav.map(n => (
                <a
                  key={n.href}
                  href={n.href}
                  className={`nav-link ${n.active ? "active" : ""}`}
                  onMouseEnter={() => handleNavHover(n.href)}
                >
                  {n.label}
                </a>
              ))}
              <div style={{ width: 1, height: 16, background: "var(--ghost-border)", margin: "0 10px" }} />
              <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="nav-link">GitHub</a>
            </nav>

            {/* Right side: live badge + hamburger */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="nav-live-badge" style={{ display: "flex", alignItems: "center", gap: 7, background: "var(--success-dim)", borderRadius: 99, padding: "5px 12px" }}>
                <span className="live-dot" />
                <span style={{ fontSize: 12, fontWeight: 500, color: "var(--success)" }}>{showModelLive ? "Model live" : "Live"}</span>
              </div>
              {/* Hamburger — only visible on mobile via CSS */}
              <button
                className="nav-hamburger"
                onClick={() => setMenuOpen(o => !o)}
                aria-label="Toggle navigation"
              >
                {menuOpen ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
                )}
              </button>
            </div>
          </div>

          {/* Mobile dropdown drawer */}
          <div className={`nav-drawer ${menuOpen ? "open" : ""}`}>
            {nav.map(n => (
              <a
                key={n.href}
                href={n.href}
                className={`nav-link ${n.active ? "active" : ""}`}
                onClick={() => setMenuOpen(false)}
                onTouchStart={() => handleNavHover(n.href)}
              >
                {n.label}
              </a>
            ))}
            <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="nav-link" onClick={() => setMenuOpen(false)}>
              GitHub
            </a>
          </div>
        </div>
      </header>
    </>
  );
}

