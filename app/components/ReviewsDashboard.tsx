"use client";
import { useState, useRef, useEffect } from "react";
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis } from "recharts";
import LISTINGS from "@/lib/listingsDirectory.json";

/* ── Types ─────────────────────────────────────────────────────────────────── */
interface ListingMeta { id: string; name: string; neighbourhood: string; room_type: string; price: number; reviews: number; }
interface Sentiment   { positive_pct: number; neutral_pct: number; negative_pct: number; avg_compound: number; sentiment_label: string; positive_count: number; neutral_count: number; negative_count: number; }
interface Theme       { theme: string; count: number; pct: number; }
interface Flag        { type: string; severity: "ok" | "warning"; message: string; }
interface Timeline    { month: string; reviews: number; }
interface Sample      { date: string; text: string; }
interface Quality     { score: number; grade: string; label: string; components: { sentiment: number; volume: number; detail: number; diversity: number; }; }
interface Analysis    { listing_id: number; total_reviews: number; avg_review_length_words: number; sentiment: Sentiment; quality_score: Quality; themes: Theme[]; red_flags: Flag[]; timeline: Timeline[]; sample_reviews: Sample[]; }

const listings = LISTINGS as ListingMeta[];

const PIE_COLORS   = ["var(--success)", "#94a3b8", "var(--error)"];
const THEME_COLORS = ["var(--primary)","#7c3aed","#0891b2","var(--warning)","#be185d","var(--success)","#0369a1"];
const TT           = { background: "var(--surface)", border: "1px solid var(--ghost-border)", borderRadius: 8, color: "var(--text-primary)", fontSize: 12, boxShadow: "var(--shadow-md)" };
const gradeColor   = (g: string) => g==="A"?"var(--success)":g==="B"?"var(--primary)":g==="C"?"var(--warning)":"var(--error)";

const ROOM_SHORT: Record<string, string> = {
  "Entire home/apt": "Entire", "Private room": "Private",
  "Hotel room": "Hotel",       "Shared room": "Shared",
};

/* ── Searchable Listing Picker ──────────────────────────────────────────────── */
function ListingPicker({ onSelect }: { onSelect: (l: ListingMeta) => void }) {
  const [query,  setQuery]  = useState("");
  const [open,   setOpen]   = useState(false);
  const [picked, setPicked] = useState<ListingMeta | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  // close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filtered = query.trim()
    ? listings.filter(l =>
        l.name.toLowerCase().includes(query.toLowerCase()) ||
        l.neighbourhood.toLowerCase().includes(query.toLowerCase()) ||
        l.id.includes(query)
      ).slice(0, 30)
    : listings.slice(0, 30);

  const selectListing = (l: ListingMeta) => {
    setPicked(l);
    setQuery("");
    setOpen(false);
    onSelect(l);
  };

  return (
    <div ref={ref} style={{ position: "relative" }}>
      {/* ── Trigger ── */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "9px 12px", borderRadius: 8, cursor: "pointer",
          background: "var(--surface)", border: `1px solid ${open ? "var(--primary)" : "rgba(67,70,85,0.15)"}`,
          boxShadow: open ? "0 0 0 3px rgba(37,99,235,0.10)" : "none",
          transition: "all 0.15s",
        }}
      >
        {picked ? (
          <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--primary)", flexShrink: 0 }}>#{picked.id.slice(-6)}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{picked.name}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{picked.neighbourhood} · {ROOM_SHORT[picked.room_type] ?? picked.room_type} · ${picked.price}/night</div>
            </div>
          </div>
        ) : (
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Select a listing…</span>
        )}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" style={{ flexShrink: 0, transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s", marginLeft: 8 }}>
          <path d="M6 9l6 6 6-6"/>
        </svg>
      </div>

      {/* ── Dropdown ── */}
      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 6px)", left: 0, right: 0, zIndex: 100,
          background: "var(--surface)", border: "1px solid var(--ghost-border)",
          borderRadius: 10, boxShadow: "var(--shadow-float)", overflow: "hidden",
        }}>
          {/* Search input */}
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--ghost-border)" }}>
            <input
              autoFocus
              className="input"
              style={{ fontSize: 13 }}
              placeholder="Search by name, neighbourhood, or ID…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onClick={e => e.stopPropagation()}
            />
          </div>

          {/* List */}
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            {filtered.length === 0 && (
              <div style={{ padding: "16px 14px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>No listings match your search</div>
            )}
            {filtered.map(l => (
              <div
                key={l.id}
                onClick={() => selectListing(l)}
                style={{
                  display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
                  cursor: "pointer", borderBottom: "1px solid var(--ghost-border)",
                  transition: "background 0.1s",
                }}
                onMouseEnter={e => (e.currentTarget.style.background = "var(--surface-low)")}
                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              >
                {/* ID chip */}
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--primary)", background: "var(--primary-dim)", borderRadius: 5, padding: "2px 6px", flexShrink: 0, whiteSpace: "nowrap" }}>
                  #{l.id.slice(-6)}
                </div>

                {/* Main info */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 500, color: "var(--text-primary)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {l.name}
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 3 }}>
                    <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{l.neighbourhood}</span>
                    <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>·</span>
                    <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{ROOM_SHORT[l.room_type] ?? l.room_type}</span>
                  </div>
                </div>

                {/* Right: price + review count */}
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div className="mono" style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>${l.price}<span style={{ fontSize: 9.5, color: "var(--text-muted)", fontFamily: "Inter" }}>/night</span></div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>{l.reviews.toLocaleString()} reviews</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ padding: "8px 14px", borderTop: "1px solid var(--ghost-border)", fontSize: 11, color: "var(--text-muted)" }}>
            Showing {filtered.length} of {listings.length} listings
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Main Dashboard ──────────────────────────────────────────────────────────── */
export default function ReviewsDashboard() {
  const [selected, setSelected] = useState<ListingMeta | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState<Analysis | null>(null);
  const [error,    setError]    = useState<string | null>(null);

  // Silently wake up the Render backend as soon as the Review page loads
  useEffect(() => {
    fetch("/api/ping", { method: "GET" }).catch(() => {});
  }, []);

  const isNotFound = error?.toLowerCase().includes("no reviews found");

  const analyze = async (listing: ListingMeta) => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res  = await fetch("/api/analyze-reviews", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ listing_id: parseInt(listing.id), max_reviews: 300 }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error ?? "Analysis failed");
      setResult(data as Analysis);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally { setLoading(false); }
  };

  const handleSelect = (l: ListingMeta) => {
    setSelected(l);
    setResult(null);
    setError(null);
  };

  const sentPie = result ? [
    { name: "Positive", value: result.sentiment.positive_pct },
    { name: "Neutral",  value: result.sentiment.neutral_pct  },
    { name: "Negative", value: result.sentiment.negative_pct },
  ] : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* ── Listing selector card ── */}
      <div className="card" style={{ overflow: "visible" }}>
        <div className="card-header" style={{ overflow: "visible" }}>
          <span className="card-header-title">Select a Listing</span>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{listings.length.toLocaleString()} NYC listings</span>
        </div>
        <div className="card-body">
          <p style={{ fontSize: 12.5, color: "var(--text-muted)", marginBottom: 12 }}>
            Choose a listing to see overall guest sentiment, common themes, and review quality score.
          </p>
          <div style={{ display: "flex", gap: 10, overflow: "visible" }}>
            <div style={{ flex: 1 }}>
              <ListingPicker onSelect={handleSelect} />
            </div>
            <button
              className="btn btn-emerald"
              disabled={!selected || loading}
              style={{ flexShrink: 0, height: 42, minWidth: 130 }}
              onClick={() => selected && analyze(selected)}
            >
              {loading ? <>Analysing…</> : "Analyse"}
            </button>
          </div>

          {error && (
            <div style={{
              marginTop: 10, padding: "10px 14px", borderRadius: 8, fontSize: 13, lineHeight: 1.5,
              background: "var(--surface-low)",
              color: "var(--text-muted)",
            }}>
              {isNotFound ? "Reviews not available for this listing." : error}
            </div>
          )}
        </div>
      </div>

      {/* ── Empty state ── */}
      {!result && !loading && !error && (
        <div className="card" style={{ padding: 48, textAlign: "center" }}>
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>Select a listing above and click <strong style={{ color: "var(--text-secondary)" }}>Analyse Reviews</strong></p>
        </div>
      )}

      {result && selected && (<>

        {/* Listing summary banner */}
        <div className="card">
          <div style={{ padding: "16px 22px", display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", marginBottom: 5 }}>{selected.name}</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                <span className="badge badge-neutral">{selected.neighbourhood}</span>
                <span className="badge badge-neutral">{selected.room_type}</span>
                <span className="badge badge-neutral">${selected.price}/night</span>
                <span className="badge badge-primary">ID {selected.id}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── 4 stat cards ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
          {[
            { v: result.total_reviews.toString(),          l: "Total Reviews",  s: "Loaded for analysis" },
            { v: `${result.quality_score.score}/100`,      l: "Quality Score",  s: result.quality_score.label },
            { v: result.sentiment.sentiment_label,         l: "Sentiment",      s: `Avg compound: ${result.sentiment.avg_compound.toFixed(2)}` },
            { v: `${result.avg_review_length_words}w`,     l: "Avg Length",     s: "Words per review" },
          ].map(s => (
            <div key={s.l} className="stat-card">
              <div className="mono stat-value">{s.v}</div>
              <div className="stat-label">{s.l}</div>
              <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 2 }}>{s.s}</div>
            </div>
          ))}
        </div>

        {/* Quality Score Card */}
        <div className="card">
          <div className="card-header">
            <span className="card-header-title">Review Quality Score</span>
          </div>
          <div className="card-body">
            {/* Grade hero */}
            <div style={{ display: "flex", alignItems: "center", gap: 18, padding: "16px 20px", borderRadius: 10, background: "var(--surface-low)", marginBottom: 18 }}>
              <div style={{ width: 60, height: 60, borderRadius: 12, background: gradeColor(result.quality_score.grade), display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <span style={{ fontSize: 28, fontWeight: 800, color: "#fff", lineHeight: 1 }}>{result.quality_score.grade}</span>
              </div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
                  {result.quality_score.label} — {result.quality_score.score}/100
                </div>
                <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.55 }}>
                  {result.quality_score.grade === "A" && "Guests love this listing. Reviews are overwhelmingly positive, detailed, and consistent."}
                  {result.quality_score.grade === "B" && "Guests are generally satisfied. Most reviews are positive with some room for improvement."}
                  {result.quality_score.grade === "C" && "Mixed reception. Reviews show both positives and negatives — worth reviewing guest feedback."}
                  {result.quality_score.grade === "D" && "Below average. Guests are frequently unhappy. Review feedback carefully for recurring issues."}
                </div>
              </div>
            </div>

            {/* Score bar */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                <span>Poor</span><span>Good</span><span>Excellent</span>
              </div>
              <div className="progress-track" style={{ height: 8 }}>
                <div className="progress-primary" style={{ width: `${result.quality_score.score}%`, height: "100%", borderRadius: 99, transition: "width 0.6s ease" }} />
              </div>
            </div>

            {/* Breakdown */}
            <div style={{ marginBottom: 16, fontSize: 13, color: "var(--text-muted)", background: "var(--surface-low)", padding: "10px 14px", borderRadius: 8 }}>
              This <strong>100-point Quality Score</strong> evaluates how guests write about your property. It is composed of 4 key metrics:
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {[
                { l: "Guest Sentiment",   hint: "How positive the language in reviews is",              v: result.quality_score.components.sentiment,  max: 40 },
                { l: "Number of Reviews", hint: "How many guests have reviewed this listing",           v: result.quality_score.components.volume,     max: 20 },
                { l: "Review Detail",     hint: "How descriptive and thorough reviews are on average",  v: result.quality_score.components.detail,     max: 20 },
                { l: "Topic Variety",     hint: "How many different aspects guests comment on",         v: result.quality_score.components.diversity,  max: 20 },
              ].map(c => (
                <div key={c.l}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <div>
                      <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary)" }}>{c.l}</span>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 1 }}>{c.hint}</div>
                    </div>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--primary)", flexShrink: 0, marginTop: 2 }}>{c.v.toFixed(0)} / {c.max} pts</span>
                  </div>
                  <div className="progress-track"><div className="progress-primary" style={{ width: `${(c.v / c.max) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Charts ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <div className="card">
            <div className="card-header"><span className="card-header-title">Sentiment Distribution</span></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={sentPie} cx="50%" cy="50%" outerRadius={65} dataKey="value" label={({ name, value }) => `${name} ${value}%`} labelLine={false} fontSize={10.5}>
                    {sentPie.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                  </Pie>
                  <Tooltip contentStyle={TT} formatter={(v: unknown) => [`${v}%`]} />
                  <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {result.timeline.length > 0 ? (
            <div className="card">
              <div className="card-header"><span className="card-header-title">Review Activity</span></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={result.timeline} barSize={12} margin={{ top: 0, right: 0, left: -16, bottom: 0 }}>
                    <XAxis dataKey="month" tick={{ fill: "var(--text-muted)", fontSize: 9.5 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={TT} />
                    <Bar dataKey="reviews" fill="var(--primary)" radius={[3,3,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <div className="card" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              <p style={{ color: "var(--text-muted)", fontSize: 12 }}>Insufficient date data for timeline</p>
            </div>
          )}
        </div>

        {/* ── Themes grid ── */}
        <div className="card">
          <div className="card-header"><span className="card-header-title">Topics Mentioned</span></div>
          <div className="card-body">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
              {result.themes.map((t, i) => (
                <div key={t.theme} className="stat-card" style={{ background: "var(--surface-low)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: THEME_COLORS[i] }} />
                    <span style={{ fontSize: 10.5, fontWeight: 600, color: "var(--text-secondary)", textTransform: "capitalize" }}>{t.theme}</span>
                  </div>
                  <div className="mono stat-value" style={{ fontSize: 18 }}>{t.pct}%</div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>{t.count} reviews</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Health check ── */}
        <div className="card">
          <div className="card-header"><span className="card-header-title">Review Health Check</span></div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {result.red_flags.map((f, i) => (
              <div key={i} style={{
                display: "flex", gap: 12, padding: "10px 14px", borderRadius: 8,
                background: f.severity === "ok" ? "var(--success-dim)" : "var(--warning-dim)",
              }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: f.severity === "ok" ? "var(--success)" : "var(--warning)", flexShrink: 0, marginTop: 1 }}>
                  {f.severity === "ok" ? "OK" : "!"}
                </span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: f.severity === "ok" ? "var(--success)" : "var(--warning)" }}>{f.type}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{f.message}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Plain-language insights */}
        {(() => {
          const s = result.sentiment;
          const q = result.quality_score;
          const insights: { type: "good" | "tip" | "note"; text: string }[] = [];

          if (s.positive_pct >= 85)
            insights.push({ type: "good", text: `${s.positive_pct.toFixed(0)}% of guests used positive language in their reviews. This is an excellent signal of overall guest satisfaction.` });
          else if (s.positive_pct >= 65)
            insights.push({ type: "note", text: `${s.positive_pct.toFixed(0)}% positive reviews is decent, but there's room to improve. Check the Topics breakdown to see what guests comment on most.` });
          else
            insights.push({ type: "tip", text: `Only ${s.positive_pct.toFixed(0)}% of reviews are positive. Guests are frequently expressing neutral or negative sentiment — worth reading recent reviews carefully.` });

          if (s.negative_pct > 20)
            insights.push({ type: "tip", text: `${s.negative_pct.toFixed(0)}% of reviews contain negative language. Look at the Topics section to identify which aspects are driving dissatisfaction.` });

          if (result.avg_review_length_words < 20)
            insights.push({ type: "tip", text: "Reviews are very short on average. Brief reviews often indicate guests weren't deeply engaged or didn't have strong feelings either way." });
          else if (result.avg_review_length_words >= 50)
            insights.push({ type: "good", text: "Guests write detailed reviews for this listing. Long reviews typically indicate memorable stays and a highly engaged guest audience." });

          if (result.total_reviews < 10)
            insights.push({ type: "note", text: "Fewer than 10 reviews were analysed. Results may not be representative — more guest feedback is needed for a reliable picture." });
          else if (result.total_reviews >= 100)
            insights.push({ type: "good", text: `${result.total_reviews} reviews analysed. A large review pool makes these results highly reliable.` });

          if (result.themes.length > 0) {
            const top = result.themes[0];
            insights.push({ type: "note", text: `The most frequently mentioned topic is "${top.theme}" — appearing in ${top.pct}% of reviews. This is what guests think about most when they stay here.` });
          }

          if (q.components.diversity < 8)
            insights.push({ type: "tip", text: "Guests are commenting on a narrow range of topics. This may indicate the listing experience is one-dimensional — consider what other aspects hosts in this category typically excel at." });

          return insights.length > 0 ? (
            <div className="card">
              <div className="card-header"><span className="card-header-title">What This Tells You</span></div>
              <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {insights.map((ins, i) => (
                  <div key={i} style={{ display: "flex", gap: 10, padding: "10px 14px", borderRadius: 8,
                    background: ins.type === "good" ? "var(--success-dim)" : ins.type === "tip" ? "var(--primary-dim)" : "var(--surface-low)",
                  }}>
                    <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.02em", flexShrink: 0, marginTop: 2,
                      color: ins.type === "good" ? "var(--success)" : ins.type === "tip" ? "var(--primary)" : "var(--text-muted)",
                    }}>
                      {ins.type === "good" ? "Good" : ins.type === "tip" ? "Tip" : "Note"}
                    </span>
                    <span style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>{ins.text}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null;
        })()}

        {/* Sample reviews */}
        {result.sample_reviews.length > 0 && (
          <div className="card">
            <div className="card-header"><span className="card-header-title">Recent Reviews</span></div>
            <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {result.sample_reviews.map((r, i) => (
                <div key={i} style={{ padding: "14px 16px", borderRadius: 8, background: "var(--surface-low)" }}>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>{r.date}</div>
                  <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65 }}>{r.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </>)}
    </div>
  );
}
