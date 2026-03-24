"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { PredictionResult, FormData } from "@/types";

interface Props { prediction: PredictionResult; formData: FormData | null; }

const FEATURE_LABELS: Record<string, string> = {
  neighbourhood_target_encoded: "Location / Neighbourhood",
  accommodates:                 "Guest Capacity",
  bedrooms:                     "Bedrooms",
  bathrooms:                    "Bathrooms",
  room_type_encoded:            "Room Type",
  amenity_count:                "Amenity Count",
  premium_amenity_score:        "Premium Amenities",
  review_scores_rating:         "Review Score",
  log_number_of_reviews:        "Review Volume",
  is_superhost:                 "Superhost Status",
  dist_times_square_km:         "Dist. Times Square",
  dist_central_park_km:         "Dist. Central Park",
  host_experience_days:         "Host Experience",
  composite_review_score:       "Composite Review Score",
  has_ac:                       "Air Conditioning",
  // NLP features
  review_avg_sentiment:         "Review Sentiment (NLP)",
  review_positive_pct:          "Positive Review %",
  review_quality_score:         "Review Quality Score",
  review_sentiment_trend:       "Sentiment Trend",
};

const BOROUGH_AVG: Record<number, { name: string; avg: number }> = {
  0: { name: "Brooklyn",      avg: 165 },
  1: { name: "Manhattan",     avg: 225 },
  2: { name: "Bronx",         avg: 95  },
  3: { name: "Staten Island", avg: 110 },
  4: { name: "Queens",        avg: 140 },
};

const TT = {
  background: "var(--surface)",
  border: "1px solid var(--ghost-border)",
  borderRadius: 8,
  color: "var(--text-primary)",
  fontSize: 12,
  boxShadow: "var(--shadow-md)",
};

/** Sentiment → label + color */
function sentimentMeta(s: number): { label: string; color: string } {
  if (s >= 0.35)  return { label: "Very Positive", color: "var(--success)" };
  if (s >= 0.10)  return { label: "Positive",       color: "#059669" };
  if (s >= -0.05) return { label: "Mixed / Neutral", color: "var(--warning)" };
  return              { label: "Negative",       color: "var(--error)" };
}

export default function PredictionDisplay({ prediction, formData }: Props) {
  const { predicted_price, price_low, price_high, top_features, review_nlp_used, review_nlp_features } = prediction;

  const borough = formData?.borough_encoded ?? 1;
  const info    = BOROUGH_AVG[borough] ?? { name: "NYC", avg: 175 };
  const diff    = predicted_price - info.avg;
  const diffPct    = ((diff / info.avg) * 100).toFixed(1);
  const diffPctNum  = parseFloat(diffPct);
  const isAbove = diff > 0;

  const compData = [
    { name: "Your listing",      value: Math.round(predicted_price), color: "var(--primary)" },
    { name: info.name + " avg",  value: info.avg,                    color: "var(--surface-high)" },
  ];

  const featData = top_features.map(f => ({
    name:  FEATURE_LABELS[f.feature] ?? f.feature.replace(/_/g, " "),
    score: f.importance,
    max:   top_features[0]?.importance || 1,
  }));

  /* Confidence bar geometry */
  const rangeMin = price_low  * 0.88;
  const rangeMax = price_high * 1.12;
  const rangeW   = rangeMax - rangeMin;
  const barLeft  = ((price_low  - rangeMin) / rangeW) * 100;
  const barWidth = ((price_high - price_low) / rangeW) * 100;
  const dotPos   = ((predicted_price - rangeMin) / rangeW) * 100;

  const nlp = review_nlp_features;
  const sm  = nlp ? sentimentMeta(nlp.review_avg_sentiment) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* Price card */}
      <div className="card-result">
        <div style={{ padding: "24px 26px 20px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
            <span className="section-label">Predicted Nightly Rate</span>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {review_nlp_used && (
                <span style={{ fontSize: 11, background: "var(--primary-dim)", color: "var(--primary)", borderRadius: 99, padding: "2px 8px", fontWeight: 500 }}>
                  NLP
                </span>
              )}
              <span className={`badge ${isAbove ? "badge-success" : "badge-warning"}`}>
                {isAbove ? "+" : ""}{diffPct}% vs {info.name}
              </span>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 22 }}>
            <span className="price-mono">${Math.round(predicted_price)}</span>
            <span style={{ fontSize: 15, color: "var(--text-muted)" }}>per night</span>
          </div>

          {/* Confidence band */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
              <span>Low estimate: <strong style={{ color: "var(--text-secondary)" }}>${Math.round(price_low)}</strong></span>
              <span>High estimate: <strong style={{ color: "var(--text-secondary)" }}>${Math.round(price_high)}</strong></span>
            </div>
            <div style={{ position: "relative", height: 6, background: "var(--surface-high)", borderRadius: 99, overflow: "visible" }}>
              <div style={{
                position: "absolute", top: 0, height: "100%", borderRadius: 99,
                background: "rgba(37,99,235,0.18)",
                left: `${barLeft}%`, width: `${barWidth}%`,
              }} />
              <div style={{
                position: "absolute", top: -4, width: 14, height: 14, borderRadius: "50%",
                background: "var(--primary)", border: "2px solid var(--surface)",
                boxShadow: "0 0 0 3px rgba(37,99,235,0.15)",
                left: `calc(${dotPos}% - 7px)`,
              }} />
            </div>
            <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10 }}>
              Approx. 18% confidence interval  ·  XGBoost  ·  R2 = 0.809
              {review_nlp_used && " · includes review NLP features"}
            </p>
          </div>
        </div>
      </div>

      {/* Review Signal Panel */}
      {review_nlp_used && nlp && sm && (
        <div className="card">
          <div className="card-header">
            <span className="card-header-title">Review Signal</span>
            <span style={{ fontSize: 11, color: "var(--primary)" }}>Used in prediction</span>
          </div>
          <div className="card-body">
            {/* 3-stat row */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
              {[
                { label: "Avg Sentiment",   value: nlp.review_avg_sentiment.toFixed(3), color: sm.color },
                { label: "Positive %",      value: `${nlp.review_positive_pct.toFixed(0)}%`, color: "var(--success)" },
                { label: "Quality Score",   value: nlp.review_quality_score.toFixed(0), color: "var(--primary)" },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ textAlign: "center", padding: "12px 6px", borderRadius: 8, background: "var(--surface-low)" }}>
                  <div style={{ fontSize: 20, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 5 }}>{label}</div>
                </div>
              ))}
            </div>

            {/* Sentiment bar */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>Sentiment</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: sm.color }}>{sm.label}</span>
            </div>
            <div style={{ position: "relative", height: 5, background: "var(--surface-high)", borderRadius: 99, marginBottom: 14 }}>
              <div style={{
                position: "absolute",
                left: nlp.review_avg_sentiment >= 0 ? "50%" : `${(0.5 + nlp.review_avg_sentiment / 2) * 100}%`,
                top: 0, height: "100%", borderRadius: 99,
                width: `${Math.abs(nlp.review_avg_sentiment) * 50}%`,
                background: nlp.review_avg_sentiment >= 0 ? "var(--success)" : "var(--error)",
              }} />
              <div style={{ position: "absolute", left: "50%", top: -2, width: 1, height: 9, background: "var(--ghost-border)" }} />
            </div>

            {/* Trend */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Avg review length: <span style={{ color: "var(--text-secondary)" }}>{nlp.review_avg_word_count.toFixed(0)} words</span>
              </span>
              {Math.abs(nlp.review_sentiment_trend) > 0.02 && (
                <span style={{ fontSize: 12, color: nlp.review_sentiment_trend > 0 ? "var(--success)" : "var(--error)", fontWeight: 500 }}>
                  {nlp.review_sentiment_trend > 0 ? "Improving" : "Declining"} trend
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* No listing_id hint */}
      {!review_nlp_used && (
        <div style={{ padding: "12px 16px", borderRadius: 10, background: "var(--surface-low)", fontSize: 12, color: "var(--text-muted)" }}>
          Enter a <strong style={{ color: "var(--primary)" }}>Listing ID</strong> above to include real review sentiment in the price prediction.
        </div>
      )}

      {/* Borough comparison */}
      <div className="card">
        <div className="card-header">
          <span className="card-header-title">Borough Comparison</span>
          <span style={{ fontSize: 11, color: isAbove ? "var(--success)" : "var(--warning)", fontWeight: 500 }}>
            {isAbove ? "Above" : "Below"} average by ${Math.round(Math.abs(diff))}
          </span>
        </div>
        <div className="card-body">
          <ResponsiveContainer width="100%" height={110}>
            <BarChart data={compData} barSize={40} margin={{ top: 4, right: 0, left: -20, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
              <Tooltip contentStyle={TT} formatter={(v: unknown) => [`$${v}/night`, ""]} cursor={{ fill: "rgba(37,99,235,0.04)" }} />
              <Bar dataKey="value" radius={[5,5,0,0]}>
                {compData.map((d,i) => <Cell key={i} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top price drivers */}
      {featData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-header-title">Top Price Drivers</span>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>XGBoost gain importance</span>
          </div>
          <div className="card-body">
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {featData.slice(0, 5).map((f, i) => (
                <div key={i}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: "var(--text-secondary)", textTransform: "capitalize" }}>{f.name}</span>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--primary)" }}>{(f.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-primary" style={{ width: `${(f.score / f.max) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Optimisation Insights */}
      <div className="card">
        <div className="card-header">
          <span className="card-header-title">Optimisation Insights</span>
        </div>
        <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[
            !isAbove && { type: "tip", text: `Your listing is ${Math.abs(diffPctNum)}% below the ${info.name} average. You may be underpricing — consider testing a slightly higher rate.` },
            isAbove && diffPctNum > 25 && { type: "tip", text: `Your listing is ${diffPct}% above the ${info.name} average. Make sure listing quality clearly justifies the premium to avoid high cancellation rates.` },
            isAbove && diffPctNum <= 25 && { type: "ok",  text: `Priced ${diffPct}% above the ${info.name} average — a healthy premium that suggests strong listing quality.` },
            !formData?.is_superhost && { type: "tip", text: "Superhost status builds guest trust and typically supports a 8–12% higher nightly rate. Focus on response time and consistent 5-star stays." },
            (formData?.amenity_count ?? 0) < 20 && { type: "tip", text: "Listings with fewer than 20 amenities often underperform on price. Adding A/C, a workspace, or a washer can meaningfully increase perceived value." },
            (formData?.amenity_count ?? 0) >= 20 && (formData?.amenity_count ?? 0) < 35 && { type: "tip", text: "A/C and a dedicated workspace are the two amenities with the highest price impact in NYC. Worth adding if you haven't yet." },
            (formData?.review_scores_rating ?? 5) < 4.5 && { type: "tip", text: "A review score below 4.5 significantly dampens predicted price. Focus on cleanliness and communication — the two most impactful factors for guest satisfaction." },
            (formData?.review_scores_rating ?? 5) >= 4.5 && (formData?.review_scores_rating ?? 5) < 4.8 && { type: "tip", text: "Scores above 4.8 unlock a price premium. You're close — a few more excellent stays could push you over the threshold." },
            (formData?.number_of_reviews ?? 0) < 10 && { type: "tip", text: "Fewer than 10 reviews means guests can't easily trust your listing. Pricing slightly below market initially helps build a review base faster." },
            (formData?.number_of_reviews ?? 0) >= 10 && (formData?.number_of_reviews ?? 0) < 30 && { type: "tip", text: "Listings with 30+ reviews command higher prices due to reduced guest uncertainty. Keep encouraging reviews after each stay." },
            review_nlp_used && nlp && nlp.review_positive_pct < 70 && { type: "tip", text: `Only ${nlp.review_positive_pct.toFixed(0)}% of your reviews use positive language. Addressing recurring guest concerns could meaningfully improve this and your predicted price.` },
            review_nlp_used && nlp && nlp.review_positive_pct >= 85 && { type: "ok", text: `${nlp.review_positive_pct.toFixed(0)}% positive reviews is excellent. Your review sentiment is actively boosting your predicted price.` },
          ].filter(Boolean).map((item, i) => {
            const it = item as { type: string; text: string };
            const isOk = it.type === "ok";
            return (
              <div key={i} style={{ display: "flex", gap: 10, padding: "10px 13px", borderRadius: 8,
                background: isOk ? "var(--success-dim)" : "var(--primary-dim)",
              }}>
                <span style={{ color: isOk ? "var(--success)" : "var(--primary)", fontWeight: 700, fontSize: 11, flexShrink: 0, marginTop: 2, letterSpacing: "0.03em" }}>
                  {isOk ? "Good" : "Tip"}
                </span>
                <span style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.55 }}>{it.text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
