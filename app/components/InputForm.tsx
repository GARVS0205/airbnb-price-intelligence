"use client";
import { useState } from "react";
import { FormData } from "@/types";

const NEIGHBOURHOODS = [
  { label: "Midtown Manhattan",  v: "midtown",       enc: 5.6, ts: 0.5,  cp: 1.2,  jfk: 22, boro: 1, lat: 40.7549, lon: -73.984  },
  { label: "Upper West Side",    v: "uws",           enc: 5.5, ts: 2.5,  cp: 0.4,  jfk: 21, boro: 1, lat: 40.787,  lon: -73.9754 },
  { label: "Chelsea",            v: "chelsea",       enc: 5.5, ts: 1.5,  cp: 2.1,  jfk: 21, boro: 1, lat: 40.7465, lon: -74.0014 },
  { label: "West Village",       v: "wv",            enc: 5.6, ts: 3.0,  cp: 2.7,  jfk: 22, boro: 1, lat: 40.7356, lon: -74.006  },
  { label: "Lower East Side",    v: "les",           enc: 5.3, ts: 3.5,  cp: 4.2,  jfk: 20, boro: 1, lat: 40.7157, lon: -73.9863 },
  { label: "Williamsburg",       v: "williamsburg",  enc: 5.2, ts: 5.0,  cp: 5.5,  jfk: 14, boro: 0, lat: 40.7081, lon: -73.9571 },
  { label: "Brooklyn Heights",   v: "bklyn",         enc: 5.2, ts: 6.0,  cp: 6.5,  jfk: 16, boro: 0, lat: 40.696,  lon: -73.9937 },
  { label: "Park Slope",         v: "parkslope",     enc: 5.1, ts: 7.5,  cp: 7.8,  jfk: 13, boro: 0, lat: 40.671,  lon: -73.9769 },
  { label: "Astoria, Queens",    v: "astoria",       enc: 4.8, ts: 8.0,  cp: 7.0,  jfk: 17, boro: 4, lat: 40.7721, lon: -73.9302 },
  { label: "Harlem",             v: "harlem",        enc: 4.9, ts: 5.5,  cp: 2.5,  jfk: 22, boro: 1, lat: 40.8116, lon: -73.9465 },
  { label: "Bronx",              v: "bronx",         enc: 4.5, ts: 13.0, cp: 8.0,  jfk: 28, boro: 2, lat: 40.8448, lon: -73.8648 },
  { label: "Staten Island",      v: "si",            enc: 4.4, ts: 20.0, cp: 18.0, jfk: 26, boro: 3, lat: 40.5795, lon: -74.1502 },
];

const ROOM_TYPES = [
  { label: "Entire home / apartment", value: 0 },
  { label: "Private room",            value: 2 },
  { label: "Hotel room",              value: 1 },
  { label: "Shared room",             value: 3 },
];

interface Props { onSubmit: (d: FormData) => void; loading: boolean; }

const Label = ({ text }: { text: string }) => (
  <div style={{ fontSize: 12, fontWeight: 500, color: "var(--text-variant)", marginBottom: 5 }}>{text}</div>
);

const Check = ({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) => (
  <label style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", userSelect: "none" }}>
    <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
    <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{label}</span>
  </label>
);

export default function InputForm({ onSubmit, loading }: Props) {
  const [listingId, setListingId] = useState("");
  const [nbhd, setNbhd]   = useState(NEIGHBOURHOODS[0]);
  const [rt,   setRt]     = useState(ROOM_TYPES[0]);
  const [f, setF] = useState({
    accommodates: 2, bedrooms: 1, bathrooms: 1, beds: 1,
    amenity_count: 25, review_scores_rating: 4.7, number_of_reviews: 50,
    host_response_rate: 0.95,
    has_pool: false, has_gym: false, has_parking: false, has_elevator: false,
    has_washer: true,  has_ac: true,  has_workspace: false,
    is_superhost: false, instant_bookable: false, has_luxury_keywords: false,
  });
  const s = (k: string, v: unknown) => setF(p => ({ ...p, [k]: v }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const premium = [f.has_pool,f.has_gym,f.has_parking,f.has_elevator,f.has_washer,f.has_ac,f.has_workspace].filter(Boolean).length;
    onSubmit({
      neighbourhood_target_encoded: nbhd.enc, borough_encoded: nbhd.boro,
      latitude: nbhd.lat, longitude: nbhd.lon,
      dist_times_square_km: nbhd.ts, dist_central_park_km: nbhd.cp,
      dist_jfk_airport_km: nbhd.jfk,
      dist_brooklyn_bridge_km: Math.abs(nbhd.lat - 40.7061) * 111,
      dist_grand_central_km: Math.abs(nbhd.lat - 40.7527) * 111,
      dist_statue_liberty_km: Math.abs(nbhd.lon - (-74.0445)) * 85,
      geo_cluster: nbhd.boro,
      room_type_encoded: rt.value,
      accommodates: f.accommodates, bedrooms: f.bedrooms,
      bathrooms: f.bathrooms, beds: f.beds,
      minimum_nights: 2, availability_365: 180,
      amenity_count: f.amenity_count, premium_amenity_score: premium,
      has_pool: f.has_pool?1:0, has_gym: f.has_gym?1:0,
      has_parking: f.has_parking?1:0, has_elevator: f.has_elevator?1:0,
      has_washer: f.has_washer?1:0, has_ac: f.has_ac?1:0,
      has_workspace: f.has_workspace?1:0,
      is_superhost: f.is_superhost?1:0,
      host_response_rate: f.host_response_rate, host_acceptance_rate: 0.85,
      calculated_host_listings_count: 1,
      number_of_reviews: f.number_of_reviews,
      review_scores_rating: f.review_scores_rating,
      reviews_per_month: 2,
      instant_bookable: f.instant_bookable?1:0,
      has_luxury_keywords: f.has_luxury_keywords?1:0,
      has_renovated_keywords: 0,
      // Phase 2: pass listing_id if provided
      ...(listingId.trim() ? { listing_id: parseInt(listingId.trim(), 10) } : {}),
    });
  };

  const Section = ({ title }: { title: string }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "4px 0 12px" }}>
      <span className="section-label">{title}</span>
      <div style={{ flex: 1, height: 1, background: "var(--ghost-border)" }} />
    </div>
  );

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 0 }}>

      {/* Listing ID — Phase 2 integration */}
      <div style={{ padding: "14px 20px", background: "var(--surface-low)", borderBottom: "1px solid var(--ghost-border)" }}>
        <label style={{ fontSize: 12, fontWeight: 500, color: "var(--text-variant)", display: "block", marginBottom: 5 }}>
          Listing ID
          <span style={{ marginLeft: 6, fontSize: 11, color: "var(--primary)", fontWeight: 400, opacity: 0.8 }}>optional · enables review sentiment</span>
        </label>
        <input
          type="number" min="1" placeholder="e.g. 2539"
          className="input"
          value={listingId}
          onChange={e => setListingId(e.target.value)}
        />
        {listingId && (
          <p style={{ fontSize: 11, color: "var(--primary)", marginTop: 6, fontWeight: 500 }}>
            NLP active, prediction will use real review sentiment
          </p>
        )}
      </div>

      {/* Location */}
      <div style={{ padding: "20px 20px 16px" }}>
        <Section title="Location" />
        <Label text="Neighborhood" />
        <select className="input" value={nbhd.v}
          onChange={e => { const n = NEIGHBOURHOODS.find(x => x.v === e.target.value); if (n) setNbhd(n); }}>
          {NEIGHBOURHOODS.map(n => <option key={n.v} value={n.v}>{n.label}</option>)}
        </select>
      </div>

      <div className="divider" />

      {/* Property */}
      <div style={{ padding: "16px 20px" }}>
        <Section title="Property" />
        <div style={{ marginBottom: 10 }}>
          <Label text="Room type" />
          <select className="input" value={rt.value}
            onChange={e => { const r = ROOM_TYPES.find(x => x.value === parseInt(e.target.value)); if (r) setRt(r); }}>
            {ROOM_TYPES.map(r => <option key={r.value} value={r.value}>{r.label}</option>)}
          </select>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {([["accommodates","Guests"],["bedrooms","Bedrooms"],["bathrooms","Bathrooms"],["beds","Beds"]] as [keyof typeof f, string][]).map(([k,l]) => (
            <div key={k}>
              <Label text={l} />
              <input type="number" min="0" max="20" step={k==="bathrooms"?"0.5":"1"}
                className="input" value={f[k] as number}
                onChange={e => s(k, parseFloat(e.target.value))} />
            </div>
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* Amenities */}
      <div style={{ padding: "16px 20px" }}>
        <Section title="Amenities" />
        <Label text={`Total count: ${f.amenity_count}`} />
        <input type="range" min="0" max="80" style={{ marginBottom: 12 }}
          value={f.amenity_count} onChange={e => s("amenity_count", parseInt(e.target.value))} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "7px 16px" }}>
          {([
            ["has_pool","Pool"],["has_gym","Gym"],["has_parking","Parking"],
            ["has_elevator","Elevator"],["has_washer","Washer/Dryer"],["has_ac","Air Conditioning"],
            ["has_workspace","Workspace"],["has_luxury_keywords","Luxury listing"],
          ] as [keyof typeof f, string][]).map(([k,l]) => (
            <Check key={k} label={l} checked={!!f[k]} onChange={v => s(k, v)} />
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* Host & Reviews */}
      <div style={{ padding: "16px 20px 20px" }}>
        <Section title="Host & Reviews" />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
          <div>
            <Label text="Review score (0–5)" />
            <input type="number" min="0" max="5" step="0.1" className="input"
              value={f.review_scores_rating} onChange={e => s("review_scores_rating", parseFloat(e.target.value))} />
          </div>
          <div>
            <Label text="No. of reviews" />
            <input type="number" min="0" max="2000" className="input"
              value={f.number_of_reviews} onChange={e => s("number_of_reviews", parseInt(e.target.value))} />
          </div>
        </div>
        <Label text={`Host response rate: ${(f.host_response_rate * 100).toFixed(0)}%`} />
        <input type="range" min="0" max="1" step="0.01" style={{ marginBottom: 12 }}
          value={f.host_response_rate} onChange={e => s("host_response_rate", parseFloat(e.target.value))} />
        <div style={{ display: "flex", gap: 16 }}>
          <Check label="Superhost" checked={f.is_superhost} onChange={v => s("is_superhost", v)} />
          <Check label="Instant Book" checked={f.instant_bookable} onChange={v => s("instant_bookable", v)} />
        </div>
      </div>

      <div className="divider" />

      <div style={{ padding: 16 }}>
        <button type="submit" disabled={loading} className="btn btn-primary" style={{ width: "100%", height: 42, fontSize: 13.5 }}>
          {loading ? <><span className="spinner" />Running prediction…</> : "Run Price Prediction"}
        </button>
      </div>
    </form>
  );
}
