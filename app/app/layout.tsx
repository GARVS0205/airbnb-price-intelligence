import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ListingLens — NYC Airbnb Price Intelligence",
  description: "ML-powered nightly price estimator for NYC Airbnb listings. XGBoost model trained on 20k listings with NLP review sentiment analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
