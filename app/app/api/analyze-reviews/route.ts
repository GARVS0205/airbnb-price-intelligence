import { NextRequest, NextResponse } from "next/server";
import path from "path";

export const maxDuration = 60;

/**
 * POST /api/analyze-reviews
 *
 * Reads precomputed review analysis directly from the SQLite database
 * (app/models/reviews_summary.db) using better-sqlite3.
 * No Python, no Flask, no external backend — everything runs on Vercel.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const listing_id = parseInt(String(body.listing_id ?? 0), 10);

    if (!listing_id || listing_id <= 0) {
      return NextResponse.json(
        { error: "A valid listing_id is required." },
        { status: 400 }
      );
    }

    // Dynamic import — better-sqlite3 is a native Node.js module
    const Database = (await import("better-sqlite3")).default;
    const dbPath   = path.join(process.cwd(), "models", "reviews_summary.db");

    let db: InstanceType<typeof Database> | null = null;
    try {
      db = new Database(dbPath, { readonly: true, fileMustExist: true });
    } catch (e) {
      return NextResponse.json(
        { error: "Reviews database not found. Ensure reviews_summary.db is deployed." },
        { status: 500 }
      );
    }

    try {
      const row = db
        .prepare("SELECT analysis_data FROM reviews_summary WHERE listing_id = ?")
        .get(listing_id) as { analysis_data: string } | undefined;

      if (!row || !row.analysis_data || row.analysis_data === "{}") {
        return NextResponse.json(
          { error: `No reviews found for listing ${listing_id}. Try IDs like 2539, 2595, 3176.` },
          { status: 404 }
        );
      }

      const data = JSON.parse(row.analysis_data);
      return NextResponse.json(data, { status: 200 });

    } finally {
      db.close();
    }

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/analyze-reviews] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
