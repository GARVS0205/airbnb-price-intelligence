import { NextResponse } from "next/server";

export const maxDuration = 10;

/**
 * GET /api/ping
 * Simple health check — returns immediately.
 * The ML backend is now embedded (ONNX + SQLite), so no warm-up needed.
 */
export async function GET() {
  return NextResponse.json({ status: "warm", message: "ML running on Vercel (no external backend)" });
}
