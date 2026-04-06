import { NextResponse } from "next/server";

export const maxDuration = 60; // Vercel Hobby plan max is 60s

/**
 * GET /api/ping
 * Wakes up the Render backend by calling its /health endpoint.
 * Called on page load AND on nav-link hover to warm the backend
 * before the user even submits a request.
 */
export async function GET() {
  const pythonApiUrl = process.env.PYTHON_API_URL;
  if (!pythonApiUrl) {
    return NextResponse.json({ status: "local" });
  }
  const t0 = Date.now();
  try {
    const res = await fetch(`${pythonApiUrl}/health`, {
      signal: AbortSignal.timeout(55_000),
      cache: "no-store",
    });
    const body = await res.json();
    return NextResponse.json({
      status: "warm",
      latency_ms: Date.now() - t0,
      backend: body,
    });
  } catch {
    return NextResponse.json(
      { status: "cold", latency_ms: Date.now() - t0 },
      { status: 503 }
    );
  }
}
