import { NextResponse } from "next/server";

export const maxDuration = 80;

/**
 * GET /api/ping
 * Silently wakes up the Render backend by calling its /health endpoint.
 * Called on page load from the Price Estimator and Review Analysis pages
 * so Render is already warm by the time the user submits a request.
 */
export async function GET() {
  const pythonApiUrl = process.env.PYTHON_API_URL;
  if (!pythonApiUrl) {
    return NextResponse.json({ status: "local" });
  }
  try {
    const res = await fetch(`${pythonApiUrl}/health`, {
      signal: AbortSignal.timeout(55_000),
    });
    const body = await res.json();
    return NextResponse.json({ status: "warm", backend: body });
  } catch {
    return NextResponse.json({ status: "cold" }, { status: 503 });
  }
}
