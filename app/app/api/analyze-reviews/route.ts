import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

/**
 * POST /api/analyze-reviews
 *
 * Calls review_analysis_api.py via child_process.
 * Body: { listing_id: number, max_reviews?: number }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const listing_id = parseInt(String(body.listing_id ?? 0));
    const max_reviews = parseInt(String(body.max_reviews ?? 300));

    if (!listing_id || listing_id <= 0) {
      return NextResponse.json({ error: "A valid listing_id is required." }, { status: 400 });
    }

    const scriptPath = path.join(process.cwd(), "review_analysis_api.py");
    const result = await runPythonScript(scriptPath, { listing_id, max_reviews });
    return NextResponse.json(result, { status: 200 });

  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[/api/analyze-reviews] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function runPythonScript(
  scriptPath: string,
  payload: Record<string, unknown>
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const python = spawn("python", ["-X", "utf8", scriptPath]);
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    python.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

    python.on("close", (code: number) => {
      if (code !== 0) {
        reject(new Error(`Python script failed (code ${code}): ${stderr}`));
        return;
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result);
        }
      } catch {
        reject(new Error(`Failed to parse Python output: ${stdout.substring(0, 300)}`));
      }
    });

    python.on("error", (err: Error) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`));
    });

    python.stdin.write(JSON.stringify(payload));
    python.stdin.end();
  });
}
