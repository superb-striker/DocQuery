/**
 * Maps raw backend error detail strings to user-friendly messages + hints.
 * @param {string|undefined} detail  - The `detail` field from the API error response.
 * @param {"Ingest"|"Query"} context - Which operation failed, for generic fallbacks.
 * @returns {{ msg: string, hint: string|null }}
 */
export function classifyError(detail, context) {
  if (!detail)
    return { msg: `${context} failed. Check the server is running.`, hint: null };
  if (detail.includes("Invalid or empty specification"))
    return {
      msg: "Could not parse that URL as a valid OpenAPI specification.",
      hint: "Make sure the URL returns a JSON OpenAPI/Swagger document.",
    };
  if (detail.includes("Internal server error") && context === "Ingest")
    return {
      msg: "Server error during ingestion.",
      hint: "Check your FastAPI logs - the URL may be unreachable or the specification malformed.",
    };
  if (detail.includes("No specification found"))
    return {
      msg: "This specification hasn't been ingested yet.",
      hint: "Go to step 1 and ingest the specification URL first.",
    };
  if (detail.includes("LLM failed"))
    return {
      msg: "The LLM returned an empty response.",
      hint: "The HuggingFace model may be rate-limited or offline. Try again in a moment.",
    };
  if (detail.includes("Internal server error") && context === "Query")
    return {
      msg: "Server error during query.",
      hint: "Check your FastAPI logs for details.",
    };
  if (detail.includes("Failed to fetch") || detail.includes("NetworkError"))
    return {
      msg: "Cannot reach the backend.",
      hint: "Make sure uvicorn is running on http://127.0.0.1:8000",
    };
  return { msg: detail, hint: null };
}