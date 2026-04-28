const METHOD_COLORS = {
  GET: "#60a5fa",
  POST: "#34d399",
  PUT: "#fbbf24",
  DELETE: "#f87171",
  PATCH: "#a78bfa",
};

/**
 * Displays a single API endpoint source reference (method badge + path + summary).
 *
 * @prop {string}      method  - HTTP method string e.g. "GET".
 * @prop {string}      path    - Endpoint path e.g. "/pets/{id}".
 * @prop {string|null} summary - Optional short description from the spec.
 * @prop {boolean}     dark    - Whether dark mode is active.
 */
export function SourceChip({ method, path, summary, dark }) {
  const color = METHOD_COLORS[method] || (dark ? "#94a3b8" : "#64748b");
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 8,
      padding: "8px 12px", borderRadius: 8,
      background: dark ? "#ffffff07" : "#00000006",
      border: `1px solid ${dark ? "#ffffff0f" : "#0000000f"}`,
    }}>
      <span style={{
        fontFamily: "'DM Mono', monospace", fontSize: 10, fontWeight: 700,
        color, background: `${color}20`, padding: "2px 7px", borderRadius: 4,
        flexShrink: 0, letterSpacing: "0.05em",
      }}>
        {method}
      </span>
      <div>
        <div style={{
          fontFamily: "'DM Mono', monospace", fontSize: 12,
          color: dark ? "#e2e8f0" : "#1e293b",
        }}>
          {path}
        </div>
        {summary && (
          <div style={{ fontSize: 11, color: dark ? "#94a3b8" : "#64748b", marginTop: 2 }}>
            {summary}
          </div>
        )}
      </div>
    </div>
  );
}