/**
 * Displays a formatted error message with an optional hint.
 * Renders nothing when `error` is null/undefined.
 *
 * @prop {{ msg: string, hint: string|null }|null} error
 * @prop {boolean} dark
 */
export function ErrorBanner({ error, dark }) {
  if (!error) return null;
  return (
    <div style={{
      padding: "14px 18px", borderRadius: 12,
      background: "#ef44440f", border: "1px solid #ef444422",
      marginBottom: 24, animation: "fadeUp 0.3s ease",
    }}>
      <div style={{ color: "#ef4444", fontSize: 13, fontFamily: "'DM Mono', monospace" }}>
        ✕ {error.msg}
      </div>
      {error.hint && (
        <div style={{
          color: dark ? "#94a3b8" : "#64748b",
          fontSize: 11, fontFamily: "'DM Mono', monospace", marginTop: 6,
        }}>
          → {error.hint}
        </div>
      )}
    </div>
  );
}