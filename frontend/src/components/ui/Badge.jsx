/**
 * Small pill-shaped label, used in the header and elsewhere.
 *
 * @prop {string}  [color]    - Hex accent colour. Defaults to blue tuned for the current theme.
 * @prop {boolean} dark       - Whether dark mode is active (affects default colour).
 * @prop {*}       children
 */
export function Badge({ children, color, dark }) {
  const c = color || (dark ? "#60a5fa" : "#3b82f6");
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "2px 10px", borderRadius: 999,
      border: `1px solid ${c}44`,
      background: `${c}18`,
      color: c, fontSize: 11, fontFamily: "'DM Mono', monospace",
      letterSpacing: "0.05em", fontWeight: 600,
    }}>
      {children}
    </span>
  );
}