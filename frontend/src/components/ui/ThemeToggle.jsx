/**
 * Sliding pill toggle that switches between dark and light mode.
 * @prop {boolean}  dark       - Current theme state.
 * @prop {function} onToggle   - Called when the button is clicked.
 */
export function ThemeToggle({ dark, onToggle }) {
  return (
    <button
      onClick={onToggle}
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      style={{
        width: 38, height: 22, borderRadius: 999, position: "relative",
        background: dark ? "#3b82f644" : "#e2e8f0",
        border: `1px solid ${dark ? "#3b82f666" : "#cbd5e1"}`,
        cursor: "pointer",
        transition: "background 0.3s, border-color 0.3s",
        flexShrink: 0,
      }}
    >
      <div style={{
        position: "absolute", top: 2, left: dark ? 18 : 2,
        width: 16, height: 16, borderRadius: "50%",
        background: dark ? "#60a5fa" : "#94a3b8",
        transition: "left 0.25s cubic-bezier(.4,0,.2,1), background 0.3s",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 9,
      }}>
        {dark ? "🌙" : "☀️"}
      </div>
    </button>
  );
}