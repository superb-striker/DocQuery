/**
 * Small spinning loading indicator.
 * @prop {boolean} dark - Whether dark mode is active (affects track colour).
 */
export function Spinner({ dark }) {
  return (
    <div style={{
      width: 18, height: 18,
      border: `2px solid ${dark ? "#ffffff1a" : "#00000015"}`,
      borderTop: "2px solid #3b82f6",
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
      flexShrink: 0,
    }} />
  );
}