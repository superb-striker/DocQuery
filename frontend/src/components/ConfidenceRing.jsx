/**
 * Circular SVG progress ring showing the LLM confidence score (0–100).
 * Colour shifts from blue → amber → red as confidence decreases.
 *
 * @prop {number}  value - Confidence value 0–100.
 * @prop {boolean} dark  - Whether dark mode is active.
 */
export function ConfidenceRing({ value, dark }) {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const color = value >= 75 ? "#3b82f6" : value >= 50 ? "#f59e0b" : "#ef4444";
  const trackColor = dark ? "#ffffff0d" : "#00000010";

  return (
    <div style={{ position: "relative", width: 72, height: 72, flexShrink: 0 }}>
      <svg width="72" height="72" style={{ transform: "rotate(-90deg)" }}>
        <circle cx="36" cy="36" r={r} fill="none" stroke={trackColor} strokeWidth="4" />
        <circle
          cx="36" cy="36" r={r} fill="none"
          stroke={color} strokeWidth="4"
          strokeDasharray={circ}
          strokeDashoffset={circ - (value / 100) * circ}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1)" }}
        />
      </svg>
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{
          color, fontSize: 15, fontWeight: 700,
          fontFamily: "'DM Mono', monospace", lineHeight: 1,
        }}>
          {value}
        </span>
        <span style={{
          color: dark ? "#ffffff55" : "#00000044",
          fontSize: 9, fontFamily: "'DM Mono', monospace", letterSpacing: "0.08em",
        }}>
          CONF
        </span>
      </div>
    </div>
  );
}