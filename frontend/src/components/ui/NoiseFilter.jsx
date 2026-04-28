/** Invisible SVG filter definition used to apply subtle film-grain noise. */
export function NoiseFilter() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }}>
      <filter id="noise">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
        <feColorMatrix type="saturate" values="0" />
        <feBlend in="SourceGraphic" mode="overlay" result="blend" />
        <feComposite in="blend" in2="SourceGraphic" operator="in" />
      </filter>
    </svg>
  );
}