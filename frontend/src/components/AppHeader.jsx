import { Badge } from "./ui/Badge";
import { ThemeToggle } from "./ui/ThemeToggle";

/**
 * Sticky top header: logo, loaded-spec count badge, live indicator, theme toggle.
 *
 * @prop {boolean}   dark         - Current theme state.
 * @prop {function}  onToggleDark - Flips the theme.
 * @prop {string[]}  loadedSpecs  - Names of currently ingested specifications.
 * @prop {object}    t            - Theme token object from theme().
 */
export function AppHeader({ dark, onToggleDark, loadedSpecs, t }) {
  return (
    <header style={{
      padding: "20px 40px",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      borderBottom: `1px solid ${dark ? "#ffffff08" : "#dbeafe"}`,
      backdropFilter: "blur(12px)",
      position: "sticky", top: 0, zIndex: 10,
      background: t.bgHeader,
    }}>
      {/* Logo + wordmark */}
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        <div>
          <img 
            src="search-icon.png" 
            alt="Logo" 
            style={{ width: "20px", height: "20px", objectFit: "contain" }} 
          />
        </div>
        <div style={{
          fontFamily: "'Syne', sans-serif", fontWeight: 800,
          fontSize: 20, letterSpacing: "-0.02em", color: t.textPrimary,
        }}>
          DocQuery
        </div>
      </div>

      {/* Right-side controls */}
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        {loadedSpecs.length > 0 && (
          <Badge dark={dark}>
            {loadedSpecs.length} specification{loadedSpecs.length > 1 ? "s" : ""} loaded
          </Badge>
        )}
        <Badge color="#22c55e" dark={dark}>● Live</Badge>
        <ThemeToggle dark={dark} onToggle={onToggleDark} />
      </div>
    </header>
  );
}