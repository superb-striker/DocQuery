export function theme(dark) {
  return {
    bg: dark ? "#080c14" : "#f0f4ff",
    bgHeader: dark ? "#080c14cc" : "#f0f4ffcc",
    bgCard: dark ? "#ffffff06" : "#ffffff90",
    bgCardAlt: dark ? "#ffffff04" : "#ffffff70",
    borderCard: dark ? "#ffffff0f" : "#dbeafe",
    borderInput: dark ? "#ffffff10" : "#cbd5e1",
    borderInputFocus: dark ? "#3b82f655" : "#3b82f6aa",
    textPrimary: dark ? "#e2e8f0" : "#0f172a",
    textSecondary: dark ? "#94a3b8" : "#475569",
    textMuted: dark ? "#475569" : "#94a3b8",
    textLabel: dark ? "#64748b" : "#94a3b8",
    accent: "#3b82f6",
    accentAlt: "#6366f1",
    accentGlow: dark ? "#3b82f633" : "#3b82f622",
    inputBg: dark ? "#ffffff08" : "#ffffff",
    pillActive: dark ? "#3b82f611" : "#eff6ff",
    pillActiveBorder: dark ? "#3b82f655" : "#93c5fd",
    pillActiveText: dark ? "#60a5fa" : "#1d4ed8",
    pillInactive: "transparent",
    pillInactiveBorder: dark ? "#ffffff15" : "#e2e8f0",
    pillInactiveText: dark ? "#64748b" : "#94a3b8",
    emptyIcon: dark ? "#1e293b" : "#cbd5e1",
    emptyText: dark ? "#334155" : "#94a3b8",
    gradientRadial1: dark
      ? "radial-gradient(ellipse 80% 50% at 50% -10%, #1d40af18 0%, transparent 60%)"
      : "radial-gradient(ellipse 80% 50% at 50% -10%, #bfdbfe40 0%, transparent 60%)",
    gradientRadial2: dark
      ? "radial-gradient(ellipse 40% 30% at 80% 80%, #312e8115 0%, transparent 50%)"
      : "radial-gradient(ellipse 40% 30% at 80% 80%, #e0e7ff50 0%, transparent 50%)",
    btnGradient: "linear-gradient(135deg, #3b82f6, #6366f1)",
    btnDisabledBg: dark ? "#ffffff08" : "#f1f5f9",
    btnDisabledText: dark ? "#ffffff33" : "#94a3b8",
    scrollThumb: dark ? "#ffffff18" : "#cbd5e1",
  };
}

/** Returns consistent inline styles for primary action buttons. */
export function primaryBtnStyle(t, disabled) {
  return {
    padding: "12px 22px", borderRadius: 10,
    background: disabled ? t.btnDisabledBg : t.btnGradient,
    color: disabled ? t.btnDisabledText : "#ffffff",
    fontSize: 13, fontWeight: 700, fontFamily: "'Syne', sans-serif",
    display: "flex", alignItems: "center", gap: 8,
    transition: "opacity 0.2s, transform 0.1s, box-shadow 0.2s",
    opacity: disabled ? 0.5 : 1,
    boxShadow: disabled ? "none" : `0 4px 20px ${t.accentGlow}`,
    cursor: disabled ? "not-allowed" : "pointer",
    border: "none",
  };
}

/** Returns consistent inline styles for text inputs. */
export function inputBaseStyle(t) {
  return {
    flex: 1, background: t.inputBg,
    border: `1px solid ${t.borderInput}`,
    borderRadius: 10, padding: "12px 16px",
    color: t.textPrimary, fontSize: 13,
    fontFamily: "'DM Mono', monospace",
    transition: "border-color 0.2s, box-shadow 0.2s",
  };
}