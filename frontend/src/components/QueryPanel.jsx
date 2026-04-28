import { Spinner } from "./ui/Spinner";
import { primaryBtnStyle } from "../theme";

/**
 * Panel containing the question textarea and submit button.
 * Also renders the specification selector pills above the textarea.
 *
 * @prop {boolean}   dark
 * @prop {object}    t            - Theme tokens.
 * @prop {string[]}  loadedSpecs  - All ingested spec names.
 * @prop {string|null} activeSpec - The currently selected spec name.
 * @prop {function}  onSelectSpec - (name: string) => void
 * @prop {string}    question     - Current textarea value.
 * @prop {function}  onQuestion   - (value: string) => void
 * @prop {boolean}   querying     - Whether a query is in-flight.
 * @prop {function}  onQuery      - () => void
 */
export function QueryPanel({
  dark, t,
  loadedSpecs, activeSpec, onSelectSpec,
  question, onQuestion,
  querying, onQuery,
}) {
  return (
    <>
      {/* Specification selector */}
      {loadedSpecs.length > 0 && (
        <div style={{
          marginBottom: 20, display: "flex", gap: 8, flexWrap: "wrap",
          alignItems: "center", animation: "fadeUp 0.4s ease both",
        }}>
          <span style={{
            fontSize: 11, color: t.textMuted,
            fontFamily: "'DM Mono', monospace", marginRight: 4,
          }}>
            active:
          </span>
          {loadedSpecs.map(name => (
            <button
              key={name}
              onClick={() => onSelectSpec(name)}
              style={{
                padding: "5px 14px", borderRadius: 999,
                border: `1px solid ${activeSpec === name ? t.pillActiveBorder : t.pillInactiveBorder}`,
                background: activeSpec === name ? t.pillActive : t.pillInactive,
                color: activeSpec === name ? t.pillActiveText : t.pillInactiveText,
                fontSize: 12, fontFamily: "'DM Mono', monospace",
                transition: "all 0.2s",
              }}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      {/* Query input */}
      <div style={{
        background: t.bgCard, border: `1px solid ${t.borderCard}`,
        borderRadius: 16, padding: 24, marginBottom: 28,
        animation: "fadeUp 0.6s 0.2s ease both", opacity: 0,
        animationFillMode: "forwards",
        boxShadow: dark ? "none" : "0 2px 16px #3b82f610",
      }}>
        <div style={{
          fontSize: 11, fontFamily: "'DM Mono', monospace",
          color: t.textLabel, letterSpacing: "0.12em", marginBottom: 14, fontWeight: 600,
        }}>
          ASK ANYTHING
        </div>
        <textarea
          value={question}
          onChange={e => onQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onQuery(); }}
          placeholder={activeSpec ? `Ask about ${activeSpec}…` : "Load a specification first, then ask a question…"}
          disabled={!activeSpec}
          rows={3}
          style={{
            width: "100%", background: t.inputBg,
            border: `1px solid ${t.borderInput}`, borderRadius: 10,
            padding: "14px 16px", color: t.textPrimary,
            fontSize: 14, fontFamily: "'DM Sans', sans-serif",
            resize: "none", lineHeight: 1.65,
            transition: "border-color 0.2s, box-shadow 0.2s",
            opacity: !activeSpec ? 0.45 : 1,
          }}
          onFocus={e => {
            e.target.style.borderColor = t.borderInputFocus;
            e.target.style.boxShadow = `0 0 0 3px ${t.accentGlow}`;
          }}
          onBlur={e => {
            e.target.style.borderColor = t.borderInput;
            e.target.style.boxShadow = "none";
          }}
        />
        <div style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "center", marginTop: 12,
        }}>
          <span style={{ fontSize: 11, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
            {activeSpec ? "⌘↵ to submit" : ""}
          </span>
          <button
            onClick={onQuery}
            disabled={querying || !question.trim() || !activeSpec}
            style={primaryBtnStyle(t, querying || !question.trim() || !activeSpec)}
            onMouseEnter={e => {
              if (!querying && question.trim() && activeSpec)
                e.currentTarget.style.transform = "scale(1.02)";
            }}
            onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
          >
            {querying ? <><Spinner dark={dark} /> Thinking…</> : "Ask →"}
          </button>
        </div>
      </div>
    </>
  );
}