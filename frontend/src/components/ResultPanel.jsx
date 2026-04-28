import { useRef } from "react";
import { ConfidenceRing } from "./ConfidenceRing";
import { SourceChip } from "./SourceChip";
import { useTypewriter } from "../hooks/useTypewriter";

/**
 * Displays the query result: animated answer text, confidence ring, and source chips.
 * Renders nothing if `result` is null.
 *
 * @prop {object|null} result  - API response: { answer, confidence, sources[] }
 * @prop {boolean}     dark
 * @prop {object}      t       - Theme tokens.
 * @prop {boolean}     animate - Trigger for the typewriter effect; toggled by parent on new results.
 */
export function ResultPanel({ result, dark, t, animate }) {
  const answerRef = useRef(null);
  const displayedAnswer = useTypewriter(result?.answer, 12, animate);

  if (!result) return null;

  return (
    <div style={{ animation: "fadeUp 0.5s ease both" }}>

      {/* Answer card */}
      <div style={{
        background: t.bgCard, border: `1px solid ${t.borderCard}`,
        borderRadius: 16, padding: 28, marginBottom: 16,
        boxShadow: dark ? "none" : "0 2px 16px #3b82f610",
      }}>
        <div style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "flex-start", marginBottom: 20,
        }}>
          <div style={{
            fontSize: 11, fontFamily: "'DM Mono', monospace",
            color: t.textLabel, letterSpacing: "0.12em", fontWeight: 600,
          }}>
            ANSWER
          </div>
          <ConfidenceRing value={result.confidence} dark={dark} />
        </div>

        <p ref={answerRef} style={{
          fontSize: 15, lineHeight: 1.8, color: t.textPrimary,
          whiteSpace: "pre-wrap", minHeight: 40,
        }}>
          {displayedAnswer}
          {/* Blinking cursor while typing */}
          <span style={{
            display: "inline-block", width: 2, height: "1em",
            background: t.accent, marginLeft: 2, verticalAlign: "text-bottom",
            animation: displayedAnswer.length < (result?.answer?.length || 0)
              ? "spin 0.7s linear infinite" : "none",
            opacity: displayedAnswer.length < (result?.answer?.length || 0) ? 1 : 0,
            transition: "opacity 0.3s",
          }} />
        </p>
      </div>

      {/* Sources */}
      {result.sources?.length > 0 && (
        <div style={{
          background: t.bgCardAlt, border: `1px solid ${t.borderCard}`,
          borderRadius: 16, padding: 20,
          boxShadow: dark ? "none" : "0 2px 16px #3b82f610",
        }}>
          <div style={{
            fontSize: 11, fontFamily: "'DM Mono', monospace",
            color: t.textLabel, letterSpacing: "0.12em", marginBottom: 14, fontWeight: 600,
          }}>
            SOURCES · {result.sources.length} endpoint{result.sources.length > 1 ? "s" : ""}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {result.sources.map((s, i) => {
              const [method, ...rest] = s.endpoint.split(" ");
              return (
                <SourceChip
                  key={i}
                  method={method}
                  path={rest.join(" ")}
                  summary={s.summary}
                  dark={dark}
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}