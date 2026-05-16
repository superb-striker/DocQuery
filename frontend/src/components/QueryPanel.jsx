import { useRef, useState } from "react";
import { Spinner } from "./ui/Spinner";
import { primaryBtnStyle } from "../theme";

/**
 * Panel containing the question textarea, mic button, and submit button.
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
  // Voice state: "idle" | "recording" | "transcribing"
  const [voiceState, setVoiceState] = useState("idle");
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const isRecording = voiceState === "recording";
  const isTranscribing = voiceState === "transcribing";
  const voiceBusy = isRecording || isTranscribing;

  const handleMic = async () => {
    // ── Stop recording ──────────────────────────────────────────────────────
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      return;
    }

    // ── Start recording ─────────────────────────────────────────────────────
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = e => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Stop all mic tracks so the browser indicator clears
        stream.getTracks().forEach(t => t.stop());

        setVoiceState("transcribing");
        try {
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const formData = new FormData();
          formData.append("file", blob, "recording.webm");

          const res = await fetch("http://127.0.0.1:8000/transcribe", {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || "Transcription failed");
          if (data.transcript) onQuestion(data.transcript);
        } catch (err) {
          console.error("Transcription error:", err);
        } finally {
          setVoiceState("idle");
        }
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setVoiceState("recording");
    } catch (err) {
      console.error("Mic access error:", err);
      setVoiceState("idle");
    }
  };

  // ── Mic button style ──────────────────────────────────────────────────────
  const micBtnStyle = {
    width: 38, height: 38, borderRadius: 10, border: "none",
    display: "flex", alignItems: "center", justifyContent: "center",
    cursor: !activeSpec || querying ? "not-allowed" : "pointer",
    flexShrink: 0,
    transition: "background 0.2s, box-shadow 0.2s, transform 0.1s",
    background: isRecording
      ? "#ef444420"
      : isTranscribing
        ? t.btnDisabledBg
        : dark ? "#ffffff0d" : "#e2e8f0",
    boxShadow: isRecording ? "0 0 0 3px #ef444440" : "none",
    opacity: !activeSpec || querying ? 0.45 : 1,
    animation: isRecording ? "pulseGlow 1.5s ease infinite" : "none",
  };

  const micColor = isRecording
    ? "#ef4444"
    : isTranscribing
      ? t.textMuted
      : t.textSecondary;

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
          placeholder={
            isTranscribing
              ? "Transcribing…"
              : activeSpec
                ? `Ask about ${activeSpec}…`
                : "Load a specification first, then ask a question…"
          }
          disabled={!activeSpec || voiceBusy}
          rows={3}
          style={{
            width: "100%", background: t.inputBg,
            border: `1px solid ${isRecording ? "#ef444455" : t.borderInput}`,
            borderRadius: 10,
            padding: "14px 16px", color: t.textPrimary,
            fontSize: 14, fontFamily: "'DM Sans', sans-serif",
            resize: "none", lineHeight: 1.65,
            transition: "border-color 0.2s, box-shadow 0.2s",
            opacity: (!activeSpec || isTranscribing) ? 0.45 : 1,
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
          alignItems: "center", marginTop: 12, gap: 8,
        }}>
          <span style={{ fontSize: 11, color: t.textMuted, fontFamily: "'DM Mono', monospace" }}>
            {activeSpec ? "⌘↵ to submit" : ""}
          </span>

          {/* Right side: mic + ask */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>

            {/* Mic button */}
            <button
              onClick={handleMic}
              disabled={!activeSpec || querying}
              title={isRecording ? "Stop recording" : "Record question"}
              style={micBtnStyle}
              onMouseEnter={e => {
                if (activeSpec && !querying && !isTranscribing)
                  e.currentTarget.style.transform = "scale(1.06)";
              }}
              onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
            >
              {isTranscribing
                ? <Spinner dark={dark} size={14} />
                : <MicIcon color={micColor} recording={isRecording} />}
            </button>

            {/* Ask button */}
            <button
              onClick={onQuery}
              disabled={querying || !question.trim() || !activeSpec || voiceBusy}
              style={primaryBtnStyle(t, querying || !question.trim() || !activeSpec || voiceBusy)}
              onMouseEnter={e => {
                if (!querying && question.trim() && activeSpec && !voiceBusy)
                  e.currentTarget.style.transform = "scale(1.02)";
              }}
              onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
            >
              {querying ? <><Spinner dark={dark} /> Thinking…</> : "Ask →"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

// ── Mic SVG icon ──────────────────────────────────────────────────────────────

function MicIcon({ color = "currentColor", recording = false }) {
  return (
    <svg
      width="16" height="16" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round"
      style={{ transition: "stroke 0.2s" }}
    >
      {recording
        // Square "stop" icon when recording
        ? <rect x="6" y="6" width="12" height="12" rx="2" fill={color} stroke="none" />
        // Mic icon when idle
        : <>
            <rect x="9" y="2" width="6" height="11" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="17" x2="12" y2="22" />
            <line x1="8" y1="22" x2="16" y2="22" />
          </>
      }
    </svg>
  );
}