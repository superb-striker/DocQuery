import { useState } from "react";
import { Spinner } from "./ui/Spinner";
import { primaryBtnStyle, inputBaseStyle } from "../theme";

/**
 * Panel for loading an OpenAPI specification either by URL or file upload.
 * Manages its own local UI state (input values, loading, messages).
 * Calls `onSpecLoaded({ name, chunksStored })` on success so the parent
 * can update its loaded-specs list.
 *
 * @prop {boolean}  dark
 * @prop {object}   t               - Theme tokens.
 * @prop {function} onSpecLoaded    - ({ name: string, chunksStored: number }) => void
 * @prop {function} onError         - (classifiedError) => void
 */
export function IngestPanel({ dark, t, onSpecLoaded, onError }) {
  const [specUrl, setSpecUrl] = useState("");
  const [specFile, setSpecFile] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [ingesting, setIngesting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [ingestMsg, setIngestMsg] = useState(null);
  const [uploadMsg, setUploadMsg] = useState(null);

  // ── URL ingest ──────────────────────────────────────────────────────────────
  const handleIngest = async () => {
    if (!specUrl.trim()) return;
    setIngesting(true);
    setIngestMsg(null);
    onError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ specification_url: specUrl }),
      });
      const data = await res.json();
      if (!res.ok) { onError({ msg: data.detail, hint: null }); return; }
      setIngestMsg(`✓ ${data.chunks_stored} endpoints indexed for "${data.specification_name}"`);
      onSpecLoaded({ name: data.specification_name, chunksStored: data.chunks_stored });
      setSpecUrl("");
    } catch (e) {
      onError({ msg: e.message, hint: "Make sure uvicorn is running on http://127.0.0.1:8000" });
    } finally {
      setIngesting(false);
    }
  };

  // ── File upload ─────────────────────────────────────────────────────────────
  const handleFileIngest = async () => {
    if (!specFile) return;
    setUploading(true);
    setUploadMsg(null);
    onError(null);
    try {
      const formData = new FormData();
      formData.append("file", specFile);
      const res = await fetch("http://localhost:8000/ingest/file", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) { onError({ msg: data.detail, hint: null }); return; }
      setUploadMsg(`✓ ${data.chunks_stored} endpoints indexed for "${data.specification_name}"`);
      onSpecLoaded({ name: data.specification_name, chunksStored: data.chunks_stored });
      setSpecFile(null);
    } catch (e) {
      onError({ msg: e.message, hint: "Make sure uvicorn is running on http://localhost:8000" });
    } finally {
      setUploading(false);
    }
  };

  const clearFile = () => {
    setSpecFile(null);
    setFileInputKey(k => k + 1); // remount the input so the same file can be re-selected
  };

  return (
    <div style={{
      background: t.bgCard, border: `1px solid ${t.borderCard}`,
      borderRadius: 16, padding: 24, marginBottom: 20,
      animation: "fadeUp 0.6s 0.1s ease both", opacity: 0,
      animationFillMode: "forwards",
      boxShadow: dark ? "none" : "0 2px 16px #3b82f610",
    }}>
      <div style={{
        fontSize: 11, fontFamily: "'DM Mono', monospace",
        color: t.textLabel, letterSpacing: "0.12em", marginBottom: 16, fontWeight: 600,
      }}>
        LOAD SPECIFICATION
      </div>

      {/* URL row */}
      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        <input
          value={specUrl}
          onChange={e => setSpecUrl(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleIngest()}
          placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
          style={inputBaseStyle(t)}
          onFocus={e => {
            e.target.style.borderColor = t.borderInputFocus;
            e.target.style.boxShadow = `0 0 0 3px ${t.accentGlow}`;
          }}
          onBlur={e => {
            e.target.style.borderColor = t.borderInput;
            e.target.style.boxShadow = "none";
          }}
        />
        <button
          onClick={handleIngest}
          disabled={ingesting || !specUrl.trim()}
          style={primaryBtnStyle(t, ingesting || !specUrl.trim())}
          onMouseEnter={e => { if (!ingesting && specUrl.trim()) e.currentTarget.style.transform = "scale(1.02)"; }}
          onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
        >
          {ingesting ? <><Spinner dark={dark} /> Indexing…</> : "→ Ingest"}
        </button>
      </div>

      {ingestMsg && (
        <div style={{
          marginBottom: 14, fontSize: 12, color: "#22c55e",
          fontFamily: "'DM Mono', monospace", animation: "fadeUp 0.3s ease",
        }}>
          {ingestMsg}
        </div>
      )}

      {/* OR divider */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12, marginBottom: 16,
        color: t.textMuted, fontSize: 11, fontFamily: "'DM Mono', monospace",
      }}>
        <div style={{ flex: 1, height: 1, background: dark ? "#ffffff0a" : "#e2e8f0" }} />
        OR UPLOAD .JSON FILE
        <div style={{ flex: 1, height: 1, background: dark ? "#ffffff0a" : "#e2e8f0" }} />
      </div>

      {/* File upload row */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label
          style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "9px 16px", borderRadius: 10, cursor: "pointer",
            background: t.inputBg, border: `1px solid ${t.borderInput}`,
            fontSize: 12, color: t.textSecondary, fontFamily: "'DM Mono', monospace",
            transition: "border-color 0.2s",
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = t.borderInputFocus; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = t.borderInput; }}
        >
          <input
            key={fileInputKey}
            type="file" accept=".json"
            onChange={e => setSpecFile(e.target.files[0])}
            style={{ display: "none" }}
          />
          📂 {specFile ? specFile.name : "Choose file…"}
        </label>

        {specFile && (
          <button
            onClick={handleFileIngest}
            disabled={uploading}
            style={primaryBtnStyle(t, uploading)}
            onMouseEnter={e => { if (!uploading) e.currentTarget.style.transform = "scale(1.02)"; }}
            onMouseLeave={e => { e.currentTarget.style.transform = "scale(1)"; }}
          >
            {uploading ? <><Spinner dark={dark} /> Uploading…</> : "→ Upload"}
          </button>
        )}

        {specFile && !uploading && (
          <button
            onClick={clearFile}
            style={{
              fontSize: 11, color: t.textMuted,
              fontFamily: "'DM Mono', monospace", padding: "4px 8px",
            }}
          >
            ✕ clear
          </button>
        )}
      </div>

      {uploadMsg && (
        <div style={{
          marginTop: 12, fontSize: 12, color: "#22c55e",
          fontFamily: "'DM Mono', monospace", animation: "fadeUp 0.3s ease",
        }}>
          {uploadMsg}
        </div>
      )}
    </div>
  );
}