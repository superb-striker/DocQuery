import { useState, useEffect } from "react";

import { theme } from "./src/theme";
import { classifyError } from "./src/utils/classifyError";

import { NoiseFilter } from "./src/components/ui/NoiseFilter";
import { ErrorBanner } from "./src/components/ui/ErrorBanner";
import { AppHeader } from "./src/components/AppHeader";
import { IngestPanel } from "./src/components/IngestPanel";
import { QueryPanel } from "./src/components/QueryPanel";
import { ResultPanel } from "./src/components/ResultPanel";

export default function DocQuery() {
  const [dark, setDark] = useState(true);
  const t = theme(dark);

  // Specification state
  const [loadedSpecs, setLoadedSpecs] = useState([]);
  const [activeSpec, setActiveSpec] = useState(null);

  // Query state
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [querying, setQuerying] = useState(false);
  const [animateResult, setAnimateResult] = useState(false);

  // Shared error state (ingest errors bubble up via onError; query errors set here directly)
  const [error, setError] = useState(null);

  // Load any already-ingested specs from the backend on mount
  useEffect(() => {
    fetch("http://127.0.0.1:8000/specifications")
      .then(r => r.json())
      .then(data => {
        if (data.specifications?.length) {
          setLoadedSpecs(data.specifications);
          setActiveSpec(data.specifications[0]);
        }
      })
      .catch(() => {}); // backend may not be up yet — fail silently
  }, []);

  // Re-trigger typewriter animation each time a new result arrives
  useEffect(() => {
    if (!result) return;
    const timer = setTimeout(() => {
      setAnimateResult(false);
      setTimeout(() => setAnimateResult(true), 10);
    }, 0);
    return () => clearTimeout(timer);
  }, [result]);

  // Called by IngestPanel when a spec is successfully loaded
  const handleSpecLoaded = ({ name }) => {
    setLoadedSpecs(prev => prev.includes(name) ? prev : [...prev, name]);
    setActiveSpec(name);
  };

  const handleQuery = async () => {
    if (!question.trim() || !activeSpec) return;
    setQuerying(true);
    setResult(null);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, specification_name: activeSpec }),
      });
      const data = await res.json();
      if (!res.ok) { setError(classifyError(data.detail, "Query")); return; }
      setResult(data);
    } catch (e) {
      setError(classifyError(e.message, "Query"));
    } finally {
      setQuerying(false);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;600&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes spin    { to { transform: rotate(360deg); } }
        @keyframes fadeUp  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 0 0 #3b82f622; } 50% { box-shadow: 0 0 0 8px #3b82f600; } }
        ::selection { background: #3b82f633; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${t.scrollThumb}; border-radius: 99px; }
        textarea:focus, input:focus { outline: none; }
        button { cursor: pointer; border: none; background: none; }
      `}</style>

      <NoiseFilter />

      <div style={{
        minHeight: "100vh",
        background: t.bg,
        fontFamily: "'DM Sans', sans-serif",
        color: t.textPrimary,
        backgroundImage: `${t.gradientRadial1}, ${t.gradientRadial2}`,
      }}>
        <AppHeader
          dark={dark}
          onToggleDark={() => setDark(d => !d)}
          loadedSpecs={loadedSpecs}
          t={t}
        />

        <main style={{ maxWidth: 860, margin: "0 auto", padding: "48px 24px 80px" }}>

          {/* Hero */}
          <div style={{ textAlign: "center", marginBottom: 56, animation: "fadeUp 0.6s ease both" }}>
            <h1
              key={dark ? "dark-title" : "light-title"}
              style={{
                fontFamily: "'Syne', sans-serif", fontWeight: 800,
                fontSize: "clamp(2rem, 5vw, 3.2rem)",
                letterSpacing: "-0.04em", lineHeight: 1.1,
                background: dark
                  ? "linear-gradient(135deg, #e2e8f0 30%, #60a5fa 65%, #818cf8 100%)"
                  : "linear-gradient(135deg, #0f172a 30%, #3b82f6 65%, #6366f1 100%)",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                WebkitTextFillColor: "transparent",
                color: "transparent",
                marginBottom: 16,
                transition: "none",
                willChange: "auto",
              }}
            >
              Stop digging through documentation.
            </h1>
            <p style={{ color: t.textSecondary, fontSize: 15, maxWidth: 480, margin: "0 auto", lineHeight: 1.7 }}>
              Make your OpenAPI documentation talk back.<br />
              Paste a link for instant, cited answers across your entire documentation — no matter how large the file.
            </p>
          </div>

          <IngestPanel
            dark={dark}
            t={t}
            onSpecLoaded={handleSpecLoaded}
            onError={setError}
          />

          <QueryPanel
            dark={dark}
            t={t}
            loadedSpecs={loadedSpecs}
            activeSpec={activeSpec}
            onSelectSpec={setActiveSpec}
            question={question}
            onQuestion={setQuestion}
            querying={querying}
            onQuery={handleQuery}
          />

          <ErrorBanner error={error} dark={dark} />

          <ResultPanel
            result={result}
            dark={dark}
            t={t}
            animate={animateResult}
          />
        </main>
      </div>
    </>
  );
}