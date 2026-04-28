import { useState, useEffect } from "react";

/**
 * Animates text appearing one character at a time.
 * Resets automatically when `text` changes.
 *
 * @param {string}  text    - The full string to type out.
 * @param {number}  speed   - Milliseconds between each character (default 18).
 * @param {boolean} trigger - Set to true to start/restart the animation.
 * @returns {string} The currently displayed partial string.
 */
export function useTypewriter(text, speed = 18, trigger = false) {
  const [displayed, setDisplayed] = useState("");
  const [prevText, setPrevText] = useState("");
  // Reset during render when text changes so the first render after a new
  // result already shows an empty string (no stale content flash).
  if (text !== prevText) {
    setPrevText(text);
    setDisplayed("");
  }
  useEffect(() => {
    if (!trigger || !text) return;
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, speed);
    return () => clearInterval(interval);
  }, [text, trigger, speed]);
  return displayed;
}