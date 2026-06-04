import os
import unittest
from unittest.mock import patch

from llm import (
    LLMProviderError,
    build_llm_request,
    get_provider_config,
    provider_catalog,
)


class LLMProviderTests(unittest.TestCase):
    def test_groq_uses_smaller_evidence_budget(self):
        self.assertEqual(get_provider_config("groq")["max_chunks"], 8)
        self.assertEqual(get_provider_config("gemini")["max_chunks"], 15)

    def test_groq_request_uses_configured_model_and_compatible_stops(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            url, headers, payload = build_llm_request("hello", "groq")

        self.assertEqual(url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(payload["model"], "openai/gpt-oss-120b")
        self.assertLessEqual(len(payload["stop"]), 4)

    def test_gemini_request_uses_openai_compatible_endpoint(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            url, _, payload = build_llm_request("hello", "gemini")

        self.assertEqual(
            url,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )
        self.assertEqual(payload["model"], "gemini-3.8-flash")
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_missing_key_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True) and self.assertRaisesRegex(LLMProviderError, "GEMINI_API_KEY"):
                build_llm_request("hello", "gemini")

    def test_provider_catalog_does_not_expose_keys(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}, clear=True):
            groq = next(item for item in provider_catalog() if item["id"] == "groq")

        self.assertTrue(groq["available"])
        self.assertNotIn("secret", str(groq))


if __name__ == "__main__":
    unittest.main()
