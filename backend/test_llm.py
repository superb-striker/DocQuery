import json
import os
import unittest
from unittest.mock import patch

from llm import (
    LLMProviderError,
    build_prompt,
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
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_completion_tokens"], 900)
        self.assertNotIn("max_tokens", payload)
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
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            LLMProviderError, "GEMINI_API_KEY"
        ):
            build_llm_request("hello", "gemini")

    def test_provider_catalog_does_not_expose_keys(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}, clear=True):
            groq = next(item for item in provider_catalog() if item["id"] == "groq")

        self.assertTrue(groq["available"])
        self.assertNotIn("secret", str(groq))

    def test_workflow_prompt_requires_every_planned_step(self):
        steps = [
            "POST create customer",
            "POST attach payment method",
            "POST create subscription",
        ]
        chunks = [{
            "text": "Method: POST\nPath: /v1/customers",
            "metadata": {
                "method": "POST",
                "path": "/v1/customers",
                "retrieval_query": steps[0],
                "workflow_steps": steps,
            },
        }]

        prompt = build_prompt("create a subscribed customer", chunks)

        self.assertIn("3-step workflow", prompt)
        self.assertIn("Repeat through **Step 3**", prompt)
        self.assertIn("3. POST create subscription", prompt)
        self.assertIn("[Evidence retrieved for: POST create customer]", prompt)

    def test_evidence_budget_compacts_large_schemas_without_dropping_steps(self):
        steps = ["POST create customer", "POST create subscription"]
        body = {
            "required_fields": [],
            "all_fields": {
                "customer": {
                    "type": "string",
                    "description": "The customer identifier used for the subscription.",
                },
                "items": {
                    "type": "array",
                    "description": "Prices and quantities for the subscription.",
                },
                "unrelated_configuration": {
                    "type": "object",
                    "description": "x" * 6000,
                },
            },
        }
        chunks = []
        for index, step in enumerate(steps):
            chunks.append({
                "text": (
                    f"Operation: POST /step-{index + 1}\nSummary: {step}\n"
                    f"Request Body Fields: {json.dumps(body)}\nResponses: 200"
                ),
                "metadata": {
                    "retrieval_query": step,
                    "workflow_steps": steps,
                },
            })

        prompt = build_prompt(
            "create a customer and subscription", chunks, max_evidence_chars=2400
        )

        self.assertIn("POST /step-1", prompt)
        self.assertIn("POST /step-2", prompt)
        self.assertIn("- customer (string, optional)", prompt)
        self.assertIn("- items (array, optional)", prompt)
        self.assertLess(len(prompt), 6000)


if __name__ == "__main__":
    unittest.main()
