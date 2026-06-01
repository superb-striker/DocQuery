import unittest

from backend.agent import run_confidence_retry


def chunk(endpoint: str):
    method, path = endpoint.split(" ", 1)
    return {
        "text": endpoint,
        "metadata": {"method": method, "path": path, "summary": endpoint},
    }


class ConfidenceRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_confidence_answer_does_not_retry(self):
        retrieve_calls = []

        def retrieve(query, specification_name):
            retrieve_calls.append((query, specification_name))
            return [chunk("GET /pets")]

        async def generate(question, chunks):
            return "grounded answer"

        async def rewrite(question, answer, chunks):
            self.fail("rewrite should not run for a confident answer")

        result = await run_confidence_retry(
            "list pets", "petstore", retrieve, generate,
            lambda question, answer, chunks: 85,
            rewrite,
        )

        self.assertEqual(result.answer, "grounded answer")
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.retried)
        self.assertEqual(retrieve_calls, [("list pets", "petstore")])

    async def test_low_confidence_answer_uses_better_retry(self):
        first_chunks = [chunk("GET /pets")]
        second_chunks = [chunk("POST /pets")]

        def retrieve(query, specification_name):
            return second_chunks if query == "create pet POST request body" else first_chunks

        async def generate(question, chunks):
            return "better answer" if chunks is second_chunks else "weak answer"

        def score(question, answer, chunks):
            return 91 if answer == "better answer" else 42

        async def rewrite(question, answer, chunks):
            return "create pet POST request body"

        result = await run_confidence_retry(
            "how do I add a pet?", "petstore", retrieve, generate, score, rewrite
        )

        self.assertEqual(result.answer, "better answer")
        self.assertEqual(result.chunks, second_chunks)
        self.assertEqual(result.confidence, 91)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.retried)
        self.assertEqual(result.selected_attempt, 2)

    async def test_worse_retry_preserves_first_answer(self):
        calls = 0

        def retrieve(query, specification_name):
            nonlocal calls
            calls += 1
            return [chunk(f"GET /attempt-{calls}")]

        async def generate(question, chunks):
            return chunks[0]["text"]

        def score(question, answer, chunks):
            return 40 if "attempt-1" in answer else 30

        async def rewrite(question, answer, chunks):
            return "different retrieval query"

        result = await run_confidence_retry(
            "question", "spec", retrieve, generate, score, rewrite
        )

        self.assertEqual(result.answer, "GET /attempt-1")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.selected_attempt, 1)

    async def test_retry_failure_preserves_first_answer(self):
        def retrieve(query, specification_name):
            return [chunk("GET /pets")]

        async def generate(question, chunks):
            return "first answer"

        async def rewrite(question, answer, chunks):
            raise RuntimeError("rewrite provider unavailable")

        result = await run_confidence_retry(
            "list pets", "petstore", retrieve, generate,
            lambda question, answer, chunks: 25,
            rewrite,
        )

        self.assertEqual(result.answer, "first answer")
        self.assertEqual(result.confidence, 25)
        self.assertEqual(result.selected_attempt, 1)

    async def test_compound_question_merges_deduplicated_evidence(self):
        calls = []

        def retrieve(query, specification_name):
            calls.append(query)
            shared = chunk("GET /pets/{id}")
            if query == "create a pet":
                return [chunk("POST /pets"), shared]
            return [shared, chunk("DELETE /pets/{id}")]

        async def generate(question, chunks):
            return ",".join(item["metadata"]["path"] for item in chunks)

        async def rewrite(question, answer, chunks):
            self.fail("a confident planned answer should not retry")

        async def decompose(question):
            return ["create a pet", "delete a pet"]

        result = await run_confidence_retry(
            "How do I create and then delete a pet?",
            "petstore",
            retrieve,
            generate,
            lambda question, answer, chunks: 90,
            rewrite,
            decompose=decompose,
        )

        self.assertEqual(calls, ["create a pet", "delete a pet"])
        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(result.subqueries, ["create a pet", "delete a pet"])

    async def test_invalid_plan_falls_back_to_original_question(self):
        calls = []

        def retrieve(query, specification_name):
            calls.append(query)
            return [chunk("GET /pets")]

        async def generate(question, chunks):
            return "answer"

        async def rewrite(question, answer, chunks):
            return question

        async def decompose(question):
            return ["", 42, "   "]

        result = await run_confidence_retry(
            "list pets", "petstore", retrieve, generate,
            lambda question, answer, chunks: 80,
            rewrite,
            decompose=decompose,
        )

        self.assertEqual(calls, ["list pets"])
        self.assertEqual(result.subqueries, ["list pets"])


if __name__ == "__main__":
    unittest.main()
