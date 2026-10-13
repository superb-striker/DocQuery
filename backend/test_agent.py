import unittest

from backend.agent import (
    _workflow_adjusted_score,
    order_chunks_by_answer_citations,
    resolve_follow_up,
    run_agent_graph,
    run_confidence_retry,
    select_specifications,
)


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
            return "**Step 1: Create**\n**Step 2: Delete**"

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
        self.assertEqual(
            result.chunks[0]["metadata"]["workflow_steps"],
            ["create a pet", "delete a pet"],
        )
        self.assertEqual(
            result.chunks[0]["metadata"]["retrieval_query"], "create a pet"
        )

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

    async def test_explicit_action_prioritizes_matching_endpoint(self):
        candidates = [
            chunk("POST /v1/subscriptions/{id}/migrate"),
            chunk("POST /v1/subscriptions"),
        ]
        candidates[0]["metadata"]["summary"] = "Migrate a subscription"
        candidates[1]["metadata"]["summary"] = "Create a subscription"

        def retrieve(query, specification_name):
            return candidates

        async def generate(question, chunks):
            return chunks[0]["metadata"]["path"]

        async def rewrite(question, answer, chunks):
            return question

        async def decompose(question):
            return ["POST create subscription for customer"]

        result = await run_confidence_retry(
            "create a subscription", "stripe", retrieve, generate,
            lambda question, answer, chunks: 90,
            rewrite,
            decompose=decompose,
        )

        self.assertEqual(result.answer, "/v1/subscriptions")

    def test_workflow_score_penalizes_missing_steps(self):
        steps = ["create customer", "attach payment method", "create subscription"]

        self.assertEqual(
            _workflow_adjusted_score(90, "**Step 1: Create customer**", steps),
            30,
        )
        self.assertEqual(
            _workflow_adjusted_score(
                90,
                "**Step 1: Customer**\n**Step 2: Attach**\n**Step 3: Subscription**",
                steps,
            ),
            90,
        )

    async def test_workflow_retry_reuses_balanced_evidence(self):
        retrieve_calls = []
        generation_chunks = []

        def retrieve(query, specification_name):
            retrieve_calls.append(query)
            if "customer" in query:
                return [chunk("POST /v1/customers")]
            if "payment" in query:
                return [chunk("POST /v1/payment_methods/{id}/attach")]
            return [chunk("POST /v1/subscriptions")]

        answers = iter([
            "**Step 1: Customer**\n**Step 2: Payment method**",
            "**Step 1: Customer**\n**Step 2: Payment method**\n**Step 3: Subscription**",
        ])

        async def generate(question, chunks):
            generation_chunks.append(chunks)
            return next(answers)

        async def rewrite(question, answer, chunks):
            self.fail("workflow retries must not collapse into one rewritten query")

        async def decompose(question):
            return ["create customer", "attach payment method", "create subscription"]

        result = await run_confidence_retry(
            "create a subscribed customer", "stripe", retrieve, generate,
            lambda question, answer, chunks: 90,
            rewrite,
            decompose=decompose,
        )

        self.assertEqual(result.selected_attempt, 2)
        self.assertEqual(result.confidence, 90)
        self.assertEqual(len(generation_chunks), 2)
        self.assertIs(generation_chunks[0], generation_chunks[1])
        self.assertEqual(len(retrieve_calls), 3)

    async def test_complete_workflow_uses_provider_specific_threshold(self):
        generate_calls = 0

        def retrieve(query, specification_name):
            return [chunk("POST /operation")]

        async def generate(question, chunks):
            nonlocal generate_calls
            generate_calls += 1
            return "**Step 1: One**\n**Step 2: Two**\n**Step 3: Three**"

        async def rewrite(question, answer, chunks):
            self.fail("a complete workflow above its threshold should not retry")

        async def decompose(question):
            return ["one", "two", "three"]

        result = await run_confidence_retry(
            "three steps", "spec", retrieve, generate,
            lambda question, answer, chunks: 65,
            rewrite,
            decompose=decompose,
            workflow_confidence_threshold=60,
        )

        self.assertEqual(result.confidence, 65)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(generate_calls, 1)

    async def test_workflow_aware_scorer_is_not_penalized_twice(self):
        def retrieve(query, specification_name):
            return [chunk(f"POST /{query}")]

        async def generate(question, chunks):
            return "**Step 1: One**\n**Step 2: Two**"

        def score(question, answer, chunks):
            return 65

        score.includes_workflow_completeness = True

        async def rewrite(question, answer, chunks):
            self.fail("the already-combined score is above the workflow threshold")

        async def decompose(question):
            return ["one", "two", "three"]

        result = await run_confidence_retry(
            "three steps", "spec", retrieve, generate, score, rewrite,
            decompose=decompose,
            workflow_confidence_threshold=60,
        )

        self.assertEqual(result.confidence, 65)
        self.assertEqual(result.attempts, 1)


class AgentGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_graph_contextualizes_routes_decomposes_and_retrieves(self):
        retrievals = []

        async def contextualize(question, history):
            self.assertEqual(len(history), 1)
            return "create and then list invoices"

        async def route(question, available):
            self.assertEqual(question, "create and then list invoices")
            return ["billing"]

        async def decompose(question):
            return ["create invoice", "list invoices"]

        def retrieve(question, selected_specs):
            retrievals.append((question, selected_specs))
            method = "POST" if question.startswith("create") else "GET"
            return [chunk(f"{method} /invoices")]

        async def generate(question, chunks):
            return "**Step 1: Create**\n**Step 2: List**"

        async def rewrite(question, answer, chunks):
            self.fail("a confident answer should not be rewritten")

        result = await run_agent_graph(
            question="What about invoices?",
            history=[{"question": "How does billing work?", "answer": "Use the API."}],
            specification_name="__all__",
            available_specifications=["identity", "billing", "orders", "shipping"],
            contextualize=contextualize,
            route=route,
            retrieve=retrieve,
            generate=generate,
            score=lambda question, answer, chunks: 90,
            rewrite=rewrite,
            decompose=decompose,
        )

        self.assertEqual(result.subqueries, ["create invoice", "list invoices"])
        self.assertEqual(retrievals, [
            ("create invoice", ["billing"]),
            ("list invoices", ["billing"]),
        ])
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.confidence, 90)
        self.assertEqual(result.resolved_question, "create and then list invoices")
        self.assertEqual(result.selected_specs, ["billing"])


class CitationOrderingTests(unittest.TestCase):
    def test_answer_endpoints_come_before_supporting_evidence(self):
        chunks = [
            chunk("POST /v1/customer_sessions"),
            chunk("POST /v1/customers"),
            chunk("POST /v1/payment_methods/{payment_method}/attach"),
            chunk("POST /v1/setup_intents"),
            chunk("POST /v1/customers/{customer}/subscriptions"),
            chunk("POST /v1/subscriptions"),
        ]
        answer = """**Step 1: Customer**
**Endpoint:** POST /v1/customers
**Step 2: Payment method**
**Endpoint:** `POST /v1/payment_methods/{payment_method}/attach`
**Step 3: Subscription**
**Endpoint:** POST /v1/customers/{customer}/subscriptions
"""

        ordered = order_chunks_by_answer_citations(answer, chunks)
        endpoints = [
            f"{item['metadata']['method']} {item['metadata']['path']}"
            for item in ordered
        ]

        self.assertEqual(endpoints[:3], [
            "POST /v1/customers",
            "POST /v1/payment_methods/{payment_method}/attach",
            "POST /v1/customers/{customer}/subscriptions",
        ])
        self.assertEqual(endpoints[3:], [
            "POST /v1/customer_sessions",
            "POST /v1/setup_intents",
            "POST /v1/subscriptions",
        ])

    def test_no_answer_citations_preserves_retrieval_order(self):
        chunks = [chunk("GET /first"), chunk("GET /second")]

        ordered = order_chunks_by_answer_citations("No endpoint found.", chunks)

        self.assertIs(ordered, chunks)


class FollowUpResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_is_resolved_from_recent_history(self):
        history_seen = []

        async def contextualize(question, history):
            history_seen.extend(history)
            return "Which request fields does POST /pets require?"

        resolved = await resolve_follow_up(
            "What fields does that need?",
            [{"question": "How do I create a pet?", "answer": "Use POST /pets."}],
            contextualize,
        )

        self.assertEqual(resolved, "Which request fields does POST /pets require?")
        self.assertEqual(len(history_seen), 1)

    async def test_empty_history_skips_contextualization(self):
        async def contextualize(question, history):
            self.fail("contextualization should not run without history")

        resolved = await resolve_follow_up(
            "How do I list pets?", [], contextualize
        )

        self.assertEqual(resolved, "How do I list pets?")

    async def test_contextualization_failure_preserves_question(self):
        async def contextualize(question, history):
            raise RuntimeError("provider unavailable")

        resolved = await resolve_follow_up(
            "What does that return?",
            [{"question": "List pets", "answer": "Use GET /pets."}],
            contextualize,
        )

        self.assertEqual(resolved, "What does that return?")


class SpecificationRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_spec_set_uses_all_without_router_call(self):
        async def route(question, available):
            self.fail("router should not run for three or fewer specs")

        selected = await select_specifications(
            "compare authentication", ["stripe", "github"], route
        )

        self.assertEqual(selected, ["stripe", "github"])

    async def test_router_selection_is_validated_and_bounded(self):
        async def route(question, available):
            return ["GITHUB", "missing", "stripe", "github", "phantom"]

        selected = await select_specifications(
            "compare APIs",
            ["stripe", "github", "phantom", "petstore"],
            route,
            max_specs=3,
        )

        self.assertEqual(selected, ["github", "stripe", "phantom"])

    async def test_router_failure_has_bounded_fallback(self):
        async def route(question, available):
            raise RuntimeError("provider unavailable")

        selected = await select_specifications(
            "question", ["one", "two", "three", "four"], route
        )

        self.assertEqual(selected, ["one", "two", "three"])

    async def test_same_endpoint_from_different_specs_is_not_deduplicated(self):
        def retrieve(query, specification_name):
            first = chunk("GET /health")
            first["metadata"]["collection_name"] = "service_a"
            second = chunk("GET /health")
            second["metadata"]["collection_name"] = "service_b"
            return [first, second]

        async def generate(question, chunks):
            return "comparison"

        async def rewrite(question, answer, chunks):
            return question

        result = await run_confidence_retry(
            "compare health endpoints", "__all__", retrieve, generate,
            lambda question, answer, chunks: 90,
            rewrite,
        )

        self.assertEqual(len(result.chunks), 2)


if __name__ == "__main__":
    unittest.main()
