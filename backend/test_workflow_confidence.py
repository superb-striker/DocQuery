import unittest

from backend.workflow_confidence import score_workflow_confidence


def workflow_chunk(step: str, endpoint: str, steps: list[str]) -> dict:
    method, path = endpoint.split(" ", 1)
    return {
        "text": endpoint,
        "metadata": {
            "method": method,
            "path": path,
            "retrieval_query": step,
            "workflow_steps": steps,
        },
    }


class WorkflowConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.steps = [
            "create customer",
            "attach payment method",
            "create subscription",
        ]
        self.chunks = [
            workflow_chunk(self.steps[0], "POST /customers", self.steps),
            workflow_chunk(self.steps[1], "POST /payment_methods/{id}/attach", self.steps),
            workflow_chunk(self.steps[2], "POST /subscriptions", self.steps),
        ]

    def test_each_step_is_scored_against_its_own_evidence(self):
        expected_scores = {
            "/customers": 80,
            "/payment_methods/{id}/attach": 70,
            "/subscriptions": 90,
        }
        seen = []

        def score_step(section, evidence):
            path = evidence[0]["metadata"]["path"]
            seen.append((section, path))
            return expected_scores[path]

        result = score_workflow_confidence(
            """**Step 1: Customer**\nfirst
**Step 2: Payment**\nsecond
**Step 3: Subscription**\nthird""",
            self.chunks,
            score_step,
        )

        self.assertEqual(result.score, 80)
        self.assertEqual(result.step_scores, [80, 70, 90])
        self.assertEqual([path for _, path in seen], list(expected_scores))
        self.assertTrue(all(f"Step {index}" in section for index, (section, _) in enumerate(seen, 1)))

    def test_missing_step_contributes_zero_for_completeness(self):
        result = score_workflow_confidence(
            "**Step 1: Customer**\nfirst\n**Step 3: Subscription**\nthird",
            self.chunks,
            lambda section, evidence: 90,
        )

        self.assertEqual(result.step_scores, [90, 0, 90])
        self.assertEqual(result.score, 60)
        self.assertEqual(result.answered_steps, 2)

    def test_missing_labelled_evidence_scores_step_as_zero(self):
        result = score_workflow_confidence(
            "**Step 1: Customer**\nfirst\n**Step 2: Payment**\nsecond\n"
            "**Step 3: Subscription**\nthird",
            self.chunks[:2],
            lambda section, evidence: 90,
        )

        self.assertEqual(result.step_scores, [90, 90, 0])
        self.assertEqual(result.score, 60)

    def test_single_query_returns_none_for_normal_scoring(self):
        chunks = [{"text": "GET /pets", "metadata": {}}]

        result = score_workflow_confidence("answer", chunks, lambda answer, evidence: 90)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
