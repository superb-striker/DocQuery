import re
from collections.abc import Callable
from dataclasses import dataclass


Chunks = list[dict]
StepScorer = Callable[[str, Chunks], int]


@dataclass
class WorkflowConfidence:
    score: int
    step_scores: list[int]
    expected_steps: int
    answered_steps: int


def score_workflow_confidence(
    answer: str,
    chunks: Chunks,
    score_step: StepScorer,
) -> WorkflowConfidence | None:
    """Score each workflow answer section only against its labelled evidence."""
    workflow_steps = _workflow_steps(chunks)
    if len(workflow_steps) <= 1:
        return None

    answer_sections = _answer_step_sections(answer)
    evidence_by_step = {step: [] for step in workflow_steps}
    for chunk in chunks:
        retrieval_query = chunk.get("metadata", {}).get("retrieval_query")
        if retrieval_query in evidence_by_step:
            evidence_by_step[retrieval_query].append(chunk)

    step_scores = []
    answered_steps = 0
    for index, step in enumerate(workflow_steps, 1):
        section = answer_sections.get(index)
        evidence = evidence_by_step[step]
        if not section or not evidence:
            step_scores.append(0)
            continue
        answered_steps += 1
        step_scores.append(max(0, min(100, score_step(section, evidence))))

    return WorkflowConfidence(
        score=round(sum(step_scores) / len(workflow_steps)),
        step_scores=step_scores,
        expected_steps=len(workflow_steps),
        answered_steps=answered_steps,
    )


def _workflow_steps(chunks: Chunks) -> list[str]:
    for chunk in chunks:
        steps = chunk.get("metadata", {}).get("workflow_steps")
        if isinstance(steps, list):
            normalized = [str(step) for step in steps if str(step).strip()]
            if normalized:
                return normalized
    return []


def _answer_step_sections(answer: str) -> dict[int, str]:
    headings = list(re.finditer(
        r"(?:\*\*)?Step\s+(\d+)\s*:[^\n]*(?:\*\*)?",
        answer,
        re.IGNORECASE,
    ))
    sections = {}
    for position, heading in enumerate(headings):
        step_number = int(heading.group(1))
        end = headings[position + 1].start() if position + 1 < len(headings) else len(answer)
        sections.setdefault(step_number, answer[heading.start():end].strip())
    return sections
