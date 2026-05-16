import json, re
from typing import List, Dict
from sentence_transformers import util
from loggerConfig import get_logger
from sentence_transformers import CrossEncoder
from vectorStore import embedder

logger = get_logger("confidence")

hhem = CrossEncoder(
    "vectara/hallucination_evaluation_model",
    revision="hhem-1.0-open",
)

# Internal helpers
# Cosine baseline 
def _cosine_score(answer: str, chunks: List[Dict]) -> float:
    """Returns a 0–1 cosine similarity between the answer and the best chunk."""
    try:
        answer_emb = embedder.encode(answer, convert_to_tensor=True)
        chunk_embs = embedder.encode([c["text"] for c in chunks], convert_to_tensor=True)
        return float(util.cos_sim(answer_emb, chunk_embs).max())
    except Exception as e:
        logger.error(f"Cosine scoring failed: {e}")
        return 0.0

def _extract_known_fields(chunks: List[Dict]) -> set:
    """
    Parse the resolved schema fields that were baked into chunk text at
    ingest time. Looks for the 'all_fields' section in each chunk.
    """
    known = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        # Find the Request Body Fields JSON block
        match = re.search(r"Request Body Fields:\s*(\{.*?\})\s*(?:Responses:|Tags:|$)", 
                         text, re.DOTALL)
        if not match:
            continue
        try:
            body = json.loads(match.group(1))
            known.update(body.get("required_fields", []))
            known.update(body.get("all_fields", {}).keys())
        except json.JSONDecodeError:
            continue
    return known

def _flatten_keys(obj, prefix="") -> list:
    """Recursively extract all keys from a nested dict."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(k)
            keys.extend(_flatten_keys(v, prefix=k))
    return keys

def _verify_json_examples(answer: str, chunks: List[Dict]) -> float:
    known_fields = _extract_known_fields(chunks)
    if not known_fields:
        return 1.0  # no schema info available, don't penalise
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", answer)
    if not code_blocks:
        return 1.0
    total, grounded = 0, 0
    for block in code_blocks:
        try:
            obj = json.loads(block.strip())
            for field in _flatten_keys(obj):
                total += 1
                if field in known_fields:
                    grounded += 1
        except json.JSONDecodeError:
            continue
    return grounded / total if total > 0 else 1.0

# Hallucination scorer
def llm_judge_score(answer: str, chunks: List[Dict]) -> float | None:
    """
    Uses Vectara HHEM to assess factual consistency of
    the answer against retrieved chunks. Returns a 0–1 float (1 = grounded,
    0 = hallucinated). Returns None on error so caller falls back to cosine.
    """
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    try:
        # HHEM scores the prose claims
        answer_for_hhem = re.sub(r"```[\s\S]*?```", "", answer).strip()
        result = hhem.predict([(context, answer_for_hhem)])
        hhem_score = float(result[0] if hasattr(result, '__len__') else result)
        # JSON verifier scores the example fields
        json_score = _verify_json_examples(answer, chunks)
        # Blend: prose grounding matters more than field names
        score = 0.7 * hhem_score + 0.3 * json_score
        logger.info(f"HHEM: {hhem_score:.3f}, JSON field grounding: {json_score:.3f}, combined: {score:.3f}")
        return score
    except Exception as e:
        logger.error(f"HHEM judge failed: {e}")
        return None

# Public scorer
def score_confidence(query: str, answer: str, chunks: List[Dict]) -> int:
    """
    Hybrid confidence score (0–100).

    Combines:
      - LLM-as-judge factual consistency (70 % weight) - primary signal
      - Cosine similarity fallback        (30 % weight) - fast & robust

    If the LLM judge fails for any reason the function falls back to
    cosine-only so the pipeline never hard-fails here.
    """
    if not answer:
        logger.warning("Empty answer provided.")
        return 0
    if not chunks:
        logger.warning("No chunks provided for confidence scoring.")
        return 0
    cosine = _cosine_score(answer, chunks)
    judge = llm_judge_score(answer, chunks)
    if judge is None:
        # Judge call failed - trust cosine only
        confidence = round(cosine * 100)
        logger.info(f"Confidence (cosine-only fallback): {confidence}/100")
    else:
        combined = 0.7 * judge + 0.3 * cosine
        confidence = round(combined * 100)
        logger.info(
            f"Confidence (judge={judge:.2f}, cosine={cosine:.2f}, "
            f"combined={combined:.2f}): {confidence}/100"
        )
    return confidence