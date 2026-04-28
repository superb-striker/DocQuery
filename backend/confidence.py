from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from loggerConfig import get_logger
from vectorStore import embedder

logger = get_logger("confidence")

# Scoring
def score_confidence(query: str, answer: str, chunks: List[Dict]) -> int:
    # Compare the answer's embedding against retrieved chunks. Returns confidence score (0–100).
    if not answer:
        logger.warning("Empty answer provided.")
        return 0
    if not chunks:
        logger.warning("No chunks provided for confidence scoring.")
        return 0
    try:
        logger.debug("Encoding answer...")
        answer_embedding = embedder.encode(answer, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Failed to encode answer: {e}")
        return 0
    try:
        chunk_texts = [c["text"] for c in chunks]
    except KeyError as e:
        logger.error(f"Missing 'text' key in chunks: {e}")
        return 0
    try:
        logger.debug(f"Encoding {len(chunk_texts)} chunks...")
        chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
    except Exception as e:
        logger.error(f"Failed to encode chunks: {e}")
        return 0
    try:
        similarities = util.cos_sim(answer_embedding, chunk_embeddings)
        max_score = float(similarities.max())
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        return 0
    confidence = round(max_score * 100)
    logger.info(f"Confidence score: {confidence}/100")
    return confidence