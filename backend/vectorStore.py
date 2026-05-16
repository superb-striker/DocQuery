import os, pickle, chromadb
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from loggerConfig import get_logger

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
logger = get_logger("vector_store")

BM25 = "./bm25_indexes"
os.makedirs(BM25, exist_ok=True)

# Embedding model, cross-encoder model & ChromaDB initialization 
try:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load embedding model: {e}")
    raise

try:
    logger.info("Loading cross-encoder re-ranking model...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-V2")
    logger.info("Cross-encoder loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load cross-encoder: {e}")
    raise

try:
    logger.info("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path="./chromaDB")
    logger.info("ChromaDB client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB: {e}")
    raise

# Helper
def normalize_name(specification_name: str) -> str:
    """Normalize a specification name to a valid ChromaDB collection name."""
    name = specification_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    name = ''.join(c for c in name if c.isalnum() or c == '_')
    name = name.strip('_')[:50]
    # Ensure minimum 3 characters
    if len(name) < 3:
        name = name + "_api"    
    return name

def bm25_path(specification_name: str) -> str:
    return os.path.join(BM25, f"{normalize_name(specification_name)}.pkl")
 
def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization - fast and good enough for BM25."""
    return text.lower().split()

# Store 
def store_chunks(chunks: List[Dict], specification_name: str) -> int:
    if not chunks:
        logger.warning("No chunks provided for storage.")
        return 0
    try:
        name = normalize_name(specification_name)
        # Get the collection if it exists, otherwise create it
        collection = client.get_or_create_collection(name)
    except Exception as e:
        logger.error(f"Failed to get/create collection: {e}")
        return 0
    try:
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{specification_name}_{i}" for i in range(len(chunks))]
    except KeyError as e:
        logger.error(f"Missing expected key in chunks: {e}")
        return 0
    try:
        logger.info(f"Embedding {len(chunks)} chunks...")
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return 0
    try:
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Stored {len(chunks)} chunks for '{specification_name}'")
    except Exception as e:
        logger.error(f"Failed to store chunks: {e}")
        return 0
    try:
        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        payload = {"bm25": bm25, "chunks": chunks}
        with open(bm25_path(specification_name), "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"BM25 index built and saved for '{specification_name}'.")
    except Exception as e:
        logger.error(f"Failed to build/save BM25 index: {e}")
        # ChromaDB succeeded - partial success is still useful
    return len(chunks)

def load_bm25(specification_name: str):
    """Load persisted BM25 index. Returns (bm25, chunks) or (None, None)."""
    path = bm25_path(specification_name)
    if not os.path.exists(path):
        logger.warning(f"BM25 index not found for '{specification_name}'.")
        return None, None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload["bm25"], payload["chunks"]
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        return None, None

# Reciprocal Rank Fusion 
 
def rrf_fuse(dense_chunks: List[Dict], bm25_chunks: List[Dict], k: int = 60) -> List[Dict]:
    """
    Fuse two ranked lists with Reciprocal Rank Fusion.
    Deduplicates by (method, path) so the same endpoint isn't scored twice.
    """
    scores: Dict[str, float] = {}
    index: Dict[str, Dict] = {}
    def _key(chunk: Dict) -> str:
        m = chunk["metadata"].get("method", "")
        p = chunk["metadata"].get("path", "")
        return f"{m}:{p}"
    for rank, chunk in enumerate(dense_chunks):
        key = _key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        index[key] = chunk
    for rank, chunk in enumerate(bm25_chunks):
        key = _key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        index[key] = chunk
    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [index[k] for k in ranked_keys]
    
# Cross-encoder re-ranking
def rerank(query: str, chunks: List[Dict], top_n: int = 5) -> List[Dict]:
    if not chunks:
        return chunks
    try:
        pairs = [(query, c["text"]) for c in chunks]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = [chunk for _, chunk in ranked[:top_n]]
        logger.info(f"Re-ranked {len(chunks)} chunks -> kept top {len(top)}.")
        return top
    except Exception as e:
        logger.error(f"Re-ranking failed, returning original chunks: {e}")
        return chunks[:top_n]
    
# Hybrid retrieval
def retrieve_hybrid(
    query: str,
    specification_name: str,
    num_results: int = 20,
    rerank_top_n: int = 5,
) -> List[Dict]:
    """
    Two-stage hybrid retrieval:
 
    Stage 1 - Candidate generation (parallel):
        • Dense:  embed query -> ChromaDB top-N
        • BM25:   tokenize query -> BM25 top-N
        • Fuse both lists with Reciprocal Rank Fusion (RRF)
 
    Stage 2 - Precision re-ranking:
        • Cross-encoder scores every (query, candidate) pair
        • Returns top rerank_top_n chunks
 
    Degrades gracefully: BM25-only if dense fails, dense-only if BM25
    index is missing, empty list if both fail.
    """
    if not query:
        logger.warning("Empty query provided.")
        return []
    # Dense retrieval
    dense_chunks: List[Dict] = []
    try:
        collection = client.get_or_create_collection(normalize_name(specification_name))
        query_embedding = embedder.encode([query]).tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
        )
        if results["documents"] and results["documents"][0]:
            dense_chunks = [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0] if results["metadatas"] else [])
            ]
            logger.info(f"Dense: {len(dense_chunks)} candidates.")
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
    # BM25 retrieval
    bm25_chunks: List[Dict] = []
    bm25, all_chunks = load_bm25(specification_name)
    if bm25 and all_chunks:
        try:
            scores = bm25.get_scores(tokenize(query))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_results]
            bm25_chunks = [all_chunks[i] for i in top_indices]
            logger.info(f"BM25: {len(bm25_chunks)} candidates.")
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
    # Fallback handling
    if not dense_chunks and not bm25_chunks:
        logger.warning("Both retrievers returned empty.")
        return []
    if not dense_chunks:
        logger.warning("Dense empty - BM25 only.")
        return rerank(query, bm25_chunks, top_n=rerank_top_n)
    if not bm25_chunks:
        logger.warning("BM25 empty - dense only.")
        return rerank(query, dense_chunks, top_n=rerank_top_n)
    # RRF + re-rank
    fused = rrf_fuse(dense_chunks, bm25_chunks)
    logger.info(f"RRF: {len(fused)} unique candidates after fusion.")
    return rerank(query, fused, top_n=rerank_top_n)