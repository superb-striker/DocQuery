import chromadb, os
from dotenv import load_dotenv
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from loggerConfig import get_logger

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
logger = get_logger("vector_store")

# Embedding model & ChromaDB 
try:
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load embedding model: {e}")
    raise
try:
    logger.info("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path="./chromaDB")
    logger.info("ChromaDB client initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize ChromaDB: {e}")
    raise

# Store 
def store_chunks(chunks: List[Dict], specification_name: str) -> int:
    if not chunks:
        logger.warning("No chunks provided for storage.")
        return 0
    try:
        name = specification_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        name = name.strip('_')[:50]
        # Ensure minimum 3 characters
        if len(name) < 3:
            name = name + "_api"
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
        return len(chunks)
    except Exception as e:
        logger.error(f"Failed to store chunks: {e}")
        return 0

# Retrieve 
def retrieve(query: str, specification_name: str, num_results: int = 3) -> List[Dict]:
    if not query:
        logger.warning("Empty query provided.")
        return []
    try:
        name = specification_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        name = name.strip('_')[:50]
        # Ensure minimum 3 characters
        if len(name) < 3:
            name = name + "_api"
        collection = client.get_or_create_collection(name)
    except Exception as e:
        logger.error(f"Failed to access collection: {e}")
        return []
    try:
        query_embedding = embedder.encode([query]).tolist()[0]
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        return []
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results
        )
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return []
    if not results["documents"] or not results["documents"][0]:
        logger.info("No documents found.")
        return []
    if not results["metadatas"] or not results["metadatas"][0]:
        logger.info("No metadata found.")
        return []
    logger.debug(f"Retrieved {len(results['documents'][0])} results.")
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    ]