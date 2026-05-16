import json, os, httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from ingest import fetch_specification, extract_chunks
from vectorStore import client, store_chunks, retrieve_hybrid
from llm import build_prompt, query_llm
from confidence import score_confidence
from loggerConfig import get_logger

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

logger = get_logger("main")

app = FastAPI(title="DocQuery")

# Adding middleware to allow connections from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models 
class IngestRequest(BaseModel):
    specification_url: str

class QueryRequest(BaseModel):
    question: str
    specification_name: str

# Ingest Endpoint
@app.post("/ingest")
async def ingest(req: IngestRequest):
    logger.info(f"Ingest request received: {req.specification_url}")
    try:
        specification = fetch_specification(req.specification_url)
        if not specification:
            logger.error("Failed to fetch specification or empty response.")
            raise HTTPException(status_code=400, detail="Invalid or empty specification.")
        specification_name = specification.get("info", {}).get("title", "Unknown API")
        logger.debug(f"Specification name: {specification_name}")
        chunks = extract_chunks(specification)
        if not chunks:
            logger.warning("No chunks extracted from specification.")
        count = store_chunks(chunks, specification_name)
        logger.info(f"Ingested {count} chunks for {specification_name}.")
        return {
            "specification_name": specification_name,
            "chunks_stored": count,
            "message": f"Successfully ingested {count} endpoints."
        }
    except HTTPException:
        raise  # already handled
    except Exception as e:
        logger.error(f"Ingest failed: {e}.")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    logger.info(f"File ingest request received: {file.filename}")
    try:
        content = await file.read()
        specification = json.loads(content)
        if not specification:
            raise HTTPException(status_code=400, detail="Invalid or empty specification.")
        specification_name = specification.get("info", {}).get("title", "Unknown API")
        chunks = extract_chunks(specification)
        count = store_chunks(chunks, specification_name)
        logger.info(f"Ingested {count} chunks for {specification_name}.")
        return {
            "specification_name": specification_name,
            "chunks_stored": count,
            "message": f"Successfully ingested {count} endpoints."
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid or empty specification.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingest failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/query")
async def query(req: QueryRequest):
    """
    Query pipeline:
      1. Hybrid retrieval : BM25 + dense in parallel, fused with RRF.
      2. Cross-encoder re-ranking : top 5 from fused candidates.
      3. Build prompt from re-ranked chunks.
      4. Query LLM for final answer.
      5. Score confidence with HHEM + cosine hybrid.
    """
    logger.info(f"Query received for specification: {req.specification_name}")
    try:
        # Stage 1+2: hybrid retrieval + re-ranking
        chunks = retrieve_hybrid(
            query=req.question,
            specification_name=req.specification_name,
            num_results=20,
            rerank_top_n=5,
        )
        if not chunks:
            logger.warning("No chunks found for query.")
            raise HTTPException(
                status_code=404,
                detail="No specification found with that name. Ingest it first.",
            )
        # Stage 3: build prompt
        prompt = build_prompt(req.question, chunks)
        logger.debug("Prompt built successfully.")
        # Stage 4: LLM answer
        answer = await query_llm(prompt)
        if not answer:
            logger.error("LLM returned empty response.")
            raise HTTPException(status_code=500, detail="LLM failed to generate response.")
        # Stage 5: confidence scoring
        confidence = score_confidence(req.question, answer, chunks)
        logger.info(f"Query processed successfully (confidence={confidence}).")
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "endpoint": f"{c['metadata']['method']} {c['metadata']['path']}",
                    "summary": c['metadata']['summary']
                }
                for c in chunks
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# List Specifications 
@app.get("/specifications")
def list_specs():
    try:
        collections = client.list_collections()
        specification_names = [c.name for c in collections]
        logger.info(f"Listing {len(specification_names)} specifications.")
        return {"specifications": specification_names}
    except Exception as e:
        logger.error(f"Failed to list specifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to list specifications")