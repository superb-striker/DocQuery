import json, os, httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse import observe, get_client
from ingest import fetch_specification, extract_chunks
from vectorStore import client, store_chunks, retrieve_hybrid
from llm import build_prompt, query_llm
from confidence import score_confidence
from loggerConfig import get_logger

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
langfuse = get_client()

logger = get_logger("main")

app = FastAPI(title="DocQuery")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    specification_url: str

class QueryRequest(BaseModel):
    question: str
    specification_name: str


# Observed helpers
@observe(name="hybrid-retrieval")
def run_retrieval(question: str, specification_name: str):
    chunks = retrieve_hybrid(
        query=question,
        specification_name=specification_name,
        num_results=20,
        rerank_top_n=5,
    )
    # Log what was retrieved so it's visible in the span
    langfuse.update_current_span(
        input={"question": question, "specification": specification_name},
        output={
            "chunks_returned": len(chunks) if chunks else 0,
            "endpoints": [
                f"{c['metadata']['method']} {c['metadata']['path']}"
                for c in (chunks or [])
            ]
        }
    )
    return chunks

@observe(name="llm-generate")
async def run_generation(prompt: str):
    # Log the full prompt so you can debug what the LLM actually received
    langfuse.update_current_span(input={"prompt": prompt})
    answer = await query_llm(prompt)
    langfuse.update_current_span(output={"answer": answer})
    return answer

# Endpoints

@app.post("/ingest")
@observe(name="ingest")
async def ingest(req: IngestRequest):
    logger.info(f"Ingest request received: {req.specification_url}")
    langfuse.set_current_trace_io(
        input={"url": req.specification_url}
    )
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
        result = {
            "specification_name": specification_name,
            "chunks_stored": count,
            "message": f"Successfully ingested {count} endpoints."
        }
        langfuse.set_current_trace_io(output={
            "specification_name": specification_name,
            "chunks_stored": count
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed: {e}.")
        langfuse.update_current_span(output={"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/ingest/file")
@observe(name="ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    logger.info(f"File ingest request received: {file.filename}")
    langfuse.set_current_trace_io(input={"filename": file.filename})
    try:
        content = await file.read()
        specification = json.loads(content)
        if not specification:
            raise HTTPException(status_code=400, detail="Invalid or empty specification.")
        specification_name = specification.get("info", {}).get("title", "Unknown API")
        chunks = extract_chunks(specification)
        count = store_chunks(chunks, specification_name)
        logger.info(f"Ingested {count} chunks for {specification_name}.")
        result = {
            "specification_name": specification_name,
            "chunks_stored": count,
            "message": f"Successfully ingested {count} endpoints."
        }
        langfuse.set_current_trace_io(output={
            "specification_name": specification_name,
            "chunks_stored": count
        })
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid or empty specification.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingest failed: {e}")
        langfuse.update_current_span(output={"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/query")
@observe(name="rag-query")
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
    # Log the user's question as the trace input — visible at the top level in Langfuse
    langfuse.set_current_trace_io(
        input={"question": req.question, "specification": req.specification_name}
    )
    try:
        # Stage 1+2: hybrid retrieval + re-ranking
        chunks = run_retrieval(req.question, req.specification_name)
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
        answer = await run_generation(prompt)
        if not answer:
            logger.error("LLM returned empty response.")
            raise HTTPException(status_code=500, detail="LLM failed to generate response.")
        # Stage 5: confidence scoring
        confidence = score_confidence(req.question, answer, chunks)
        logger.info(f"Query processed successfully (confidence={confidence}).")
        sources = [
            {
                "endpoint": f"{c['metadata']['method']} {c['metadata']['path']}",
                "summary": c['metadata']['summary']
            }
            for c in chunks
        ]
        # Log the answer + confidence as the trace output
        langfuse.set_current_trace_io(
            output={"answer": answer, "confidence": confidence, "sources": sources}
        )
        # Confidence as a 0-1 metric — shows up in Langfuse's scores tab and dashboard charts
        langfuse.score_current_trace(name="confidence", value=confidence / 100)
        return {"answer": answer, "confidence": confidence, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        langfuse.update_current_span(output={"error": str(e)})
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

# Transcribe endpoint — not traced, it's infrastructure not RAG pipeline
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logger.info(f"Transcribe request received: {file.filename}")
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("recording.webm", audio_bytes, file.content_type or "audio/webm")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "language": "en",
                    "response_format": "json",
                    "temperature": "0.0",
                },
            )
            response.raise_for_status()
            data = response.json()
        transcript = data.get("text", "").strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio.")
        logger.info(f"Transcribed: '{transcript[:80]}'")
        return {"transcript": transcript}
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq API error: {e.response.status_code} — {e.response.text}")
        raise HTTPException(status_code=502, detail="Transcription service error.")
    except Exception as e:
        logger.error(f"Transcribe failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")