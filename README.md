# DocQuery 🔍

**Chat with any OpenAPI specification using RAG - no context window limits, no hallucinations, source-grounded answers.**

---

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19+-61DAFB?style=flat-square&logo=react&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Llama_3.1-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

---

## Why This Project Matters

The standard advice for understanding an API is "just paste the spec into ChatGPT." That breaks down fast.

| Problem | DocQuery's Solution |
|---|---|
| Stripe's spec is ~50,000 lines - won't fit in any context window | RAG retrieves only the relevant chunks per query |
| Internal/private APIs can't be pasted into public LLMs | Fully local vector store, nothing leaves your machine |
| LLM training data has a cutoff - new endpoints are unknown | Re-ingest any time, answers always reflect the current spec |
| Raw LLM answers can't cite where they got the answer | Every response includes the exact source endpoints |
| Can't query multiple APIs simultaneously | Multi-spec support with per-spec ChromaDB collections |

---

## Features

- **RAG pipeline** - OpenAPI spec is chunked, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB; each query retrieves the top-N relevant chunks before hitting the LLM.
- **`$ref` resolution** - request body schemas are fully dereferenced at ingest time, so field names like `name` are semantically searchable.
- **Confidence scoring** - cosine similarity between the LLM answer and retrieved chunks, returned as a 0–100 score
- **Source citations** - every answer includes the exact HTTP method + path that grounded it
- **Multi-spec support** - ingest multiple APIs simultaneously and switch between them with one click
- **Dual ingest modes** - paste a public URL or upload a local `.json` file (for private/internal specs)
- **Persistent vector store** - ChromaDB persists to disk; specs survive restarts without re-ingestion

---

## Architecture

![Architecture](architecture.svg)

---

## Design Decisions

**Summary-first chunking** - Each chunk starts with `{summary} - {METHOD} {path}` before the technical details. The sentence transformer sees the human-readable description first, which dramatically improves retrieval for natural language queries over path-only chunking.

**`$ref` resolution at ingest time** - OpenAPI specs use `$ref` pointers for all request/response schemas. Without resolving them, chunks contain `#/components/schemas/SecretCreate` instead of the actual field names (`notify_on_view`, `webhook_url`). Resolving at ingest time means field names are semantically searchable.

**Confidence scoring as a hallucination signal** - Cosine similarity between the LLM's answer embedding and the retrieved chunk embeddings gives a proxy for groundedness. A low score (< 50) means the answer diverged from the source material - useful for identifying when the model is making things up.

**Per-spec ChromaDB collections** - Each ingested API gets its own collection with a sanitized name. This allows simultaneous multi-spec support with zero cross-contamination and O(1) collection switching.

**Dual ingest modes** - URL ingest for public specs, file upload for private/internal APIs that can't be sent to external services. Both paths converge at the same `extract_chunks()` function.

**Llama 3.1 8B over Zephyr 7B** - Zephyr consistently ignored strict prompt rules and hallucinated parameters not in the spec. Llama 3.1 follows instruction constraints reliably enough for production-quality answers on API documentation tasks.

---

## Quick Start

### Prerequisites

- Python 3.11+
- React.js 18+
- A free [HuggingFace token](https://huggingface.co/settings/tokens)

### Backend

Clone the repository:

```bash
git clone https://github.com/superb-striker/DocQuery
cd DocQuery
```

Create a virtual environment and install required libraries:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```env
HF_TOKEN=your_token_here
```

Start the backend server:

```bash
uvicorn main:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

Install required packages and start the frontend server:

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI running at http://localhost:5173
```

---

## API Reference

### `POST /ingest`

Fetch and index an OpenAPI spec from a public URL.

**Request**

```json
{
  "specification_url": "https://petstore3.swagger.io/api/v3/openapi.json"
}
```

**Response**

```json
{
  "specification_name": "Swagger Petstore - OpenAPI 3.0",
  "chunks_stored": 19,
  "message": "Successfully ingested 19 endpoints."
}
```

---

### `POST /ingest/file`

Upload a local `.json` spec file (for private/internal APIs).

**Request**

```
file: <your-openapi-spec.json>
```

**Response** - same as `/ingest`

---

### `POST /query`

Ask a question against an ingested spec.

**Request**

```json
{
  "question": "How do I create a secret with a view notification?",
  "specification_name": "phantom_share"
}
```

**Response**

```json
{
  "answer": "Use POST /api/secrets with notify_on_view: true and provide a notify_email...",
  "confidence": 87,
  "sources": [
    {
      "endpoint": "POST /api/secrets",
      "summary": "Create Secret"
    }
  ]
}
```

---

### `GET /specifications`

List all currently ingested specification names.

**Response**

```json
{
  "specifications": ["swagger_petstore__openapi_30", "phantom_share"]
}
```

---

## Local Development

### Project Structure

```markdown
DocQuery/
└── backend/
|   ├── main.py     
|   ├── ingest.py             
|   ├── vectorStore.py 
|   ├── llm.py            
|   ├── confidence.py     
|   ├── loggerConfig.py       
|   ├── chromaDB/          
├── .env            
└── frontend/
    ├── DocQuery.jsx      
    ├── main.jsx          
    ├── src/
        ├── theme.js      
        ├── utils/
        │   └── classifyError.js
        ├── hooks/
        |   └── useTypewriter.js
        └── components/
            ├── ui/ 
            |   ├── ErrorBanner.jsx
            |   ├── Badge.jsx
            |   ├── NoiseFilter.jsx
            |   ├── Spinner.jsx
            |   ├── ThemeToggle.jsx
            ├── AppHeader.jsx
            ├── ConfidenceRing.jsx
            ├── SourceChip.jsx
            ├── IngestPanel.jsx
            ├── QueryPanel.jsx
            └── ResultPanel.jsx
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace access token for the inference router |

### Changing the LLM

The model is set in `llm.py`. Any model available on the [HuggingFace router](https://huggingface.co/docs/inference-providers) works as a drop-in replacement.

### Resetting the Vector Store

```bash
# Wipe all ingested specs and start fresh
rm -rf chromaDB/                       # Linux/Mac
Remove-Item -Recurse -Force chromaDB   # Windows PowerShell
```

---

## Limitations

- **HuggingFace free tier** - rate limited; responses may take 5–15 seconds under load. For production use, swap to Groq (free, ~1s responses) or a paid inference provider.
- **Schema depth** - `$ref` resolution is one level deep. Nested `$ref` chains (schemas referencing other schemas) are not fully resolved.
- **YAML specs** - only JSON OpenAPI specs are supported. For YAML specs, convert first: `python -c "import yaml,json,sys; json.dump(yaml.safe_load(open('spec.yaml')), open('spec.json','w'))"`.

---
