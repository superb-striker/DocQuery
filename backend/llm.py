import httpx, os
from typing import Dict, List
from dotenv import load_dotenv
from loggerConfig import get_logger

logger = get_logger("llm")

# Env 
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.critical("HF_TOKEN is missing in environment variables")
    raise ValueError("HF_TOKEN not found")
API_URL = "https://router.huggingface.co/v1/chat/completions"

# Prompt Builder 
def build_prompt(query: str, chunks: List[Dict]) -> str:
    if not query:
        logger.warning("Empty query passed to build_prompt.")
    if not chunks:
        logger.warning("No chunks provided to build_prompt.")
    try:
        context = "\n\n---\n\n".join([c["text"] for c in chunks])
    except KeyError as e:
        logger.error(f"Chunk missing 'text' field: {e}")
        context = ""
    prompt = f"""You are an API documentation assistant. Answer concisely and accurately.

            STRICT RULES:
            - Use ONLY the information in the SPEC SECTIONS below
            - If the answer is not in the spec, say "This information is not in the provided specification"
            - Do NOT repeat yourself or list unrelated endpoints
            - Reference exact HTTP methods, paths, and field names from the spec
            - If the question involves creating or sending data, always include a clean example request body in valid JSON
            - In the example, use realistic placeholder values (not the internal field list format)
            - Stop after answering the question fully

            SPEC SECTIONS:
            {context}

            QUESTION: {query}

            ANSWER:"""
    logger.debug("Prompt built successfully.")
    return prompt

# LLM Query
async def query_llm(prompt: str) -> str:
    if not prompt:
        logger.error("Empty prompt passed to as query.")
        return ""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            logger.info("Sending request to Hugging Face API")
            response = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={
                    "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.1,
                    "stop": [
                        "USER QUESTION:",    # stops hallucinated follow-ups
                        "QUESTION:",
                        "[INST]",
                        "\n\nNote:",         # stops unnecessary caveats
                        "SPEC SECTIONS:",   
                        "\n\nQUESTION",     
                    ]
                }
            )
            response.raise_for_status()
            try:
                data = response.json()
            except Exception:
                logger.error("Failed to parse JSON response.")
                return ""
            if "error" in data:
                logger.error(f"HF API error: {data['error']}")
                return ""
            try:
                answer = data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Unexpected response format: {e}.")
                return ""
            logger.info("Received response from LLM.")
            return answer
    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}.")
        return ""
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}.")
        return ""
    except Exception as e:
        logger.critical(f"Unexpected error in query_llm: {e}.")
        return ""