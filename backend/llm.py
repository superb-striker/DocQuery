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
    prompt = f"""You are an API documentation assistant. Answer using ONLY the SPEC SECTIONS below.
    
            OUTPUT FORMAT - follow this exactly:
            **Endpoint:** [HTTP METHOD] [path]
            **Summary:** [one sentence from the spec]
            **Parameters:** [list only parameters explicitly named in the spec, or "None listed in spec"]
            **Request Body:** [list only fields explicitly named in the spec, or "Not specified in spec"]
            **Notes:** [any other relevant details from the spec only, do not reference other things in this spec.]

            If the spec sections do not contain enough information to fill a field, write "Not specified in spec" for that field.
            Do not add information from outside the spec. 

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
                        "However,",
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
                if "error" in data:
                    logger.error(f"HF API error: {data['error']}")
                    return ""
                # also guard against missing choices
                if not data.get("choices"):
                    logger.error(f"Unexpected response format, no choices: {data}")
                    return ""
            except Exception:
                logger.error("Failed to parse JSON response.")
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