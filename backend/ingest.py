import httpx
import json
from typing import Dict, List
from loggerConfig import get_logger

logger = get_logger("ingest")

def fetch_specification(url: str) -> Dict:
    try:
        logger.info(f"Fetching specification from {url}")
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()   # Raises error if: 4xx (client error) or 5xx (server error)
        try:
            data = response.json()
            logger.info("Successfully fetched and parsed JSON.")
            return data
        except json.JSONDecodeError:
            logger.error("Response is not valid JSON.")
            return {}
    except httpx.RequestError as e:
        logger.error(f"Request failed: {e}.")
        return {}
    except httpx.HTTPStatusError as e:
        logger.error(f"Bad HTTP status: {e.response.status_code}.")
        return {}

def resolve_ref(ref: str, specification: dict) -> dict:
    """Follow a $ref pointer like #/components/schemas/SecretCreate"""
    parts = ref.lstrip("#/").split("/")
    result = specification
    for part in parts:
        result = result.get(part, {})
    return result

def extract_chunks(specification: dict) -> list[dict]:
    chunks = []
    paths = specification.get("paths", {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, details in methods.items():
            if method.lower() not in ["get", "post", "put", "patch", "delete"]:
                continue
            if not isinstance(details, dict):
                continue
            try:
                # Resolve requestBody $ref
                request_body = details.get("requestBody", {})
                resolved_body = {}
                try:
                    ref = request_body.get("content", {}).get("application/json", {}).get("schema", {}).get("$ref")
                    if ref:
                        schema = resolve_ref(ref, specification)
                        resolved_body = {
                            "required_fields": schema.get("required", []),
                            "all_fields": {
                                k: {
                                    "type": v.get("type", "string"),
                                    "default": v.get("default", None),
                                }
                                for k, v in schema.get("properties", {}).items()
                            }
                        }
                except Exception:
                    resolved_body = request_body
                text = f"""
                        {details.get('summary', 'No summary')} - {method.upper()} {path}
                        Operation: {method.upper()} {path}
                        Summary: {details.get('summary', 'No summary')}
                        Description: {details.get('description', 'No description')}
                        Parameters: {json.dumps(details.get('parameters', []), indent=2)}
                        Request Body Fields: {json.dumps(resolved_body, indent=2)}
                        Responses: {json.dumps(details.get('responses', {}), indent=2)}
                        Tags: {', '.join(details.get('tags', []))}
                        """.strip()
                chunks.append({
                    "text": text,
                    "metadata": {
                        "path": path,
                        "method": method.upper(),
                        "summary": details.get("summary", ""),
                        "specification_name": specification.get("info", {}).get("title", "Unknown API")
                    }
                })
                logger.debug(f"Processed {method.upper()} {path}.")
            except (TypeError, ValueError) as e:
                logger.error(f"Failed processing {method.upper()} {path}: {e}.")
                continue
    logger.info(f"Extracted {len(chunks)} chunks.")
    return chunks