"""
Thin wrapper around the Groq API for structured (JSON) agent outputs.force JSON mode, validate
against a Pydantic schema, retry on malformed output 
"""

import json
import os
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()  
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


def get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key, "
                "or `export GROQ_API_KEY=...` before running."
            )
        _client = Groq(api_key=api_key)
    return _client


def structured_completion(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 2,
    temperature: float = 0.2,
) -> T:
    client = get_client()
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system_prompt}\n\n"
        "Respond with a single valid JSON object matching this JSON Schema and "
        f"nothing else — no markdown fences, no preamble:\n{schema_hint}"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            data = json.loads(response.choices[0].message.content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError, Exception) as e:  # noqa: BLE001
            last_error = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"LLM call failed after {max_retries + 1} attempts: {last_error}"
            ) from last_error
