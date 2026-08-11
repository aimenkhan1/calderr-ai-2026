"""
Groq API client wrapper — supports both structured (JSON-schema-validated)
completions and simple tool-calling
"""

import json
import os
import time
from typing import Type, TypeVar, Callable, Optional

from pydantic import BaseModel, ValidationError
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()  

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


@traceable
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
    temperature: float = 0.3,
) -> T:
    """Force JSON output validated against `schema`. Retries on malformed output."""
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



FETCH_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_full_document",
        "description": (
            "Fetch the FULL text of a seed document by its source filename, when "
            "the retrieved excerpt isn't enough to be confident in your finding. "
            "Only call this if you genuinely need more context than the excerpt gives you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_filename": {
                    "type": "string",
                    "description": "The exact filename of the source document, e.g. 'ai_alignment_basics.txt'",
                }
            },
            "required": ["source_filename"],
        },
    },
}


def maybe_call_tool(
    system_prompt: str,
    user_prompt: str,
    tool_executor: Callable[[str], Optional[str]],
    temperature: float = 0.2,
) -> str:

    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=[FETCH_DOCUMENT_TOOL],
        tool_choice="auto",
        temperature=temperature,
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return ""

    extra_context_parts = []
    for call in tool_calls:
        try:
            args = json.loads(call.function.arguments)
            filename = args.get("source_filename", "")
            result = tool_executor(filename)
            if result:
                extra_context_parts.append(f"[Full document '{filename}']:\n{result}")
        except Exception:  # noqa: BLE001 — tool failures should not break the pipeline
            continue

    return "\n\n".join(extra_context_parts)
