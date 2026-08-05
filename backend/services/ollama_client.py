"""Async client for a local Ollama server (meal-plan generation).

Uses Ollama's ``/api/generate`` endpoint with ``format: "json"`` so the model
returns a single JSON object. Any connectivity or parse failure is surfaced as
:class:`LLMUnavailableError` so the caller can fall back gracefully.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import httpx

from backend.config import settings


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be reached or returns unusable output."""


def _build_prompt(system_prompt: str, user_message: str) -> str:
    """Combine the system prompt and user message into a single prompt string."""
    return f"System: {system_prompt}\nUser: {user_message}\nAssistant:"


async def generate_meal_plan(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """Generate a meal plan via Ollama and return the parsed JSON object.

    Args:
        system_prompt: Instructions + context (user stats, remaining macros, foods).
        user_message: The user's natural-language request.

    Returns:
        The parsed JSON object produced by the model.

    Raises:
        LLMUnavailableError: If Ollama is unreachable/timed out, returns a non-2xx
            status, or emits output that is not valid JSON.
    """
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": _build_prompt(system_prompt, user_message),
        "stream": False,
        "format": "json",
        # Ollama reads generation params from the nested "options" object.
        "options": {
            "temperature": settings.OLLAMA_TEMPERATURE,
            "num_predict": settings.OLLAMA_MAX_TOKENS,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.post(settings.OLLAMA_BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.ConnectError as exc:
        raise LLMUnavailableError(
            f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Is `ollama serve` running?"
        ) from exc
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc

    raw = data.get("response", "")
    if not raw:
        raise LLMUnavailableError("Ollama returned an empty response.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(
            f"Ollama did not return valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMUnavailableError("Ollama returned JSON that is not an object.")
    return parsed
