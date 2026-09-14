"""Async LLM client for meal-plan generation.

Two backends, chosen automatically:
  * **Hosted** — when a hosted key is configured (``settings.LLM_HOSTED``), calls
    an OpenAI-compatible ``/chat/completions`` endpoint (Groq by default) with
    JSON mode. This is what makes the *deployed* demo work for every visitor.
  * **Local**  — otherwise, calls a local Ollama ``/api/generate`` server.

Either way the result is a single parsed JSON object; any connectivity or parse
failure is surfaced as :class:`LLMUnavailableError` so callers fall back cleanly.
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


def _parse_json_object(raw: str, source: str) -> Dict[str, Any]:
    """Parse ``raw`` into a JSON object or raise a descriptive LLMUnavailableError."""
    if not raw:
        raise LLMUnavailableError(f"{source} returned an empty response.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(f"{source} did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMUnavailableError(f"{source} returned JSON that is not an object.")
    return parsed


async def generate_meal_plan(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """Generate a meal plan and return the parsed JSON object.

    Dispatches to the hosted provider when a key is configured, else to Ollama.

    Args:
        system_prompt: Instructions + context (user stats, remaining macros, foods).
        user_message: The user's natural-language request.

    Returns:
        The parsed JSON object produced by the model.

    Raises:
        LLMUnavailableError: If the provider is unreachable/timed out, returns a
            non-2xx status, or emits output that is not valid JSON.
    """
    if settings.LLM_HOSTED:
        return await _generate_hosted(system_prompt, user_message)
    return await _generate_ollama(system_prompt, user_message)


async def _generate_hosted(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """Generate via an OpenAI-compatible chat-completions API (Groq by default)."""
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        # JSON mode: providers return a single well-formed JSON object.
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.LLM_BASE_URL, json=payload, headers=headers
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200] if exc.response is not None else ""
        raise LLMUnavailableError(
            f"Hosted LLM request failed ({exc.response.status_code}): {detail}"
        ) from exc
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        raise LLMUnavailableError(f"Hosted LLM request failed: {exc}") from exc

    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailableError(
            f"Hosted LLM returned an unexpected payload shape: {exc}"
        ) from exc
    return _parse_json_object(raw, "Hosted LLM")


async def _generate_ollama(system_prompt: str, user_message: str) -> Dict[str, Any]:
    """Generate via a local Ollama ``/api/generate`` server."""
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

    return _parse_json_object(data.get("response", ""), "Ollama")
