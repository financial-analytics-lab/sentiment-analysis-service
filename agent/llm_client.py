"""
Thin async LLM client over the Anthropic Messages API.
Uses requests + asyncio.to_thread so no new dependencies are required.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import requests

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    LLM_TIMEOUT,
)

_BASE = ANTHROPIC_BASE_URL.rstrip("/")
_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _post_sync(
    model: str,
    messages: list[dict],
    system: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        payload["system"] = system

    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(
                f"{_BASE}/v1/messages",
                headers=_HEADERS,
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code in _RETRYABLE_STATUS:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
            resp.raise_for_status()
            data: dict = resp.json()
            if data.get("type") == "error":
                raise RuntimeError(f"API error: {data}")
            content = data.get("content", [])
            text_block = next((b for b in content if b.get("type") == "text"), None)
            if text_block:
                return text_block["text"]
            raise RuntimeError(f"Unexpected response shape: {data}")
        except Exception as exc:
            last_err = exc
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(LLM_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} retries") from last_err


async def complete(
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Async wrapper: runs the blocking HTTP call in a thread pool."""
    return await asyncio.to_thread(
        _post_sync, model, messages, system, max_tokens, temperature
    )


async def complete_json(
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    parse_retries: int = 2,
) -> dict:
    """
    Like `complete`, but parses the result as JSON and retries on parse failure.

    The API gateway forces extended thinking, whose trace counts toward
    max_tokens. When the trace is long the JSON answer can be truncated before
    a single '{' is emitted. On a parse failure we retry with a 1.6x larger
    token budget (capped) — a fresh, often shorter, thinking trace plus more
    room for the answer usually succeeds.
    """
    last_err: Exception | None = None
    budget = max_tokens
    for attempt in range(parse_retries + 1):
        raw = await complete(
            model=model,
            messages=messages,
            system=system,
            max_tokens=budget,
            temperature=temperature,
        )
        try:
            return _parse_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            budget = min(int(budget * 1.6), 16000)
    raise ValueError(
        f"JSON parse failed after {parse_retries + 1} attempts "
        f"(last budget={budget}): {last_err}"
    )


def _parse_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output, stripping markdown fences if present.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first complete JSON object
        depth = 0
        start_idx = text.find("{")
        if start_idx == -1:
            raise ValueError(f"No JSON object found in LLM output: {text[:200]}")
        for i, ch in enumerate(text[start_idx:], start_idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start_idx : i + 1])
        raise ValueError(f"Could not parse JSON from LLM output: {text[:200]}")
