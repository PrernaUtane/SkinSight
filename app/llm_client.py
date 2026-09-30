"""
OpenRouter client used for disease cards and follow-up chat.

The API key stays on the server. If the first free model is missing or
rate-limited, later models in the fallback list are tried automatically.
"""

from __future__ import annotations

import json
from typing import Any

import requests
from fastapi import HTTPException

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_HTTP_HEADERS,
    OPENROUTER_PRIMARY_MODEL,
    OPENROUTER_URL,
)


def _model_queue() -> list[str]:
    return [OPENROUTER_PRIMARY_MODEL] + OPENROUTER_FALLBACK_MODELS


def strip_code_fences(text: str) -> str:
    """Some models wrap JSON in markdown fences. Remove those before parsing."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[-1].strip() == "```":
        inner = lines[1:-1]
    else:
        inner = lines[1:]
    return "\n".join(inner).strip()


def complete_chat(messages: list[dict[str, str]], temperature: float = 0.3) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not set. Copy .env.example to .env and add a key.",
        )

    last_error = ""
    for model_id in _model_queue():
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
        }
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=OPENROUTER_HTTP_HEADERS,
                data=json.dumps(payload),
                timeout=60,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            print(f"[SkinSight] OpenRouter request failed ({model_id}): {exc}")
            continue

        if response.status_code == 200:
            body = response.json()
            content = body["choices"][0]["message"].get("content", "")
            print(f"[SkinSight] OpenRouter model used: {model_id}")
            return content

        if response.status_code in (404, 429):
            last_error = response.text[:200]
            print(f"[SkinSight] {model_id} returned {response.status_code}; trying next model")
            continue

        print(f"[SkinSight] OpenRouter error {response.status_code}: {response.text[:300]}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {response.status_code}: {response.text[:300]}",
        )

    raise HTTPException(
        status_code=502,
        detail=f"No OpenRouter model responded successfully. Last error: {last_error[:200]}",
    )


def disease_card_prompt(label: str, confidence_pct: float) -> str:
    return (
        f'You are a medical information assistant. A skin-lesion classifier predicted '
        f'"{label}" with {confidence_pct:.1f}% confidence.\n\n'
        f'Write a concise educational card for "{label}" as JSON only:\n\n'
        "{\n"
        '  "disease": "<proper name>",\n'
        '  "overview": "<2-3 sentences in plain language>",\n'
        '  "symptoms": ["<s1>", "<s2>", "<s3>", "<s4>"],\n'
        '  "causes": ["<c1>", "<c2>", "<c3>"],\n'
        '  "risk_factors": ["<r1>", "<r2>", "<r3>"],\n'
        '  "precautions": ["<p1>", "<p2>", "<p3>"],\n'
        '  "when_to_see_doctor": "<1-2 sentences>",\n'
        '  "severity": "Low | Medium | High",\n'
        '  "is_contagious": true | false\n'
        "}\n\n"
        "Return valid JSON only. This is educational text, not a diagnosis."
    )


def request_disease_card(label: str, confidence_pct: float) -> dict[str, Any]:
    raw = complete_chat(
        [{"role": "user", "content": disease_card_prompt(label, confidence_pct)}],
        temperature=0.3,
    )
    parsed = json.loads(strip_code_fences(raw))
    return parsed
