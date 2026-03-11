"""LLM-as-a-judge integration via the OpenRouter chat completions API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


_OPENROUTER_DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_DEFAULT_MODEL = "google/gemini-3.1-pro-preview"


def _build_judge_messages(document: str, summary: str, reference: str) -> list[dict[str, str]]:
    system_prompt = (
        "You are a strict summarization judge. Score the candidate summary against the source "
        "document and reference summary. Return JSON only with integer scores from 1 to 5."
    )
    user_prompt = f"""
Evaluate the candidate summary.

Scoring rubric:
- faithfulness_1_5: factual consistency with the source document
- coverage_1_5: how well the candidate covers the key ideas from the reference summary
- coherence_1_5: fluency, organization, and readability
- conciseness_1_5: whether the summary is appropriately compressed without obvious bloat
- overall_1_5: overall summary quality considering the dimensions above

Return strict JSON with exactly these keys:
{{
  "faithfulness_1_5": 1,
  "coverage_1_5": 1,
  "coherence_1_5": 1,
  "conciseness_1_5": 1,
  "overall_1_5": 1,
  "rationale": "short explanation"
}}

Source document:
{document}

Reference summary:
{reference}

Candidate summary:
{summary}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    raise ValueError("Unsupported OpenRouter message content format.")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Judge response did not contain a JSON object.")
    return json.loads(stripped[start : end + 1])


def _normalize_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    required_scores = [
        "faithfulness_1_5",
        "coverage_1_5",
        "coherence_1_5",
        "conciseness_1_5",
        "overall_1_5",
    ]
    normalized: dict[str, Any] = {}
    for key in required_scores:
        value = int(result[key])
        if not 1 <= value <= 5:
            raise ValueError(f"Judge score {key} must be between 1 and 5.")
        normalized[key] = value
    normalized["rationale"] = str(result.get("rationale", "")).strip()
    return normalized


def judge_summary_openrouter(
    document: str,
    summary: str,
    reference: str,
    model: str = _OPENROUTER_DEFAULT_MODEL,
    api_base: str = _OPENROUTER_DEFAULT_API_BASE,
    timeout_seconds: int = 120,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to use OpenRouter LLM judge evaluation.")

    payload = {
        "model": model,
        "messages": _build_judge_messages(document, summary, reference),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    x_title = os.environ.get("OPENROUTER_X_TITLE", "").strip()
    if x_title:
        headers["X-Title"] = x_title

    req = request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = json.loads(response.read().decode("utf-8"))

    choices = raw.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter judge response did not include choices.")
    message = choices[0].get("message", {})
    content = _extract_text_content(message.get("content", ""))
    return _normalize_judge_result(_extract_json_object(content))
