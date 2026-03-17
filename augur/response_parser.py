"""
Augur — Specialist Response Parser

Extracts structured forecast JSON from heterogeneous LLM outputs.
Different models wrap JSON differently (markdown fences, XML tags, raw text,
or with surrounding explanation). This module normalizes all of them.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default fallback for unparseable responses
# ---------------------------------------------------------------------------

_PARSE_ERROR: Dict[str, Any] = {
    "status": "parse_error",
    "probability": 0.5,
    "confidence": 0.0,
    "reasoning": "Unparseable response.",
    "key_assumptions": [],
    "key_uncertainties": [],
    "would_change_if": None,
}


# ---------------------------------------------------------------------------
# Extraction strategies (tried in order)
# ---------------------------------------------------------------------------

def _try_markdown_fenced(raw: str) -> Optional[dict]:
    """Extract JSON from markdown code fences: ```json ... ``` or ``` ... ```."""
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _try_xml_tagged(raw: str) -> Optional[dict]:
    """Extract JSON from XML-style tags: <json>...</json> or <response>...</response>."""
    for tag in ("json", "response"):
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _try_raw_json(raw: str) -> Optional[dict]:
    """Try parsing the entire raw string as JSON directly."""
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_regex_brace(raw: str) -> Optional[dict]:
    """Find the first {...} block that contains 'probability'."""
    # Walk through all top-level brace pairs and try each
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw[start : i + 1]
                if "probability" in candidate:
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        pass
                start = None
    return None


# ---------------------------------------------------------------------------
# Validation + coercion
# ---------------------------------------------------------------------------

def _coerce_float(value: Any, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Coerce a value to float, clamping to [lo, hi]."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def _validate(parsed: dict) -> dict:
    """Validate and coerce parsed JSON into the canonical forecast schema."""
    return {
        "status": "success",
        "probability": _coerce_float(parsed.get("probability"), default=0.5),
        "confidence": _coerce_float(parsed.get("confidence"), default=0.5),
        "reasoning": str(parsed.get("reasoning", "") or ""),
        "key_assumptions": list(parsed.get("key_assumptions") or []),
        "key_uncertainties": list(parsed.get("key_uncertainties") or []),
        "would_change_if": parsed.get("would_change_if"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_specialist_response(raw: str) -> dict:
    """
    Extract a structured forecast dict from raw LLM output.

    Tries multiple extraction strategies in order of specificity, then validates
    and coerces all fields. Returns a dict with ``status`` set to ``"success"``
    on successful extraction or ``"parse_error"`` if every strategy fails.
    """
    if not raw or not raw.strip():
        return dict(_PARSE_ERROR)

    # Try each extraction strategy in order
    for strategy in (_try_markdown_fenced, _try_xml_tagged, _try_raw_json, _try_regex_brace):
        result = strategy(raw)
        if result is not None and isinstance(result, dict):
            return _validate(result)

    return dict(_PARSE_ERROR)
