"""
Augur — Multi-Specialist Ensemble Forecasting Engine

Fans a probabilistic question out to N domain specialists in parallel, aggregates
their independent probability estimates, and returns a calibrated ensemble forecast
with full per-specialist reasoning chains.

Each specialist:
  - Receives the question + their own domain playbook as system prompt
  - Returns structured JSON: {probability, confidence, reasoning, assumptions, uncertainties}
  - Runs in parallel (asyncio.gather) with an independent timeout per specialist
  - Falls back gracefully if a specialist errors or times out

Aggregation:
  - Weighted average probability (weight = confidence score per specialist)
  - Consensus label: strongly_yes / lean_yes / uncertain / lean_no / strongly_no
  - Synthesis: optional final LLM call to explain specialist agreement/disagreement
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SPECIALISTS_DIR = Path(__file__).parent / "specialists"

# Specialists available for forecasting — subset chosen for breadth of perspective.
DEFAULT_SPECIALISTS = [
    "reasoner",
    "intelligence_analyst",
    "market_analyst",
    "researcher",
    "data_scientist",
]

# Specialists that exist but aren't useful for general forecasting
FORECAST_EXCLUDED = {
    "coder", "coder_autonomous", "debugger", "ton_operator",
    "wave_coordinator", "roundabout",
}

# ---------------------------------------------------------------------------
# TOML loading + system prompt construction
# ---------------------------------------------------------------------------

_toml_cache: Dict[str, Dict[str, Any]] = {}


def get_specialists_dir() -> Path:
    """Return the path to specialist TOML manifests. Can be overridden via env."""
    custom = os.environ.get("AUGUR_SPECIALISTS_DIR")
    if custom:
        return Path(custom)
    return _SPECIALISTS_DIR


def load_toml(name: str) -> Optional[Dict[str, Any]]:
    """Load and cache a specialist TOML manifest."""
    if name in _toml_cache:
        return _toml_cache[name]
    path = get_specialists_dir() / f"{name}.toml"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        _toml_cache[name] = data
        return data
    except Exception as exc:
        logger.warning(f"[Augur] Failed to load {name}.toml: {exc}")
        return None


def build_system_prompt(name: str, config: Dict[str, Any]) -> str:
    """Build a forecasting system prompt from a specialist's TOML manifest."""
    playbook = config.get("playbook", {})
    phases = playbook.get("phases", [])
    principles = playbook.get("principles", [])

    phases_text = "\n".join(phases) if phases else "Think carefully and systematically."
    principles_text = "\n".join(f"- {p}" for p in principles) if principles else ""

    return f"""You are the {name.replace("_", " ").title()} specialist in an AI forecasting ensemble.

Your analytical framework:

REASONING PHASES:
{phases_text}

CORE PRINCIPLES:
{principles_text}

FORECASTING TASK:
You will receive a probabilistic question. Apply your full analytical framework and return
a structured JSON forecast. Be calibrated — do not round to 0.5 out of uncertainty;
commit to a specific estimate and explain why.

RESPONSE FORMAT (return ONLY this JSON, no markdown, no explanation outside it):
{{
  "probability": 0.XX,
  "confidence": 0.XX,
  "reasoning": "Step-by-step analysis leading to your estimate.",
  "key_assumptions": ["assumption 1", "assumption 2"],
  "key_uncertainties": ["uncertainty 1", "uncertainty 2"],
  "would_change_if": "What evidence would significantly shift your estimate."
}}

Probability: 0.0 = definitely will not happen, 1.0 = definitely will happen.
Confidence: 0.0 = pure guess, 1.0 = highly certain of your estimate."""


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def consensus_label(p: float) -> str:
    """Map ensemble probability to a human-readable consensus label."""
    if p >= 0.80:
        return "strongly_yes"
    elif p >= 0.60:
        return "lean_yes"
    elif p >= 0.40:
        return "uncertain"
    elif p >= 0.20:
        return "lean_no"
    else:
        return "strongly_no"


def weighted_average(estimates: list) -> tuple[float, float]:
    """Return (weighted_probability, mean_confidence) for successful estimates."""
    successful = [e for e in estimates if e.get("status") == "success" and e.get("confidence", 0) > 0]
    if not successful:
        return 0.5, 0.0
    total_weight = sum(e["confidence"] for e in successful)
    weighted_prob = sum(e["probability"] * e["confidence"] for e in successful) / total_weight
    mean_conf = total_weight / len(successful)
    return round(weighted_prob, 3), round(mean_conf, 3)


# ---------------------------------------------------------------------------
# Per-specialist LLM call
# ---------------------------------------------------------------------------

async def call_specialist(
    name: str,
    question: str,
    context: Optional[str],
    timeout_seconds: int,
    api_key: str,
) -> dict:
    """Run one specialist forecast via direct Anthropic API call."""
    import anthropic

    config = load_toml(name)
    if not config:
        return {
            "specialist": name, "probability": 0.5, "confidence": 0.0,
            "reasoning": "Specialist manifest not found.",
            "key_assumptions": [], "key_uncertainties": [],
            "would_change_if": None,
            "status": "api_error", "latency_ms": 0.0,
        }

    system_prompt = build_system_prompt(name, config)
    model_cfg = config.get("model", {})
    model = "claude-sonnet-4-6"

    user_message = f"QUESTION: {question}"
    if context:
        user_message += f"\n\nCONTEXT:\n{context}"

    t0 = time.time()
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=float(model_cfg.get("temperature", 0.4)),
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ),
            timeout=timeout_seconds,
        )
        latency_ms = (time.time() - t0) * 1000
        raw = response.content[0].text.strip()

        # Strip markdown fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        return {
            "specialist": name,
            "probability": float(parsed.get("probability", 0.5)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reasoning": str(parsed.get("reasoning", "")),
            "key_assumptions": list(parsed.get("key_assumptions", [])),
            "key_uncertainties": list(parsed.get("key_uncertainties", [])),
            "would_change_if": parsed.get("would_change_if"),
            "status": "success",
            "latency_ms": round(latency_ms, 1),
        }

    except asyncio.TimeoutError:
        latency_ms = (time.time() - t0) * 1000
        logger.warning(f"[Augur] {name} timed out after {timeout_seconds}s")
        return {
            "specialist": name, "probability": 0.5, "confidence": 0.0,
            "reasoning": f"Specialist timed out after {timeout_seconds}s.",
            "key_assumptions": [], "key_uncertainties": [],
            "would_change_if": None,
            "status": "timeout", "latency_ms": round(latency_ms, 1),
        }
    except json.JSONDecodeError as exc:
        latency_ms = (time.time() - t0) * 1000
        logger.warning(f"[Augur] {name} returned non-JSON: {exc}")
        return {
            "specialist": name, "probability": 0.5, "confidence": 0.0,
            "reasoning": "Specialist returned unparseable response.",
            "key_assumptions": [], "key_uncertainties": [],
            "would_change_if": None,
            "status": "parse_error", "latency_ms": round(latency_ms, 1),
        }
    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000
        logger.warning(f"[Augur] {name} error: {exc}")
        return {
            "specialist": name, "probability": 0.5, "confidence": 0.0,
            "reasoning": f"Specialist call failed: {type(exc).__name__}",
            "key_assumptions": [], "key_uncertainties": [],
            "would_change_if": None,
            "status": "api_error", "latency_ms": round(latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Synthesis pass
# ---------------------------------------------------------------------------

async def synthesize(
    question: str,
    estimates: list,
    ensemble_prob: float,
    api_key: str,
) -> Optional[str]:
    """One final LLM call to explain what the ensemble found and why."""
    import anthropic

    summaries = "\n".join(
        f"- {e['specialist']} ({e['probability']:.0%}, conf={e['confidence']:.0%}): {e['reasoning'][:300]}"
        for e in estimates if e["status"] == "success"
    )
    if not summaries:
        return None

    prompt = (
        f"QUESTION: {question}\n\n"
        f"ENSEMBLE RESULT: {ensemble_prob:.0%} probability\n\n"
        f"SPECIALIST ESTIMATES:\n{summaries}\n\n"
        "Write a concise 2-3 sentence synthesis explaining: "
        "(1) where specialists agreed and disagreed, "
        "(2) the key driver of the ensemble estimate, "
        "(3) the main residual uncertainty. "
        "Be specific — reference the actual reasoning above."
    )

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning(f"[Augur] Synthesis failed: {exc}")
        return None
