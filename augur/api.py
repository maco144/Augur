"""
Augur — Multi-Specialist Ensemble Forecasting API

Endpoints:
    POST /v1/forecast                  — Run an ensemble forecast across specialists
    GET  /v1/forecast/specialists      — List available specialist forecasters
    GET  /v1/forecast/history          — Recent forecast history
"""
from __future__ import annotations

import asyncio
import os
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .engine import (
    DEFAULT_SPECIALISTS,
    FORECAST_EXCLUDED,
    call_specialist,
    consensus_label,
    get_specialists_dir,
    load_toml,
    synthesize,
    weighted_average,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/forecast", tags=["Augur"])

# In-memory forecast history (persists per process lifetime)
_forecast_history: list[dict] = []
_MAX_HISTORY = 200


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    question: str = Field(
        description="A binary probabilistic question. E.g. 'Will the Fed cut rates before September 2026?'"
    )
    context: Optional[str] = Field(
        None,
        description="Optional background context to help specialists. Keep concise.",
    )
    specialists: Optional[List[str]] = Field(
        None,
        description=f"Specialist names to query. Defaults to: {DEFAULT_SPECIALISTS}",
    )
    synthesize: bool = Field(
        True,
        description="Run a synthesis pass to explain specialist agreement/disagreement.",
    )
    timeout_seconds: int = Field(
        60,
        ge=10,
        le=180,
        description="Per-specialist timeout in seconds.",
    )


class SpecialistEstimate(BaseModel):
    specialist: str
    probability: float
    confidence: float
    reasoning: str
    key_assumptions: List[str]
    key_uncertainties: List[str]
    would_change_if: Optional[str] = None
    status: str  # "success" | "timeout" | "parse_error" | "api_error"
    latency_ms: float


class ForecastResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    question: str
    ensemble_probability: float = Field(description="Confidence-weighted average across specialists")
    ensemble_confidence: float = Field(description="Average confidence of contributing specialists")
    consensus: str = Field(
        description="strongly_yes | lean_yes | uncertain | lean_no | strongly_no"
    )
    synthesis: Optional[str] = Field(
        None,
        description="LLM-generated explanation of specialist agreement/disagreement.",
    )
    specialists: List[SpecialistEstimate]
    successful: int
    failed: int
    latency_ms: float
    model_used: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/specialists", summary="List available specialist forecasters")
async def list_forecast_specialists():
    """List all specialists available for forecasting, with their TOML-derived metadata."""
    specialists_dir = get_specialists_dir()
    available = []
    for toml_path in sorted(specialists_dir.glob("*.toml")):
        name = toml_path.stem
        if name in FORECAST_EXCLUDED:
            continue
        config = load_toml(name)
        if not config:
            continue
        available.append({
            "name": name,
            "model": config.get("model", {}).get("primary", "unknown"),
            "temperature": config.get("model", {}).get("temperature", 0.4),
            "default": name in DEFAULT_SPECIALISTS,
        })
    return {"specialists": available, "defaults": DEFAULT_SPECIALISTS}


@router.get("/history", summary="Recent forecast history")
async def forecast_history(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Return the most recent forecast results (newest first)."""
    return {"forecasts": _forecast_history[:limit], "total": len(_forecast_history)}


@router.post("", summary="Run a multi-specialist ensemble forecast")
async def run_forecast(body: ForecastRequest) -> ForecastResponse:
    """
    Fan a probabilistic question out to N specialists in parallel.

    Each specialist applies its domain playbook (loaded from TOML at runtime),
    independently estimates the probability, and explains its reasoning.
    Responses are aggregated via confidence-weighted average.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    specialist_names = body.specialists or DEFAULT_SPECIALISTS
    specialists_dir = get_specialists_dir()

    # Validate specialist names
    unknown = [n for n in specialist_names if not (specialists_dir / f"{n}.toml").exists()]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown specialists: {unknown}")

    t0 = time.time()

    # Fan out to all specialists in parallel
    tasks = [
        call_specialist(name, body.question, body.context, body.timeout_seconds, api_key)
        for name in specialist_names
    ]
    raw_estimates: list[dict] = await asyncio.gather(*tasks)

    ensemble_prob, ensemble_conf = weighted_average(raw_estimates)
    successful = sum(1 for e in raw_estimates if e["status"] == "success")
    failed = len(raw_estimates) - successful

    # Optional synthesis pass
    synthesis_text = None
    if body.synthesize and successful > 0:
        synthesis_text = await synthesize(body.question, raw_estimates, ensemble_prob, api_key)

    total_latency_ms = round((time.time() - t0) * 1000, 1)

    logger.info(
        f"[Augur] question={body.question[:60]!r} "
        f"ensemble={ensemble_prob:.0%} conf={ensemble_conf:.0%} "
        f"specialists={successful}/{len(raw_estimates)} latency={total_latency_ms}ms"
    )

    # Save to history
    _forecast_history.insert(0, {
        "question": body.question,
        "ensemble_probability": ensemble_prob,
        "ensemble_confidence": ensemble_conf,
        "consensus": consensus_label(ensemble_prob),
        "successful": successful,
        "failed": failed,
        "latency_ms": total_latency_ms,
        "specialist_count": len(raw_estimates),
        "timestamp": time.time(),
    })
    if len(_forecast_history) > _MAX_HISTORY:
        _forecast_history[:] = _forecast_history[:_MAX_HISTORY]

    # Convert raw dicts to Pydantic models
    estimates = [SpecialistEstimate(**e) for e in raw_estimates]

    return ForecastResponse(
        question=body.question,
        ensemble_probability=ensemble_prob,
        ensemble_confidence=ensemble_conf,
        consensus=consensus_label(ensemble_prob),
        synthesis=synthesis_text,
        specialists=estimates,
        successful=successful,
        failed=failed,
        latency_ms=total_latency_ms,
        model_used="claude-sonnet-4-6",
    )
