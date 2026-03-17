"""
Augur — Multi-Specialist Ensemble Forecasting API

Endpoints:
    POST /v1/forecast                  — Run an ensemble forecast across specialists
    GET  /v1/forecast/specialists      — List available specialist forecasters
    GET  /v1/forecast/history          — Recent forecast history
    POST /v1/forecast/resolve          — Mark a forecast as resolved with actual outcome
    GET  /v1/forecast/calibration      — Calibration curve data with confidence intervals
"""
from __future__ import annotations

import asyncio
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
from .router import available_providers
from .base_rates import get_base_rates, list_categories, search_base_rates
from .calibration import calibration_report, resolve_forecast
from .divergence import detect_divergence, notify_divergence

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
    model: str = ""
    probability: float
    confidence: float
    reasoning: str
    key_assumptions: List[str]
    key_uncertainties: List[str]
    would_change_if: Optional[str] = None
    status: str  # "success" | "timeout" | "parse_error" | "api_error"
    latency_ms: float


class ResolveRequest(BaseModel):
    question: str = Field(description="The question text to match against forecast history.")
    actual_outcome: bool = Field(description="True if the forecasted event occurred, False otherwise.")
    ensemble_probability: Optional[float] = Field(
        None,
        description="Override: provide the ensemble probability directly instead of matching history.",
    )


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
    divergence: Optional[dict] = Field(
        None,
        description="Divergence info when 3+ specialists disagree by >20pp. None if no significant divergence.",
    )
    specialists: List[SpecialistEstimate]
    successful: int
    failed: int
    latency_ms: float
    models_used: List[str] = Field(description="Unique model strings used across specialists")


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
    avail = available_providers()
    if not any(avail.values()):
        raise HTTPException(status_code=503, detail="No LLM provider API keys configured")

    specialist_names = body.specialists or DEFAULT_SPECIALISTS
    specialists_dir = get_specialists_dir()

    # Validate specialist names
    unknown = [n for n in specialist_names if not (specialists_dir / f"{n}.toml").exists()]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown specialists: {unknown}")

    t0 = time.time()

    # Fan out to all specialists in parallel
    tasks = [
        call_specialist(name, body.question, body.context, body.timeout_seconds)
        for name in specialist_names
    ]
    raw_estimates: list[dict] = await asyncio.gather(*tasks)

    ensemble_prob, ensemble_conf = weighted_average(raw_estimates)
    successful = sum(1 for e in raw_estimates if e["status"] == "success")
    failed = len(raw_estimates) - successful

    # Optional synthesis pass
    synthesis_text = None
    if body.synthesize and successful > 0:
        synthesis_text = await synthesize(body.question, raw_estimates, ensemble_prob)

    # Divergence detection
    divergence_info = detect_divergence(raw_estimates)
    if divergence_info:
        logger.warning(
            f"[Augur] Divergence detected: spread={divergence_info['spread_pp']}pp "
            f"divergent={divergence_info['num_divergent']}/{divergence_info['num_successful']}"
        )
        # Fire-and-forget webhook notifications
        asyncio.create_task(notify_divergence(body.question, divergence_info))

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
        "divergence_flags": divergence_info,
        "timestamp": time.time(),
    })
    if len(_forecast_history) > _MAX_HISTORY:
        _forecast_history[:] = _forecast_history[:_MAX_HISTORY]

    # Convert raw dicts to Pydantic models
    estimates = [SpecialistEstimate(**e) for e in raw_estimates]

    models_used = sorted(set(e["model"] for e in raw_estimates if e.get("model")))

    return ForecastResponse(
        question=body.question,
        ensemble_probability=ensemble_prob,
        ensemble_confidence=ensemble_conf,
        consensus=consensus_label(ensemble_prob),
        synthesis=synthesis_text,
        divergence=divergence_info,
        specialists=estimates,
        successful=successful,
        failed=failed,
        latency_ms=total_latency_ms,
        models_used=models_used,
    )


@router.post("/resolve", summary="Mark a forecast as resolved with actual outcome")
async def resolve(body: ResolveRequest) -> dict:
    """Record that a previously-forecasted question resolved to a known outcome."""
    try:
        record = resolve_forecast(
            question=body.question,
            actual_outcome=body.actual_outcome,
            ensemble_probability=body.ensemble_probability,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "resolved", "record": record}


@router.get("/calibration", summary="Calibration curve with confidence intervals")
async def get_calibration(quarter: Optional[str] = Query(None, description="Filter to a quarter, e.g. '2026-Q1'")) -> dict:
    """
    Return calibration dashboard data: bucketed confidence-vs-resolution rates
    with Wilson score confidence intervals, Brier score, grouped by quarter.
    """
    return calibration_report(quarter=quarter)


@router.get("/base-rates", summary="Reference class base-rate library")
async def get_base_rate_library(
    category: Optional[str] = Query(None, description="Filter by category (e.g. 'markets', 'ai_ml', 'healthcare')"),
    q: Optional[str] = Query(None, description="Keyword search across all base-rate entries"),
) -> dict:
    """
    Return structured base-rate anchoring templates for reference class forecasting.

    Use these historical frequencies to anchor probability estimates before adjusting
    for question-specific factors.  Supports filtering by category or free-text search.
    """
    if q:
        entries = search_base_rates(q)
    elif category:
        entries = get_base_rates(category)
    else:
        from .base_rates import BASE_RATE_REGISTRY
        entries = list(BASE_RATE_REGISTRY)
    return {
        "base_rates": entries,
        "count": len(entries),
        "categories": list_categories(),
    }
