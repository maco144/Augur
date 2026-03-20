"""
Augur — Structured Prediction Book

Multi-leg positions across correlated forecasting questions. Built for
operators who think in spreads, not single bets.

Concepts:
  - Instrument: a structured position defined by legs across questions
    (spread, conditional, basket, calendar)
  - Book: a participant's collection of active instruments with live P&L
  - Arbitrage scanner: detects when question aggregates violate probability
    axioms (Dutch book opportunities)

This is an arbitrage desk for prediction markets.
"""
from __future__ import annotations

import logging
import time
import uuid
from enum import Enum
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .submissions import (
    _combined_aggregate,
    _participants,
    _questions,
    _submissions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class InstrumentType(str, Enum):
    SPREAD = "spread"          # Long Q_A, short Q_B — profit on relative move
    CONDITIONAL = "conditional" # P(A|B) — what happens to A if B resolves yes
    BASKET = "basket"          # Weighted exposure to N questions
    CALENDAR = "calendar"      # Same thesis, different time horizons


class LegSide(str, Enum):
    LONG = "long"   # Profit if probability rises
    SHORT = "short"  # Profit if probability falls


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LegCreate(BaseModel):
    question_id: str
    side: LegSide
    weight: float = Field(1.0, ge=0.01, le=10.0, description="Relative weight in the instrument.")


class Leg(BaseModel):
    question_id: str
    side: LegSide
    weight: float
    entry_probability: float  # Snapshot at instrument creation
    current_probability: float
    pnl: float  # Unrealized P&L in probability points


class InstrumentCreate(BaseModel):
    participant_id: str
    name: str = Field(description="Descriptive name, e.g. 'Fed hike spread Q2 vs Q3'")
    type: InstrumentType
    legs: list[LegCreate] = Field(min_length=1, max_length=10)
    thesis: Optional[str] = Field(None, description="Why this position exists.")


class Instrument(BaseModel):
    id: str
    participant_id: str
    name: str
    type: InstrumentType
    legs: list[Leg]
    thesis: Optional[str] = None
    created_at: float
    closed_at: Optional[float] = None
    closed: bool = False
    net_pnl: float = 0.0


class ArbitrageOpportunity(BaseModel):
    type: str  # "dutch_book", "exhaustive_violation", "correlation_dislocation"
    description: str
    questions: list[str]  # question IDs involved
    implied_edge: float  # How mispriced, in probability points
    confidence: float  # How sure the scanner is


class CorrelationPair(BaseModel):
    question_a_id: str
    question_b_id: str
    question_a_text: str
    question_b_text: str
    prob_a: float
    prob_b: float
    implied_joint: float  # Naive P(A)*P(B)
    participant_overlap: float  # Fraction of participants who submitted to both
    directional_agreement: float  # Do overlapping participants lean the same way?


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_instruments: dict[str, Instrument] = {}  # instrument_id -> Instrument
_participant_books: dict[str, list[str]] = {}  # participant_id -> [instrument_ids]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _current_prob(question_id: str) -> float:
    """Get current combined aggregate probability for a question."""
    prob, _ = _combined_aggregate(question_id)
    return prob


def _mark_to_market(leg: Leg) -> Leg:
    """Update a leg with current market data."""
    current = _current_prob(leg.question_id)
    if leg.side == LegSide.LONG:
        pnl = (current - leg.entry_probability) * leg.weight
    else:
        pnl = (leg.entry_probability - current) * leg.weight
    leg.current_probability = current
    leg.pnl = round(pnl, 4)
    return leg


def _mtm_instrument(inst: Instrument) -> Instrument:
    """Mark-to-market all legs and compute net P&L."""
    for leg in inst.legs:
        _mark_to_market(leg)
    inst.net_pnl = round(sum(leg.pnl for leg in inst.legs), 4)
    return inst


def _find_exhaustive_sets() -> list[ArbitrageOpportunity]:
    """
    Find groups of questions that should sum to 1.0 (exhaustive/mutually exclusive).

    Heuristic: questions with identical deadlines and overlapping keywords
    are candidates for exhaustive sets. If their probabilities sum to != 1.0,
    there's an arb.
    """
    opps: list[ArbitrageOpportunity] = []

    # Group questions by deadline (exact match = likely same event, different outcomes)
    by_deadline: dict[float, list] = {}
    for q in _questions.values():
        if q.resolved:
            continue
        by_deadline.setdefault(q.deadline, []).append(q)

    for deadline, group in by_deadline.items():
        if len(group) < 2:
            continue

        # Check if probabilities sum to ~1.0
        probs = [(q.id, _current_prob(q.id)) for q in group]
        total = sum(p for _, p in probs)

        if abs(total - 1.0) > 0.05:  # More than 5pp off
            edge = abs(total - 1.0)
            direction = "overpriced" if total > 1.0 else "underpriced"
            opps.append(ArbitrageOpportunity(
                type="exhaustive_violation",
                description=(
                    f"{len(group)} questions sharing deadline "
                    f"{deadline:.0f} sum to {total:.2%} (should be ~100%). "
                    f"Market is {direction} by {edge:.2%}. "
                    f"{'Sell all legs.' if total > 1.0 else 'Buy all legs.'}"
                ),
                questions=[q_id for q_id, _ in probs],
                implied_edge=round(edge, 4),
                confidence=min(0.9, 0.5 + len(group) * 0.1),
            ))

    return opps


def _find_dutch_books() -> list[ArbitrageOpportunity]:
    """
    Detect Dutch book opportunities: where open/dark pool spreads on the
    same question create a guaranteed profit.

    If open says 70% and dark says 40% on the same question, someone is
    wrong and you can take both sides.
    """
    from .submissions import _pool_aggregate

    opps: list[ArbitrageOpportunity] = []

    for q_id, q in _questions.items():
        if q.resolved:
            continue
        # Dark pool only visible after deadline
        if time.time() <= q.deadline:
            continue

        open_prob, _ = _pool_aggregate(q_id, "open")
        dark_prob, _ = _pool_aggregate(q_id, "dark")

        spread = abs(open_prob - dark_prob)
        if spread > 0.15:  # 15pp+ spread between pools
            opps.append(ArbitrageOpportunity(
                type="dutch_book",
                description=(
                    f"Pool divergence on '{q.question[:60]}': "
                    f"open={open_prob:.0%} vs dark={dark_prob:.0%} "
                    f"(spread={spread:.0%}). "
                    f"Dark pool participants had conviction without social signal. "
                    f"One pool is mispriced."
                ),
                questions=[q_id],
                implied_edge=round(spread, 4),
                confidence=round(min(0.95, 0.5 + spread), 4),
            ))

    return opps


def _find_correlation_dislocations() -> list[ArbitrageOpportunity]:
    """
    Find question pairs where the same participants submitted very different
    probabilities to logically correlated questions.

    If analyst X says 80% on 'Fed hikes in June' but 20% on 'Bond yields
    rise in Q3', there might be a dislocation worth trading.
    """
    opps: list[ArbitrageOpportunity] = []
    questions = [q for q in _questions.values() if not q.resolved]

    for i, q_a in enumerate(questions):
        for q_b in questions[i + 1:]:
            # Find participants who submitted to both
            subs_a = {s.participant_id: s for s in _submissions.get(q_a.id, [])}
            subs_b = {s.participant_id: s for s in _submissions.get(q_b.id, [])}

            overlap = set(subs_a.keys()) & set(subs_b.keys())
            if len(overlap) < 2:
                continue

            # Check directional agreement
            agreements = 0
            for pid in overlap:
                a_above = subs_a[pid].probability > 0.5
                b_above = subs_b[pid].probability > 0.5
                if a_above == b_above:
                    agreements += 1

            agreement_rate = agreements / len(overlap)

            # High overlap + low agreement = potential dislocation
            if agreement_rate < 0.3 and len(overlap) >= 3:
                prob_a = _current_prob(q_a.id)
                prob_b = _current_prob(q_b.id)

                opps.append(ArbitrageOpportunity(
                    type="correlation_dislocation",
                    description=(
                        f"Participants who forecast both questions disagree on direction: "
                        f"'{q_a.question[:40]}' ({prob_a:.0%}) vs "
                        f"'{q_b.question[:40]}' ({prob_b:.0%}). "
                        f"Only {agreement_rate:.0%} directional agreement across "
                        f"{len(overlap)} shared forecasters. "
                        f"Either the questions are uncorrelated or one leg is mispriced."
                    ),
                    questions=[q_a.id, q_b.id],
                    implied_edge=round(1.0 - agreement_rate, 4),
                    confidence=round(min(0.8, len(overlap) * 0.1), 4),
                ))

    return opps


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/v1/book", tags=["Structured Book"])

# -- Instruments --


@router.post("/instruments", summary="Create a structured instrument (spread, conditional, basket, calendar)")
async def create_instrument(body: InstrumentCreate) -> Instrument:
    participant = _participants.get(body.participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found. Register first.")

    # Validate all legs reference real questions
    legs: list[Leg] = []
    for leg_in in body.legs:
        q = _questions.get(leg_in.question_id)
        if not q:
            raise HTTPException(status_code=404, detail=f"Question {leg_in.question_id!r} not found")
        if q.resolved:
            raise HTTPException(status_code=409, detail=f"Question {leg_in.question_id!r} already resolved")

        entry_prob = _current_prob(leg_in.question_id)
        legs.append(Leg(
            question_id=leg_in.question_id,
            side=leg_in.side,
            weight=leg_in.weight,
            entry_probability=entry_prob,
            current_probability=entry_prob,
            pnl=0.0,
        ))

    # Validate instrument type constraints
    if body.type == InstrumentType.SPREAD and len(legs) != 2:
        raise HTTPException(status_code=422, detail="Spread instruments require exactly 2 legs")

    inst = Instrument(
        id=str(uuid.uuid4()),
        participant_id=body.participant_id,
        name=body.name,
        type=body.type,
        legs=legs,
        thesis=body.thesis,
        created_at=time.time(),
    )
    _instruments[inst.id] = inst
    _participant_books.setdefault(body.participant_id, []).append(inst.id)

    logger.info(
        f"[Augur] Instrument created: {inst.id[:8]} — {body.type.value} "
        f"'{body.name}' ({len(legs)} legs) by {participant.display_name}"
    )
    return inst


@router.get("/instruments/{instrument_id}", summary="Get instrument with live mark-to-market")
async def get_instrument(instrument_id: str) -> Instrument:
    inst = _instruments.get(instrument_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instrument not found")
    return _mtm_instrument(inst)


@router.post("/instruments/{instrument_id}/close", summary="Close an instrument and lock in P&L")
async def close_instrument(instrument_id: str) -> Instrument:
    inst = _instruments.get(instrument_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instrument not found")
    if inst.closed:
        raise HTTPException(status_code=409, detail="Instrument already closed")

    _mtm_instrument(inst)
    inst.closed = True
    inst.closed_at = time.time()

    logger.info(
        f"[Augur] Instrument closed: {inst.id[:8]} — "
        f"net P&L {inst.net_pnl:+.4f} probability points"
    )
    return inst


# -- Book --


@router.get("/positions/{participant_id}", summary="Get a participant's full book with live P&L")
async def get_book(
    participant_id: str,
    include_closed: bool = Query(False, description="Include closed instruments"),
) -> dict:
    participant = _participants.get(participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    inst_ids = _participant_books.get(participant_id, [])
    instruments = []
    total_pnl = 0.0
    open_count = 0

    for iid in inst_ids:
        inst = _instruments.get(iid)
        if not inst:
            continue
        if inst.closed and not include_closed:
            continue

        _mtm_instrument(inst)
        instruments.append(inst)
        total_pnl += inst.net_pnl
        if not inst.closed:
            open_count += 1

    instruments.sort(key=lambda i: i.created_at, reverse=True)

    return {
        "participant": participant,
        "instruments": instruments,
        "summary": {
            "total_instruments": len(instruments),
            "open_positions": open_count,
            "net_pnl": round(total_pnl, 4),
        },
    }


# -- Arbitrage scanner --


@router.get("/arbitrage/scan", summary="Scan for arbitrage opportunities across all questions")
async def scan_arbitrage() -> dict:
    dutch = _find_dutch_books()
    exhaustive = _find_exhaustive_sets()
    dislocations = _find_correlation_dislocations()

    all_opps = dutch + exhaustive + dislocations
    all_opps.sort(key=lambda o: o.implied_edge, reverse=True)

    return {
        "opportunities": all_opps,
        "summary": {
            "total": len(all_opps),
            "dutch_books": len(dutch),
            "exhaustive_violations": len(exhaustive),
            "correlation_dislocations": len(dislocations),
        },
        "scanned_at": time.time(),
    }


# -- Correlation map --


@router.get("/correlations", summary="Cross-question correlation map from participant overlap")
async def correlation_map(
    min_overlap: int = Query(2, ge=1, description="Min shared participants to report"),
) -> dict:
    questions = [q for q in _questions.values() if not q.resolved]
    pairs: list[CorrelationPair] = []

    for i, q_a in enumerate(questions):
        for q_b in questions[i + 1:]:
            subs_a = {s.participant_id: s for s in _submissions.get(q_a.id, [])}
            subs_b = {s.participant_id: s for s in _submissions.get(q_b.id, [])}

            overlap = set(subs_a.keys()) & set(subs_b.keys())
            if len(overlap) < min_overlap:
                continue

            all_a = set(subs_a.keys())
            all_b = set(subs_b.keys())
            union = all_a | all_b
            overlap_frac = len(overlap) / len(union) if union else 0.0

            agreements = sum(
                1 for pid in overlap
                if (subs_a[pid].probability > 0.5) == (subs_b[pid].probability > 0.5)
            )
            agreement_rate = agreements / len(overlap)

            prob_a = _current_prob(q_a.id)
            prob_b = _current_prob(q_b.id)

            pairs.append(CorrelationPair(
                question_a_id=q_a.id,
                question_b_id=q_b.id,
                question_a_text=q_a.question,
                question_b_text=q_b.question,
                prob_a=prob_a,
                prob_b=prob_b,
                implied_joint=round(prob_a * prob_b, 4),
                participant_overlap=round(overlap_frac, 4),
                directional_agreement=round(agreement_rate, 4),
            ))

    pairs.sort(key=lambda p: p.participant_overlap, reverse=True)
    return {"pairs": pairs, "total": len(pairs)}
