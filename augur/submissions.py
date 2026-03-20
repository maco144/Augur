"""
Augur — Dual-Pool Open Submission System

Every question automatically gets two pools:
  - Open: visible consensus, submissions public, rewards early movers
  - Dark: sealed consensus, submissions hidden, rewards conviction

Participants submit to either or both. The spread between pools
measures herding/anchoring effects.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .scoring import reputation_weight, score_submission

logger = logging.getLogger(__name__)

VALID_POOLS = ("open", "dark")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuestionCreate(BaseModel):
    question: str = Field(description="A binary probabilistic question.")
    context: Optional[str] = Field(None, description="Background context.")
    resolution_criteria: str = Field(description="Unambiguous criteria for how this resolves yes/no.")
    deadline: float = Field(description="Unix timestamp — submissions close after this.")
    resolution_deadline: float = Field(description="Unix timestamp — question must resolve by this.")


class Question(BaseModel):
    id: str
    question: str
    context: Optional[str] = None
    resolution_criteria: str
    deadline: float
    resolution_deadline: float
    created_at: float
    resolved: bool = False
    outcome: Optional[bool] = None
    resolved_at: Optional[float] = None


class SubmissionCreate(BaseModel):
    participant_id: str = Field(description="ID of the registered participant.")
    probability: float = Field(ge=0.0, le=1.0, description="Probability estimate [0, 1].")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the estimate [0, 1].")
    pool: str = Field("open", description="'open' (visible) or 'dark' (sealed).")
    reasoning: Optional[str] = Field(None, description="Optional explanation.")


class Submission(BaseModel):
    id: str
    question_id: str
    participant_id: str
    probability: float
    confidence: float
    pool: str  # "open" | "dark"
    reasoning: Optional[str] = None
    consensus_at_time: float  # Pool-specific aggregate when submitted
    submitted_at: float
    brier_score: Optional[float] = None
    novelty_score: Optional[float] = None
    pool_multiplier: Optional[float] = None
    combined_score: Optional[float] = None


class ParticipantCreate(BaseModel):
    display_name: str = Field(description="Display name for the participant.")
    type: str = Field(description="'agent' | 'human' | 'ensemble'")


class Participant(BaseModel):
    id: str
    display_name: str
    type: str  # "agent" | "human" | "ensemble"
    created_at: float
    total_submissions: int = 0
    resolved_submissions: int = 0
    brier_score: Optional[float] = None
    novelty_score: Optional[float] = None
    combined_score: Optional[float] = None
    reputation_weight: float = 1.0


class ResolveQuestionRequest(BaseModel):
    outcome: bool = Field(description="True if the event occurred, False otherwise.")


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_questions: dict[str, Question] = {}
_submissions: dict[str, list[Submission]] = {}  # question_id -> list of submissions
_participants: dict[str, Participant] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pool_aggregate(question_id: str, pool: str) -> tuple[float, float]:
    """
    Compute reputation-weighted aggregate for a specific pool.

    Returns (probability, confidence). Returns (0.5, 0.0) if no submissions.
    """
    subs = [s for s in _submissions.get(question_id, []) if s.pool == pool]
    if not subs:
        return 0.5, 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    conf_sum = 0.0

    for s in subs:
        participant = _participants.get(s.participant_id)
        rep = participant.reputation_weight if participant else 1.0
        w = s.confidence * rep
        weighted_sum += s.probability * w
        weight_total += w
        conf_sum += s.confidence

    if weight_total == 0:
        return 0.5, 0.0

    return round(weighted_sum / weight_total, 4), round(conf_sum / len(subs), 4)


def _combined_aggregate(question_id: str) -> tuple[float, float]:
    """Aggregate across both pools."""
    subs = _submissions.get(question_id, [])
    if not subs:
        return 0.5, 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    conf_sum = 0.0

    for s in subs:
        participant = _participants.get(s.participant_id)
        rep = participant.reputation_weight if participant else 1.0
        w = s.confidence * rep
        weighted_sum += s.probability * w
        weight_total += w
        conf_sum += s.confidence

    if weight_total == 0:
        return 0.5, 0.0

    return round(weighted_sum / weight_total, 4), round(conf_sum / len(subs), 4)


def _pool_counts(question_id: str) -> dict[str, int]:
    subs = _submissions.get(question_id, [])
    counts = {"open": 0, "dark": 0}
    for s in subs:
        counts[s.pool] = counts.get(s.pool, 0) + 1
    return counts


def _update_participant_stats(participant_id: str) -> None:
    """Recompute a participant's rolling stats from all their scored submissions."""
    participant = _participants.get(participant_id)
    if not participant:
        return

    scored = []
    for subs in _submissions.values():
        for s in subs:
            if s.participant_id == participant_id and s.brier_score is not None:
                scored.append(s)

    if not scored:
        return

    avg_brier = sum(s.brier_score for s in scored) / len(scored)
    avg_novelty = sum(s.novelty_score or 0.0 for s in scored) / len(scored)
    avg_combined = sum(s.combined_score or 0.0 for s in scored) / len(scored)

    participant.resolved_submissions = len(scored)
    participant.brier_score = round(avg_brier, 4)
    participant.novelty_score = round(avg_novelty, 4)
    participant.combined_score = round(avg_combined, 4)
    participant.reputation_weight = reputation_weight(len(scored), avg_combined)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Submissions"])

# -- Questions --


@router.post("/v1/questions", summary="Register a new forecasting question (dual-pool)")
async def create_question(body: QuestionCreate) -> dict:
    q = Question(
        id=str(uuid.uuid4()),
        question=body.question,
        context=body.context,
        resolution_criteria=body.resolution_criteria,
        deadline=body.deadline,
        resolution_deadline=body.resolution_deadline,
        created_at=time.time(),
    )
    _questions[q.id] = q
    _submissions[q.id] = []
    logger.info(f"[Augur] Question registered (dual-pool): {q.id} — {q.question[:60]!r}")
    return {"question": q, "pools": ["open", "dark"]}


@router.get("/v1/questions", summary="List questions")
async def list_questions(
    status: Optional[str] = Query(None, description="'open' | 'resolved' | 'all'"),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    qs = list(_questions.values())
    now = time.time()
    if status == "open":
        qs = [q for q in qs if not q.resolved and q.deadline > now]
    elif status == "resolved":
        qs = [q for q in qs if q.resolved]
    qs.sort(key=lambda q: q.created_at, reverse=True)
    return {"questions": qs[:limit], "total": len(qs)}


@router.get("/v1/questions/{question_id}", summary="Get question with dual-pool aggregates")
async def get_question(question_id: str) -> dict:
    q = _questions.get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    open_prob, open_conf = _pool_aggregate(question_id, "open")
    counts = _pool_counts(question_id)
    combined_prob, combined_conf = _combined_aggregate(question_id)

    # Dark pool is always sealed until deadline passes or resolved
    dark_sealed = not q.resolved and time.time() <= q.deadline
    if dark_sealed:
        dark_prob, dark_conf = None, None
        spread = None
    else:
        dark_prob, dark_conf = _pool_aggregate(question_id, "dark")
        spread = round(abs(open_prob - dark_prob), 4) if dark_prob is not None else None

    return {
        "question": q,
        "pools": {
            "open": {
                "aggregate_probability": open_prob,
                "aggregate_confidence": open_conf,
                "submission_count": counts["open"],
            },
            "dark": {
                "aggregate_probability": dark_prob,
                "aggregate_confidence": dark_conf,
                "submission_count": counts["dark"] if not dark_sealed else None,
                "sealed": dark_sealed,
            },
        },
        "combined": {
            "aggregate_probability": combined_prob,
            "aggregate_confidence": combined_conf,
            "total_submissions": counts["open"] + counts["dark"],
        },
        "spread": spread,
    }


# -- Submissions --


@router.post("/v1/questions/{question_id}/submit", summary="Submit a prediction to open or dark pool")
async def submit_prediction(question_id: str, body: SubmissionCreate) -> Submission:
    q = _questions.get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    if q.resolved:
        raise HTTPException(status_code=409, detail="Question already resolved")
    now = time.time()
    if now > q.deadline:
        raise HTTPException(status_code=409, detail="Submission deadline has passed")
    if body.pool not in VALID_POOLS:
        raise HTTPException(status_code=422, detail=f"pool must be one of {VALID_POOLS}")

    participant = _participants.get(body.participant_id)
    if not participant:
        raise HTTPException(status_code=404, detail=f"Participant {body.participant_id!r} not found. Register first.")

    # Snapshot pool-specific consensus at submission time
    consensus_now, _ = _pool_aggregate(question_id, body.pool)

    sub = Submission(
        id=str(uuid.uuid4()),
        question_id=question_id,
        participant_id=body.participant_id,
        probability=body.probability,
        confidence=body.confidence,
        pool=body.pool,
        reasoning=body.reasoning,
        consensus_at_time=consensus_now,
        submitted_at=now,
    )
    _submissions[question_id].append(sub)
    participant.total_submissions += 1

    new_agg, _ = _pool_aggregate(question_id, body.pool)
    logger.info(
        f"[Augur] Submission ({body.pool}): {participant.display_name} → q={question_id[:8]} "
        f"p={body.probability:.0%} conf={body.confidence:.0%} "
        f"(pool consensus {consensus_now:.0%} → {new_agg:.0%})"
    )

    # Dark pool: redact consensus from response
    if body.pool == "dark":
        return sub.model_copy(update={"consensus_at_time": -1.0})
    return sub


@router.get("/v1/questions/{question_id}/submissions", summary="List submissions for a question")
async def list_submissions(
    question_id: str,
    pool: Optional[str] = Query(None, description="Filter by pool: 'open' or 'dark'"),
) -> dict:
    q = _questions.get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    dark_sealed = not q.resolved and time.time() <= q.deadline

    if pool and pool not in VALID_POOLS:
        raise HTTPException(status_code=422, detail=f"pool must be one of {VALID_POOLS}")

    subs = _submissions.get(question_id, [])

    if pool == "dark" and dark_sealed:
        raise HTTPException(status_code=403, detail="Dark pool is sealed — submissions hidden until deadline or resolution")
    if pool:
        subs = [s for s in subs if s.pool == pool]
    elif dark_sealed:
        # No pool filter: only show open submissions while dark is sealed
        subs = [s for s in subs if s.pool == "open"]

    return {"submissions": subs, "count": len(subs)}


# -- Resolution --


@router.post("/v1/questions/{question_id}/resolve", summary="Resolve a question and score both pools")
async def resolve_question(question_id: str, body: ResolveQuestionRequest) -> dict:
    q = _questions.get(question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    if q.resolved:
        raise HTTPException(status_code=409, detail="Question already resolved")

    # Mark resolved
    q.resolved = True
    q.outcome = body.outcome
    q.resolved_at = time.time()

    # Score all submissions with pool-aware multipliers
    subs = _submissions.get(question_id, [])
    affected_participants: set[str] = set()
    pool_stats = {"open": 0, "dark": 0}

    for s in subs:
        scores = score_submission(
            probability=s.probability,
            confidence=s.confidence,
            consensus_at_time=s.consensus_at_time,
            outcome=body.outcome,
            pool=s.pool,
            submitted_at=s.submitted_at,
            question_created_at=q.created_at,
            deadline=q.deadline,
        )
        s.brier_score = scores["brier_score"]
        s.novelty_score = scores["novelty_score"]
        s.pool_multiplier = scores["pool_multiplier"]
        s.combined_score = scores["combined_score"]
        affected_participants.add(s.participant_id)
        pool_stats[s.pool] += 1

    # Update participant rolling stats
    for pid in affected_participants:
        _update_participant_stats(pid)

    # Compute final pool spread
    open_prob, _ = _pool_aggregate(question_id, "open")
    dark_prob, _ = _pool_aggregate(question_id, "dark")
    spread = round(abs(open_prob - dark_prob), 4)

    logger.info(
        f"[Augur] Resolved: q={question_id[:8]} outcome={body.outcome} "
        f"open={pool_stats['open']} dark={pool_stats['dark']} spread={spread:.0%}"
    )

    return {
        "question": q,
        "scored_submissions": len(subs),
        "participants_updated": len(affected_participants),
        "pool_breakdown": pool_stats,
        "final_spread": spread,
    }


# -- Participants --


@router.post("/v1/participants", summary="Register a participant")
async def create_participant(body: ParticipantCreate) -> Participant:
    if body.type not in ("agent", "human", "ensemble"):
        raise HTTPException(status_code=422, detail="type must be 'agent', 'human', or 'ensemble'")
    p = Participant(
        id=str(uuid.uuid4()),
        display_name=body.display_name,
        type=body.type,
        created_at=time.time(),
    )
    _participants[p.id] = p
    logger.info(f"[Augur] Participant registered: {p.id} — {p.display_name} ({p.type})")
    return p


@router.get("/v1/participants/{participant_id}", summary="Get participant profile and stats")
async def get_participant(participant_id: str) -> Participant:
    p = _participants.get(participant_id)
    if not p:
        raise HTTPException(status_code=404, detail="Participant not found")
    return p


# -- Leaderboard --


@router.get("/v1/leaderboard", summary="Ranked participants by combined score")
async def leaderboard(limit: int = Query(50, ge=1, le=200)) -> dict:
    ranked = sorted(
        [p for p in _participants.values() if p.resolved_submissions > 0],
        key=lambda p: p.combined_score or 0.0,
        reverse=True,
    )
    return {
        "leaderboard": [
            {
                "rank": i + 1,
                "participant_id": p.id,
                "display_name": p.display_name,
                "type": p.type,
                "resolved_submissions": p.resolved_submissions,
                "brier_score": p.brier_score,
                "novelty_score": p.novelty_score,
                "combined_score": p.combined_score,
                "reputation_weight": p.reputation_weight,
            }
            for i, p in enumerate(ranked[:limit])
        ],
        "total_participants": len(ranked),
    }
