"""
Augur — Scoring Engine

Multi-dimensional scoring for dual-pool prediction submissions:
  - Quality: Brier score (lower is better)
  - Novelty: directional divergence from consensus (higher is better)
  - Pool multipliers: dark pool rewards conviction, open pool rewards early movers
  - Combined: weighted blend feeding into reputation
"""
from __future__ import annotations


def brier(probability: float, outcome: bool) -> float:
    """Squared error of a probability estimate vs binary outcome. Range [0, 1]."""
    return (probability - (1.0 if outcome else 0.0)) ** 2


def resolved_novelty(submission_prob: float, consensus_at_time: float, outcome: bool) -> float:
    """
    Directional novelty: reward divergence from consensus only if it was
    in the correct direction. Returns 0.0 if divergence was wrong-way.

    Range [0, 1].
    """
    divergence = abs(submission_prob - consensus_at_time)
    correct_direction = (
        (submission_prob > consensus_at_time and outcome)
        or (submission_prob < consensus_at_time and not outcome)
    )
    return divergence if correct_direction else 0.0


def dark_multiplier(confidence: float) -> float:
    """
    Dark pool multiplier: rewards conviction without social signal.

    High confidence in the dark = you had real conviction.
    Range [1.0, 2.0].
    """
    return 1.0 + confidence


def early_multiplier(submitted_at: float, question_created_at: float, deadline: float) -> float:
    """
    Open pool multiplier: rewards early submissions.

    Full bonus (2.0x) at question creation, decays to 1.0x at deadline.
    Range [1.0, 2.0].
    """
    window = deadline - question_created_at
    if window <= 0:
        return 1.0
    elapsed = submitted_at - question_created_at
    remaining_pct = max(0.0, min(1.0, 1.0 - elapsed / window))
    return 1.0 + remaining_pct


def score_submission(
    probability: float,
    confidence: float,
    consensus_at_time: float,
    outcome: bool,
    pool: str,
    submitted_at: float = 0.0,
    question_created_at: float = 0.0,
    deadline: float = 0.0,
    alpha: float = 0.7,
) -> dict:
    """
    Compute all scoring dimensions for a single submission.

    Returns dict with brier_score, novelty_score, multiplier, combined_score.
    """
    b = brier(probability, outcome)
    novelty = resolved_novelty(probability, consensus_at_time, outcome)

    if pool == "dark":
        mult = dark_multiplier(confidence)
    else:
        mult = early_multiplier(submitted_at, question_created_at, deadline)

    # Base combined: quality + novelty blend
    quality = 1.0 - b
    base = alpha * quality + (1.0 - alpha) * novelty

    # Apply multiplier to the base score
    combined = round(base * mult, 4)

    return {
        "brier_score": round(b, 4),
        "novelty_score": round(novelty, 4),
        "pool_multiplier": round(mult, 4),
        "combined_score": combined,
    }


def combined_score(brier_score: float, novelty: float, alpha: float = 0.7) -> float:
    """
    Blend quality and novelty into a single score. Higher is better.
    Legacy interface without pool multipliers.
    """
    quality = 1.0 - brier_score
    return round(alpha * quality + (1.0 - alpha) * novelty, 4)


def reputation_weight(resolved_submissions: int, combined: float | None, min_track_record: int = 5) -> float:
    """
    Map a participant's combined score to an aggregation weight.

    New participants (fewer than min_track_record resolved submissions) get
    a neutral weight of 1.0. Established participants get their combined score
    as weight, floored at 0.1 to never fully silence anyone.
    """
    if resolved_submissions < min_track_record:
        return 1.0
    return max(0.1, combined or 1.0)
