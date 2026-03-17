"""
Augur — Calibration Tracking

Tracks forecast outcomes and computes calibration metrics:
  - Per-bucket confidence-vs-resolution rates
  - Brier scores
  - Quarterly aggregation with Wilson score confidence intervals
"""
from __future__ import annotations

import math
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory resolution store
# ---------------------------------------------------------------------------

_resolved_forecasts: list[dict] = []
_MAX_RESOLVED = 2000

# Bucket boundaries: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
_BUCKET_EDGES = [i / 10 for i in range(11)]
_NUM_BUCKETS = 10


def _bucket_index(p: float) -> int:
    """Map a probability to its calibration bucket index (0-9)."""
    idx = int(p * 10)
    return min(idx, _NUM_BUCKETS - 1)


def _bucket_label(idx: int) -> str:
    lo = idx * 10
    hi = (idx + 1) * 10
    return f"{lo}-{hi}%"


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) bounds. With z=1.96 this gives a 95% CI.
    Returns (0.0, 1.0) if total == 0.
    """
    if total == 0:
        return 0.0, 1.0
    n = total
    p_hat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) / denom
    lo = max(0.0, centre - spread)
    hi = min(1.0, centre + spread)
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(forecasts: list[dict]) -> Optional[float]:
    """Mean squared error of probability vs binary outcome."""
    if not forecasts:
        return None
    total = sum(
        (f["ensemble_probability"] - (1.0 if f["actual_outcome"] else 0.0)) ** 2
        for f in forecasts
    )
    return round(total / len(forecasts), 4)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_forecast(
    question: str,
    actual_outcome: bool,
    ensemble_probability: Optional[float] = None,
) -> dict:
    """
    Record that a forecast resolved to a known outcome.

    If ensemble_probability is not provided, we search the forecast history
    (imported from api module) for a matching question.
    """
    from .api import _forecast_history

    prob = ensemble_probability
    if prob is None:
        # Find the most recent matching forecast
        for entry in _forecast_history:
            if entry["question"] == question:
                prob = entry["ensemble_probability"]
                break

    if prob is None:
        raise ValueError(f"No matching forecast found for question: {question!r}")

    record = {
        "question": question,
        "ensemble_probability": prob,
        "actual_outcome": actual_outcome,
        "resolved_at": time.time(),
        "quarter": _quarter_label(time.time()),
    }
    _resolved_forecasts.insert(0, record)
    if len(_resolved_forecasts) > _MAX_RESOLVED:
        _resolved_forecasts[:] = _resolved_forecasts[:_MAX_RESOLVED]

    logger.info(
        f"[Augur] Resolved: q={question[:60]!r} "
        f"p={prob:.0%} outcome={actual_outcome}"
    )
    return record


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _quarter_label(ts: float) -> str:
    """Return 'YYYY-QN' for a unix timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


# ---------------------------------------------------------------------------
# Calibration curve computation
# ---------------------------------------------------------------------------

def calibration_curve(forecasts: list[dict]) -> list[dict]:
    """
    Compute calibration buckets from a list of resolved forecasts.

    Returns a list of dicts, one per bucket:
        {bucket, midpoint, count, resolution_rate, ci_lower, ci_upper}
    """
    buckets: list[list[dict]] = [[] for _ in range(_NUM_BUCKETS)]
    for f in forecasts:
        idx = _bucket_index(f["ensemble_probability"])
        buckets[idx].append(f)

    result = []
    for i, bucket_items in enumerate(buckets):
        n = len(bucket_items)
        successes = sum(1 for f in bucket_items if f["actual_outcome"])
        rate = round(successes / n, 4) if n > 0 else None
        ci_lo, ci_hi = wilson_interval(successes, n)
        result.append({
            "bucket": _bucket_label(i),
            "midpoint": round((i + 0.5) / _NUM_BUCKETS, 2),
            "count": n,
            "resolution_rate": rate,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })
    return result


# ---------------------------------------------------------------------------
# Full calibration report (optionally grouped by quarter)
# ---------------------------------------------------------------------------

def calibration_report(quarter: Optional[str] = None) -> dict:
    """
    Build calibration data suitable for dashboard visualization.

    If quarter is given (e.g. '2026-Q1'), filter to that quarter only.
    Always includes an 'overall' section plus per-quarter breakdown.
    """
    all_forecasts = list(_resolved_forecasts)

    # Group by quarter
    by_quarter: Dict[str, list[dict]] = {}
    for f in all_forecasts:
        q = f.get("quarter", _quarter_label(f["resolved_at"]))
        by_quarter.setdefault(q, []).append(f)

    # If filtering to a specific quarter
    if quarter:
        filtered = by_quarter.get(quarter, [])
        return {
            "quarter": quarter,
            "total_resolved": len(filtered),
            "brier_score": brier_score(filtered),
            "calibration_curve": calibration_curve(filtered),
        }

    # Overall + per-quarter
    quarters_data = []
    for q_label in sorted(by_quarter.keys()):
        q_forecasts = by_quarter[q_label]
        quarters_data.append({
            "quarter": q_label,
            "total_resolved": len(q_forecasts),
            "brier_score": brier_score(q_forecasts),
            "calibration_curve": calibration_curve(q_forecasts),
        })

    return {
        "overall": {
            "total_resolved": len(all_forecasts),
            "brier_score": brier_score(all_forecasts),
            "calibration_curve": calibration_curve(all_forecasts),
        },
        "quarters": quarters_data,
    }
