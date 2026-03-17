"""
Augur — Specialist Divergence Detection & Notification

Detects when 3+ specialists produce substantially different probability estimates
on the same forecast question, and optionally fires webhook notifications so that
downstream consumers can act on the disagreement.

Divergence rule:
  1. Filter to successful specialists only.
  2. Compute the median probability.
  3. A specialist is "divergent" if its estimate is more than (threshold_pp / 2)
     percentage points away from the median.
  4. If the total spread among successful specialists exceeds threshold_pp AND
     at least min_divergent specialists are divergent, flag the forecast.
"""
from __future__ import annotations

import asyncio
import logging
import os
import statistics
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_divergence(
    estimates: list[dict],
    threshold_pp: float = 20.0,
    min_divergent: int = 3,
) -> Optional[Dict[str, Any]]:
    """Check whether specialists diverge significantly on a forecast.

    Parameters
    ----------
    estimates:
        Raw specialist estimate dicts (must include ``status``, ``probability``,
        ``specialist``).
    threshold_pp:
        Minimum spread in percentage points to consider divergence meaningful.
    min_divergent:
        Minimum number of divergent specialists required to trigger a flag.

    Returns
    -------
    A structured divergence dict if significant divergence is detected, else
    ``None``.
    """
    successful = [e for e in estimates if e.get("status") == "success"]
    if len(successful) < min_divergent:
        return None

    probs = [e["probability"] for e in successful]
    spread_pp = (max(probs) - min(probs)) * 100

    if spread_pp < threshold_pp:
        return None

    median_prob = statistics.median(probs)
    half_threshold = threshold_pp / 2

    divergent: List[Dict[str, Any]] = []
    aligned: List[Dict[str, Any]] = []
    for e in successful:
        distance_pp = abs(e["probability"] - median_prob) * 100
        entry = {
            "specialist": e["specialist"],
            "probability": e["probability"],
            "distance_from_median_pp": round(distance_pp, 1),
        }
        if distance_pp > half_threshold:
            divergent.append(entry)
        else:
            aligned.append(entry)

    if len(divergent) < min_divergent:
        return None

    # Build clusters — group specialists by proximity (within half_threshold of
    # each other).  Simple greedy clustering: sort by probability, start a new
    # cluster when the gap exceeds half_threshold pp.
    sorted_specs = sorted(successful, key=lambda e: e["probability"])
    clusters: List[Dict[str, Any]] = []
    current_cluster: List[str] = [sorted_specs[0]["specialist"]]
    current_anchor = sorted_specs[0]["probability"]
    for e in sorted_specs[1:]:
        if (e["probability"] - current_anchor) * 100 <= half_threshold:
            current_cluster.append(e["specialist"])
        else:
            clusters.append({
                "specialists": current_cluster,
                "mean_probability": round(
                    statistics.mean(
                        s["probability"] for s in successful
                        if s["specialist"] in current_cluster
                    ),
                    3,
                ),
            })
            current_cluster = [e["specialist"]]
            current_anchor = e["probability"]
    # Flush last cluster
    clusters.append({
        "specialists": current_cluster,
        "mean_probability": round(
            statistics.mean(
                s["probability"] for s in successful
                if s["specialist"] in current_cluster
            ),
            3,
        ),
    })

    return {
        "flagged": True,
        "spread_pp": round(spread_pp, 1),
        "median_probability": round(median_prob, 3),
        "threshold_pp": threshold_pp,
        "divergent_specialists": divergent,
        "aligned_specialists": aligned,
        "clusters": clusters,
        "num_divergent": len(divergent),
        "num_successful": len(successful),
    }


# ---------------------------------------------------------------------------
# Webhook notification (fire-and-forget)
# ---------------------------------------------------------------------------

def _get_webhook_urls() -> List[str]:
    """Read webhook URLs from the ``AUGUR_DIVERGENCE_WEBHOOKS`` env var."""
    raw = os.environ.get("AUGUR_DIVERGENCE_WEBHOOKS", "")
    if not raw.strip():
        return []
    return [u.strip() for u in raw.split(",") if u.strip()]


async def _post_webhook(url: str, payload: dict) -> None:
    """POST a JSON payload to a single webhook URL. Logs errors, never raises."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            logger.warning(
                f"[Augur] Divergence webhook {url} returned {resp.status_code}"
            )
        else:
            logger.info(f"[Augur] Divergence webhook {url} notified ({resp.status_code})")
    except Exception as exc:
        logger.warning(f"[Augur] Divergence webhook {url} failed: {exc}")


async def notify_divergence(
    question: str,
    divergence_info: Dict[str, Any],
    callbacks: Optional[List[str]] = None,
) -> None:
    """Fire-and-forget webhook notifications for a divergence event.

    Parameters
    ----------
    question:
        The forecast question that triggered divergence.
    divergence_info:
        The structured dict returned by :func:`detect_divergence`.
    callbacks:
        Explicit list of webhook URLs.  Falls back to
        ``AUGUR_DIVERGENCE_WEBHOOKS`` env var if *None*.
    """
    urls = callbacks if callbacks is not None else _get_webhook_urls()
    if not urls:
        return

    payload = {
        "event": "specialist_divergence",
        "question": question,
        "divergence": divergence_info,
    }

    tasks = [_post_webhook(url, payload) for url in urls]
    await asyncio.gather(*tasks, return_exceptions=True)
