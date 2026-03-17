"""Tests for the Augur calibration tracking system."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest
import httpx
from fastapi import FastAPI

from augur.calibration import (
    _resolved_forecasts,
    brier_score,
    calibration_curve,
    calibration_report,
    resolve_forecast,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    from augur.api import router
    app = FastAPI()
    app.include_router(router)
    return app


async def _req(method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_app()),
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


def _make_resolved(prob: float, outcome: bool, ts: float = None) -> dict:
    """Build a resolved forecast record for testing."""
    if ts is None:
        ts = time.time()
    from augur.calibration import _quarter_label
    return {
        "question": f"Test question p={prob}?",
        "ensemble_probability": prob,
        "actual_outcome": outcome,
        "resolved_at": ts,
        "quarter": _quarter_label(ts),
    }


@pytest.fixture(autouse=True)
def _clear_resolved():
    """Clear resolved forecasts before each test."""
    _resolved_forecasts.clear()
    yield
    _resolved_forecasts.clear()


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------

class TestWilsonInterval:

    def test_zero_total_returns_full_range(self):
        lo, hi = wilson_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_successes(self):
        lo, hi = wilson_interval(10, 10)
        assert lo > 0.6
        assert hi == 1.0

    def test_no_successes(self):
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0
        assert hi < 0.4

    def test_half_and_half(self):
        lo, hi = wilson_interval(50, 100)
        assert 0.3 < lo < 0.5
        assert 0.5 < hi < 0.7

    def test_bounds_never_exceed_01(self):
        for s in range(0, 21):
            lo, hi = wilson_interval(s, 20)
            assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

class TestBrierScore:

    def test_perfect_forecasts(self):
        forecasts = [
            {"ensemble_probability": 1.0, "actual_outcome": True},
            {"ensemble_probability": 0.0, "actual_outcome": False},
        ]
        assert brier_score(forecasts) == 0.0

    def test_worst_forecasts(self):
        forecasts = [
            {"ensemble_probability": 1.0, "actual_outcome": False},
            {"ensemble_probability": 0.0, "actual_outcome": True},
        ]
        assert brier_score(forecasts) == 1.0

    def test_uncertain_forecasts(self):
        forecasts = [
            {"ensemble_probability": 0.5, "actual_outcome": True},
            {"ensemble_probability": 0.5, "actual_outcome": False},
        ]
        assert brier_score(forecasts) == 0.25

    def test_empty_returns_none(self):
        assert brier_score([]) is None


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

class TestCalibrationCurve:

    def test_empty_input(self):
        result = calibration_curve([])
        assert len(result) == 10
        for bucket in result:
            assert bucket["count"] == 0
            assert bucket["resolution_rate"] is None

    def test_single_resolved_forecast(self):
        forecasts = [_make_resolved(0.75, True)]
        result = calibration_curve(forecasts)
        # 0.75 -> bucket index 7 (70-80%)
        bucket_7 = result[7]
        assert bucket_7["count"] == 1
        assert bucket_7["resolution_rate"] == 1.0

    def test_bucket_midpoints(self):
        result = calibration_curve([])
        assert result[0]["midpoint"] == 0.05
        assert result[4]["midpoint"] == 0.45
        assert result[9]["midpoint"] == 0.95

    def test_bucket_labels(self):
        result = calibration_curve([])
        assert result[0]["bucket"] == "0-10%"
        assert result[9]["bucket"] == "90-100%"

    def test_p_equals_1_goes_to_last_bucket(self):
        forecasts = [_make_resolved(1.0, True)]
        result = calibration_curve(forecasts)
        assert result[9]["count"] == 1

    def test_mixed_outcomes(self):
        forecasts = [
            _make_resolved(0.85, True),
            _make_resolved(0.82, False),
            _make_resolved(0.88, True),
        ]
        result = calibration_curve(forecasts)
        bucket_8 = result[8]
        assert bucket_8["count"] == 3
        assert abs(bucket_8["resolution_rate"] - 2 / 3) < 0.001
        # CI should bracket the resolution rate
        assert bucket_8["ci_lower"] <= bucket_8["resolution_rate"]
        assert bucket_8["ci_upper"] >= bucket_8["resolution_rate"]


# ---------------------------------------------------------------------------
# Resolve forecast
# ---------------------------------------------------------------------------

class TestResolveForecast:

    def test_resolve_with_explicit_probability(self):
        record = resolve_forecast("Will X happen?", True, ensemble_probability=0.7)
        assert record["actual_outcome"] is True
        assert record["ensemble_probability"] == 0.7
        assert "resolved_at" in record
        assert "quarter" in record

    def test_resolve_matches_history(self):
        from augur.api import _forecast_history
        _forecast_history.clear()
        _forecast_history.insert(0, {
            "question": "Will GDP grow?",
            "ensemble_probability": 0.65,
        })
        try:
            record = resolve_forecast("Will GDP grow?", False)
            assert record["ensemble_probability"] == 0.65
            assert record["actual_outcome"] is False
        finally:
            _forecast_history.clear()

    def test_resolve_no_match_raises(self):
        with pytest.raises(ValueError, match="No matching forecast"):
            resolve_forecast("Unknown question?", True)

    def test_resolved_stored_in_list(self):
        resolve_forecast("Q?", True, ensemble_probability=0.9)
        assert len(_resolved_forecasts) == 1
        assert _resolved_forecasts[0]["question"] == "Q?"


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

class TestCalibrationReport:

    def test_empty_report(self):
        report = calibration_report()
        assert report["overall"]["total_resolved"] == 0
        assert report["overall"]["brier_score"] is None
        assert report["quarters"] == []

    def test_report_with_data(self):
        _resolved_forecasts.extend([
            _make_resolved(0.8, True),
            _make_resolved(0.2, False),
            _make_resolved(0.6, True),
        ])
        report = calibration_report()
        assert report["overall"]["total_resolved"] == 3
        assert report["overall"]["brier_score"] is not None
        assert len(report["overall"]["calibration_curve"]) == 10

    def test_quarter_filter(self):
        # Q1 2026 timestamp: 2026-01-15
        q1_ts = 1768521600.0  # approx 2026-01-15 UTC
        _resolved_forecasts.extend([
            _make_resolved(0.7, True, ts=q1_ts),
            _make_resolved(0.3, False, ts=q1_ts),
        ])
        report = calibration_report(quarter="2026-Q1")
        assert report["quarter"] == "2026-Q1"
        assert report["total_resolved"] == 2

    def test_quarter_filter_no_match(self):
        _resolved_forecasts.append(_make_resolved(0.5, True))
        report = calibration_report(quarter="2099-Q4")
        assert report["total_resolved"] == 0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class TestResolveEndpoint:

    @pytest.mark.asyncio
    async def test_resolve_with_probability(self):
        resp = await _req("POST", "/v1/forecast/resolve", json={
            "question": "Will it rain?",
            "actual_outcome": True,
            "ensemble_probability": 0.8,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "resolved"
        assert body["record"]["actual_outcome"] is True

    @pytest.mark.asyncio
    async def test_resolve_no_match_returns_404(self):
        resp = await _req("POST", "/v1/forecast/resolve", json={
            "question": "Nonexistent question?",
            "actual_outcome": False,
        })
        assert resp.status_code == 404


class TestCalibrationEndpoint:

    @pytest.mark.asyncio
    async def test_empty_calibration(self):
        resp = await _req("GET", "/v1/forecast/calibration")
        assert resp.status_code == 200
        body = resp.json()
        assert "overall" in body
        assert "quarters" in body

    @pytest.mark.asyncio
    async def test_calibration_with_data(self):
        # Seed some resolved forecasts
        _resolved_forecasts.extend([
            _make_resolved(0.9, True),
            _make_resolved(0.1, False),
        ])
        resp = await _req("GET", "/v1/forecast/calibration")
        assert resp.status_code == 200
        body = resp.json()
        assert body["overall"]["total_resolved"] == 2
        assert body["overall"]["brier_score"] is not None

    @pytest.mark.asyncio
    async def test_calibration_quarter_filter(self):
        resp = await _req("GET", "/v1/forecast/calibration?quarter=2026-Q1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["quarter"] == "2026-Q1"
