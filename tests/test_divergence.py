"""Tests for the Augur divergence detection and notification system."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from augur.divergence import detect_divergence, notify_divergence, _get_webhook_urls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _est(specialist: str, probability: float, confidence: float = 0.7, status: str = "success") -> dict:
    """Shorthand for building a specialist estimate dict."""
    return {
        "specialist": specialist,
        "probability": probability,
        "confidence": confidence,
        "reasoning": "test",
        "key_assumptions": [],
        "key_uncertainties": [],
        "would_change_if": None,
        "status": status,
        "latency_ms": 100.0,
    }


# ---------------------------------------------------------------------------
# detect_divergence
# ---------------------------------------------------------------------------

class TestDetectDivergence:

    def test_returns_none_when_all_agree(self):
        estimates = [
            _est("a", 0.50),
            _est("b", 0.52),
            _est("c", 0.48),
            _est("d", 0.51),
        ]
        assert detect_divergence(estimates) is None

    def test_returns_none_with_fewer_than_min_divergent_successful(self):
        estimates = [
            _est("a", 0.30),
            _est("b", 0.80),
        ]
        assert detect_divergence(estimates) is None

    def test_returns_none_when_failed_specialists_excluded(self):
        estimates = [
            _est("a", 0.30),
            _est("b", 0.80, status="timeout"),
            _est("c", 0.70, status="api_error"),
            _est("d", 0.90, status="parse_error"),
        ]
        # Only 1 successful — not enough
        assert detect_divergence(estimates) is None

    def test_detects_clear_three_way_divergence(self):
        # Four specialists: three far from the median, triggering divergence
        estimates = [
            _est("low", 0.15),
            _est("mid_low", 0.40),
            _est("mid_high", 0.65),
            _est("high", 0.85),
        ]
        # Median of [0.15, 0.40, 0.65, 0.85] = 0.525
        # half_threshold = 10pp. Distance from median:
        #   low: 37.5pp (divergent), mid_low: 12.5pp (divergent),
        #   mid_high: 12.5pp (divergent), high: 32.5pp (divergent)
        # Spread = 70pp, all 4 divergent
        result = detect_divergence(estimates, threshold_pp=20.0, min_divergent=3)
        assert result is not None
        assert result["flagged"] is True
        assert result["spread_pp"] == 70.0
        assert result["num_divergent"] >= 3

    def test_detects_divergence_with_mixed_statuses(self):
        estimates = [
            _est("a", 0.10),
            _est("b", 0.55),
            _est("c", 0.90),
            _est("d", 0.15),
            _est("e", 0.45, status="timeout"),  # excluded
        ]
        # 4 successful: [0.10, 0.15, 0.55, 0.90], median=0.35, spread=80pp
        # half_threshold=10pp. All are >10pp from 0.35 => 4 divergent
        result = detect_divergence(estimates)
        assert result is not None
        assert result["num_successful"] == 4

    def test_spread_below_threshold_returns_none(self):
        estimates = [
            _est("a", 0.40),
            _est("b", 0.50),
            _est("c", 0.55),
        ]
        # Spread = 15pp, below default 20pp
        assert detect_divergence(estimates) is None

    def test_custom_threshold(self):
        estimates = [
            _est("a", 0.35),
            _est("b", 0.50),
            _est("c", 0.60),
        ]
        # Spread = 25pp, median = 0.50, half_threshold = 5pp
        # a: 15pp from median (divergent), b: 0pp (aligned), c: 10pp (divergent)
        # Only 2 divergent with min_divergent=3 — won't trigger
        # Use min_divergent=2 instead
        result = detect_divergence(estimates, threshold_pp=10.0, min_divergent=2)
        assert result is not None
        assert result["threshold_pp"] == 10.0

    def test_clusters_formed_correctly(self):
        estimates = [
            _est("lo1", 0.20),
            _est("lo2", 0.22),
            _est("hi1", 0.80),
            _est("hi2", 0.82),
        ]
        result = detect_divergence(estimates, threshold_pp=20.0, min_divergent=3)
        assert result is not None
        assert len(result["clusters"]) == 2
        # First cluster should contain the low pair
        low_cluster = result["clusters"][0]
        assert set(low_cluster["specialists"]) == {"lo1", "lo2"}
        high_cluster = result["clusters"][1]
        assert set(high_cluster["specialists"]) == {"hi1", "hi2"}

    def test_divergence_info_structure(self):
        estimates = [
            _est("a", 0.10),
            _est("b", 0.45),
            _est("c", 0.90),
            _est("d", 0.15),
        ]
        # 4 successful: [0.10, 0.15, 0.45, 0.90], median=0.30, spread=80pp
        # half_threshold=10pp. All >10pp from 0.30 => 4 divergent
        result = detect_divergence(estimates)
        assert result is not None
        # Verify all expected keys
        assert "flagged" in result
        assert "spread_pp" in result
        assert "median_probability" in result
        assert "threshold_pp" in result
        assert "divergent_specialists" in result
        assert "aligned_specialists" in result
        assert "clusters" in result
        assert "num_divergent" in result
        assert "num_successful" in result
        assert result["num_successful"] == 4

    def test_not_enough_divergent_specialists(self):
        # 4 specialists, but only 2 far from median — below min_divergent=3
        estimates = [
            _est("a", 0.50),
            _est("b", 0.51),
            _est("c", 0.10),
            _est("d", 0.90),
        ]
        # Median ~ 0.505, spread = 80pp, but only c and d are >10pp from median
        # a and b are aligned. So only 2 divergent, below min_divergent=3
        result = detect_divergence(estimates, threshold_pp=20.0, min_divergent=3)
        assert result is None

    def test_empty_estimates(self):
        assert detect_divergence([]) is None

    def test_all_failed(self):
        estimates = [
            _est("a", 0.10, status="timeout"),
            _est("b", 0.90, status="api_error"),
        ]
        assert detect_divergence(estimates) is None


# ---------------------------------------------------------------------------
# Webhook URL parsing
# ---------------------------------------------------------------------------

class TestGetWebhookUrls:

    def test_empty_env_returns_empty(self):
        with patch.dict("os.environ", {"AUGUR_DIVERGENCE_WEBHOOKS": ""}):
            assert _get_webhook_urls() == []

    def test_missing_env_returns_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _get_webhook_urls() == []

    def test_single_url(self):
        with patch.dict("os.environ", {"AUGUR_DIVERGENCE_WEBHOOKS": "https://hooks.example.com/alert"}):
            urls = _get_webhook_urls()
            assert urls == ["https://hooks.example.com/alert"]

    def test_multiple_urls_with_whitespace(self):
        with patch.dict("os.environ", {"AUGUR_DIVERGENCE_WEBHOOKS": "  https://a.com , https://b.com ,  "}):
            urls = _get_webhook_urls()
            assert urls == ["https://a.com", "https://b.com"]


# ---------------------------------------------------------------------------
# notify_divergence
# ---------------------------------------------------------------------------

class TestNotifyDivergence:

    @pytest.mark.asyncio
    async def test_noop_when_no_callbacks(self):
        # Should return without error when no URLs configured
        with patch.dict("os.environ", {"AUGUR_DIVERGENCE_WEBHOOKS": ""}):
            await notify_divergence("Will X?", {"flagged": True}, callbacks=None)

    @pytest.mark.asyncio
    async def test_noop_with_empty_explicit_callbacks(self):
        await notify_divergence("Will X?", {"flagged": True}, callbacks=[])

    @pytest.mark.asyncio
    async def test_posts_to_webhook(self):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("augur.divergence.httpx.AsyncClient", return_value=mock_client_instance):
            await notify_divergence(
                "Will X happen?",
                {"flagged": True, "spread_pp": 40.0},
                callbacks=["https://hooks.example.com/alert"],
            )

        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "https://hooks.example.com/alert"
        payload = call_args[1]["json"]
        assert payload["event"] == "specialist_divergence"
        assert payload["question"] == "Will X happen?"
        assert payload["divergence"]["flagged"] is True

    @pytest.mark.asyncio
    async def test_logs_error_on_failure_without_raising(self):
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("augur.divergence.httpx.AsyncClient", return_value=mock_client_instance):
            # Should not raise
            await notify_divergence(
                "Will X?",
                {"flagged": True},
                callbacks=["https://down.example.com/hook"],
            )

    @pytest.mark.asyncio
    async def test_uses_env_var_when_callbacks_none(self):
        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"AUGUR_DIVERGENCE_WEBHOOKS": "https://env.example.com/hook"}), \
             patch("augur.divergence.httpx.AsyncClient", return_value=mock_client_instance):
            await notify_divergence("Will X?", {"flagged": True}, callbacks=None)

        mock_client_instance.post.assert_called_once()
        assert mock_client_instance.post.call_args[0][0] == "https://env.example.com/hook"
