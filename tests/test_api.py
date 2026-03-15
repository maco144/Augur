"""Tests for the Augur forecast API."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from fastapi import FastAPI


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


def _mock_anthropic_response(probability: float, confidence: float, reasoning: str = "test") -> MagicMock:
    payload = json.dumps({
        "probability": probability,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_assumptions": ["assumption A"],
        "key_uncertainties": ["uncertainty B"],
        "would_change_if": "New evidence would change this.",
    })
    msg = MagicMock()
    msg.content = [MagicMock(text=payload)]
    return msg


class TestListSpecialists:

    @pytest.mark.asyncio
    async def test_returns_200_with_list(self):
        resp = await _req("GET", "/v1/forecast/specialists")
        assert resp.status_code == 200
        body = resp.json()
        assert "specialists" in body
        assert "defaults" in body
        assert len(body["specialists"]) > 0

    @pytest.mark.asyncio
    async def test_each_entry_has_required_fields(self):
        resp = await _req("GET", "/v1/forecast/specialists")
        for s in resp.json()["specialists"]:
            assert "name" in s
            assert "model" in s
            assert "default" in s

    @pytest.mark.asyncio
    async def test_default_specialists_flagged(self):
        from augur.engine import DEFAULT_SPECIALISTS
        resp = await _req("GET", "/v1/forecast/specialists")
        defaults = {s["name"] for s in resp.json()["specialists"] if s["default"]}
        for name in DEFAULT_SPECIALISTS:
            assert name in defaults


class TestRunForecast:

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_503(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            resp = await _req("POST", "/v1/forecast", json={"question": "Will X happen?"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_unknown_specialist_returns_422(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Will X?",
                "specialists": ["nonexistent_xyz"],
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_successful_forecast_shape(self):
        mock_msg = _mock_anthropic_response(0.7, 0.8)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_msg)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value="Synthesis text."):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Will the Fed cut rates before September 2026?",
                "specialists": ["reasoner"],
            })

        assert resp.status_code == 200
        body = resp.json()
        assert "ensemble_probability" in body
        assert "ensemble_confidence" in body
        assert "consensus" in body
        assert body["successful"] == 1
        assert body["failed"] == 0

    @pytest.mark.asyncio
    async def test_ensemble_probability_is_weighted_average(self):
        responses = [
            _mock_anthropic_response(0.6, 1.0),
            _mock_anthropic_response(0.8, 1.0),
        ]
        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            r = responses[call_count % len(responses)]
            call_count += 1
            return r

        mock_client = AsyncMock()
        mock_client.messages.create = fake_create

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test question?",
                "specialists": ["reasoner", "researcher"],
            })

        body = resp.json()
        assert abs(body["ensemble_probability"] - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_all_failed_gives_neutral_ensemble(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API down"))

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test?",
                "specialists": ["reasoner"],
            })

        assert resp.status_code == 200
        body = resp.json()
        assert body["ensemble_probability"] == 0.5
        assert body["failed"] == 1

    @pytest.mark.asyncio
    async def test_markdown_fences_stripped(self):
        payload = json.dumps({
            "probability": 0.55, "confidence": 0.6,
            "reasoning": "ok", "key_assumptions": [], "key_uncertainties": [],
        })
        fenced_msg = MagicMock()
        fenced_msg.content = [MagicMock(text=f"```json\n{payload}\n```")]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=fenced_msg)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}), \
             patch("anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test?",
                "specialists": ["reasoner"],
            })

        s = resp.json()["specialists"][0]
        assert s["status"] == "success"
        assert abs(s["probability"] - 0.55) < 0.001
