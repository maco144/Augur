"""Tests for the Augur forecast API."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

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


def _mock_send_response(probability: float, confidence: float, reasoning: str = "test") -> str:
    """Return a raw JSON string as send_message would."""
    return json.dumps({
        "probability": probability,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_assumptions": ["assumption A"],
        "key_uncertainties": ["uncertainty B"],
        "would_change_if": "New evidence would change this.",
    })


def _one_provider_available():
    """Patch available_providers to report at least one key."""
    from augur.router import Provider
    return {Provider.ANTHROPIC: True, Provider.OPENAI: False, Provider.GOOGLE: False, Provider.OPENROUTER: False}


def _no_providers_available():
    from augur.router import Provider
    return {Provider.ANTHROPIC: False, Provider.OPENAI: False, Provider.GOOGLE: False, Provider.OPENROUTER: False}


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
    async def test_no_providers_returns_503(self):
        with patch("augur.api.available_providers", return_value=_no_providers_available()):
            resp = await _req("POST", "/v1/forecast", json={"question": "Will X happen?"})
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_unknown_specialist_returns_422(self):
        with patch("augur.api.available_providers", return_value=_one_provider_available()):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Will X?",
                "specialists": ["nonexistent_xyz"],
            })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_successful_forecast_shape(self):
        raw_response = _mock_send_response(0.7, 0.8)

        with patch("augur.api.available_providers", return_value=_one_provider_available()), \
             patch("augur.engine.send_message", new_callable=AsyncMock, return_value=raw_response), \
             patch("augur.engine.resolve_model", return_value=("claude-sonnet-4-6", "anthropic")), \
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
        assert "models_used" in body
        assert body["successful"] == 1
        assert body["failed"] == 0
        # Specialist should include model field
        assert body["specialists"][0]["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_ensemble_probability_is_weighted_average(self):
        responses = [
            _mock_send_response(0.6, 1.0),
            _mock_send_response(0.8, 1.0),
        ]
        call_count = 0

        async def fake_send(**kwargs):
            nonlocal call_count
            r = responses[call_count % len(responses)]
            call_count += 1
            return r

        with patch("augur.api.available_providers", return_value=_one_provider_available()), \
             patch("augur.engine.send_message", side_effect=fake_send), \
             patch("augur.engine.resolve_model", return_value=("claude-sonnet-4-6", "anthropic")), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test question?",
                "specialists": ["reasoner", "researcher"],
            })

        body = resp.json()
        assert abs(body["ensemble_probability"] - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_all_failed_gives_neutral_ensemble(self):
        with patch("augur.api.available_providers", return_value=_one_provider_available()), \
             patch("augur.engine.send_message", new_callable=AsyncMock, side_effect=Exception("API down")), \
             patch("augur.engine.resolve_model", return_value=("claude-sonnet-4-6", "anthropic")), \
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
        fenced_response = f"```json\n{payload}\n```"

        with patch("augur.api.available_providers", return_value=_one_provider_available()), \
             patch("augur.engine.send_message", new_callable=AsyncMock, return_value=fenced_response), \
             patch("augur.engine.resolve_model", return_value=("claude-sonnet-4-6", "anthropic")), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test?",
                "specialists": ["reasoner"],
            })

        s = resp.json()["specialists"][0]
        assert s["status"] == "success"
        assert abs(s["probability"] - 0.55) < 0.001

    @pytest.mark.asyncio
    async def test_models_used_reflects_diverse_models(self):
        """When specialists use different models, models_used should list all of them."""
        from augur.router import Provider

        responses = [
            _mock_send_response(0.6, 0.8),
            _mock_send_response(0.7, 0.8),
        ]
        call_count = 0

        async def fake_send(**kwargs):
            nonlocal call_count
            r = responses[call_count % len(responses)]
            call_count += 1
            return r

        models = [("deepseek/deepseek-r1-0528", Provider.OPENROUTER), ("claude-sonnet-4-6", Provider.ANTHROPIC)]
        resolve_count = 0

        def fake_resolve(cfg):
            nonlocal resolve_count
            m = models[resolve_count % len(models)]
            resolve_count += 1
            return m

        with patch("augur.api.available_providers", return_value=_one_provider_available()), \
             patch("augur.engine.send_message", side_effect=fake_send), \
             patch("augur.engine.resolve_model", side_effect=fake_resolve), \
             patch("augur.engine.synthesize", new_callable=AsyncMock, return_value=None):
            resp = await _req("POST", "/v1/forecast", json={
                "question": "Test?",
                "specialists": ["reasoner", "intelligence_analyst"],
            })

        body = resp.json()
        assert len(body["models_used"]) == 2
        assert "deepseek/deepseek-r1-0528" in body["models_used"]
        assert "claude-sonnet-4-6" in body["models_used"]
