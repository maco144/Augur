"""Tests for the Augur reference class base-rate library."""
from __future__ import annotations

import pytest
import httpx
from fastapi import FastAPI

from augur.base_rates import (
    BASE_RATE_REGISTRY,
    get_anchor,
    get_base_rates,
    list_categories,
    search_base_rates,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"category", "subcategory", "base_rate", "source", "time_period", "notes", "last_updated"}
EXPECTED_CATEGORIES = {
    "geopolitics", "technology", "markets", "climate", "healthcare",
    "elections", "cybersecurity", "energy", "legal", "space",
    "economics", "ai_ml",
}


class TestRegistryIntegrity:

    def test_registry_is_non_empty(self):
        assert len(BASE_RATE_REGISTRY) > 0

    def test_all_entries_have_required_keys(self):
        for i, entry in enumerate(BASE_RATE_REGISTRY):
            missing = REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"Entry {i} ({entry.get('subcategory', '?')}) missing keys: {missing}"

    def test_base_rates_are_valid_probabilities(self):
        for entry in BASE_RATE_REGISTRY:
            assert 0.0 <= entry["base_rate"] <= 1.0, (
                f"{entry['category']}/{entry['subcategory']} base_rate={entry['base_rate']} out of [0,1]"
            )

    def test_at_least_12_categories(self):
        cats = list_categories()
        assert len(cats) >= 12, f"Only {len(cats)} categories: {cats}"

    def test_expected_categories_present(self):
        cats = set(list_categories())
        missing = EXPECTED_CATEGORIES - cats
        assert not missing, f"Missing expected categories: {missing}"

    def test_last_updated_format(self):
        """Dates should be ISO-ish YYYY-MM-DD."""
        import re
        for entry in BASE_RATE_REGISTRY:
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", entry["last_updated"]), (
                f"{entry['subcategory']} has bad last_updated: {entry['last_updated']}"
            )


# ---------------------------------------------------------------------------
# get_base_rates
# ---------------------------------------------------------------------------

class TestGetBaseRates:

    def test_returns_list_for_known_category(self):
        results = get_base_rates("markets")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(r["category"] == "markets" for r in results)

    def test_case_insensitive(self):
        assert get_base_rates("Markets") == get_base_rates("markets")
        assert get_base_rates("AI_ML") == get_base_rates("ai_ml")

    def test_unknown_category_returns_empty(self):
        assert get_base_rates("nonexistent_domain") == []

    def test_strips_whitespace(self):
        assert get_base_rates("  markets  ") == get_base_rates("markets")


# ---------------------------------------------------------------------------
# get_anchor
# ---------------------------------------------------------------------------

class TestGetAnchor:

    def test_exact_lookup_found(self):
        anchor = get_anchor("healthcare", "clinical_trial_phase2_success")
        assert anchor is not None
        assert anchor["base_rate"] == 0.29

    def test_exact_lookup_not_found(self):
        assert get_anchor("healthcare", "nonexistent_sub") is None

    def test_case_insensitive(self):
        a = get_anchor("Healthcare", "Clinical_Trial_Phase2_Success")
        b = get_anchor("healthcare", "clinical_trial_phase2_success")
        assert a == b

    def test_returns_all_required_fields(self):
        anchor = get_anchor("markets", "us_recession_annual")
        assert anchor is not None
        for key in REQUIRED_KEYS:
            assert key in anchor


# ---------------------------------------------------------------------------
# search_base_rates
# ---------------------------------------------------------------------------

class TestSearchBaseRates:

    def test_single_keyword(self):
        results = search_base_rates("recession")
        assert len(results) >= 1
        assert any("recession" in r["subcategory"] or "recession" in r["notes"].lower() for r in results)

    def test_multi_keyword(self):
        results = search_base_rates("clinical trial phase")
        assert len(results) >= 2

    def test_empty_query_returns_empty(self):
        assert search_base_rates("") == []
        assert search_base_rates("   ") == []

    def test_no_match_returns_empty(self):
        assert search_base_rates("xyzzyplugh") == []


# ---------------------------------------------------------------------------
# list_categories
# ---------------------------------------------------------------------------

class TestListCategories:

    def test_returns_sorted(self):
        cats = list_categories()
        assert cats == sorted(cats)

    def test_no_duplicates(self):
        cats = list_categories()
        assert len(cats) == len(set(cats))


# ---------------------------------------------------------------------------
# API endpoint integration
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    from augur.api import router
    app = FastAPI()
    app.include_router(router)
    return app


async def _get(path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_make_app()),
        base_url="http://testserver",
    ) as client:
        return await client.get(path, **kwargs)


class TestBaseRatesEndpoint:

    @pytest.mark.asyncio
    async def test_returns_200_with_all_entries(self):
        resp = await _get("/v1/forecast/base-rates")
        assert resp.status_code == 200
        body = resp.json()
        assert "base_rates" in body
        assert "count" in body
        assert "categories" in body
        assert body["count"] == len(BASE_RATE_REGISTRY)

    @pytest.mark.asyncio
    async def test_category_filter(self):
        resp = await _get("/v1/forecast/base-rates", params={"category": "healthcare"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] > 0
        assert all(r["category"] == "healthcare" for r in body["base_rates"])

    @pytest.mark.asyncio
    async def test_unknown_category_returns_empty_list(self):
        resp = await _get("/v1/forecast/base-rates", params={"category": "zzz_fake"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_keyword_search(self):
        resp = await _get("/v1/forecast/base-rates", params={"q": "FDA approval"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1

    @pytest.mark.asyncio
    async def test_categories_always_present(self):
        resp = await _get("/v1/forecast/base-rates", params={"category": "markets"})
        body = resp.json()
        assert len(body["categories"]) >= 12
