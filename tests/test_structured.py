"""Tests for structured prediction book — instruments, mark-to-market, arb scanner."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from augur.server import app
from augur.submissions import (
    _participants,
    _questions,
    _submissions,
)
from augur.structured import (
    _instruments,
    _participant_books,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    """Clear state between tests."""
    _questions.clear()
    _submissions.clear()
    _participants.clear()
    _instruments.clear()
    _participant_books.clear()
    yield
    _questions.clear()
    _submissions.clear()
    _participants.clear()
    _instruments.clear()
    _participant_books.clear()


def _make_participant(name: str = "Trader A", type_: str = "human") -> str:
    resp = client.post("/v1/participants", json={"display_name": name, "type": type_})
    assert resp.status_code == 200
    return resp.json()["id"]


def _make_question(question: str = "Will X happen?", hours: float = 24) -> str:
    now = time.time()
    resp = client.post("/v1/questions", json={
        "question": question,
        "resolution_criteria": "Yes if X occurs.",
        "deadline": now + hours * 3600,
        "resolution_deadline": now + hours * 3600 * 2,
    })
    assert resp.status_code == 200
    return resp.json()["question"]["id"]


def _submit(question_id: str, participant_id: str, probability: float,
            confidence: float = 0.8, pool: str = "open") -> dict:
    resp = client.post(f"/v1/questions/{question_id}/submit", json={
        "participant_id": participant_id,
        "probability": probability,
        "confidence": confidence,
        "pool": pool,
    })
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Instrument creation
# ---------------------------------------------------------------------------


class TestCreateInstrument:
    def test_create_spread(self):
        pid = _make_participant()
        q1 = _make_question("Will A happen?")
        q2 = _make_question("Will B happen?")

        _submit(q1, pid, 0.7)
        _submit(q2, pid, 0.3)

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "A vs B spread",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
            "thesis": "A and B are inversely correlated",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "spread"
        assert len(data["legs"]) == 2
        assert data["legs"][0]["side"] == "long"
        assert data["legs"][1]["side"] == "short"
        assert data["net_pnl"] == 0.0  # Just created, no movement yet

    def test_spread_requires_two_legs(self):
        pid = _make_participant()
        q1 = _make_question()

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Bad spread",
            "type": "spread",
            "legs": [{"question_id": q1, "side": "long", "weight": 1.0}],
        })
        assert resp.status_code == 422

    def test_create_basket(self):
        pid = _make_participant()
        q1 = _make_question("Will A?")
        q2 = _make_question("Will B?")
        q3 = _make_question("Will C?")

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Macro basket",
            "type": "basket",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 2.0},
                {"question_id": q2, "side": "long", "weight": 1.0},
                {"question_id": q3, "side": "short", "weight": 0.5},
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["type"] == "basket"
        assert len(resp.json()["legs"]) == 3

    def test_invalid_participant(self):
        q1 = _make_question()
        resp = client.post("/v1/book/instruments", json={
            "participant_id": "nonexistent",
            "name": "Bad",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q1, "side": "short", "weight": 1.0},
            ],
        })
        assert resp.status_code == 404

    def test_resolved_question_rejected(self):
        pid = _make_participant()
        q1 = _make_question()
        q2 = _make_question()
        _submit(q1, pid, 0.8)

        # Resolve q1
        client.post(f"/v1/questions/{q1}/resolve", json={"outcome": True})

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Stale spread",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
        })
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------


class TestMarkToMarket:
    def test_pnl_moves_with_market(self):
        pid = _make_participant()
        p2 = _make_participant("Trader B")
        q1 = _make_question("Will rates rise?")
        q2 = _make_question("Will rates fall?")

        # Initial submissions set market at 0.6 and 0.4
        _submit(q1, pid, 0.6)
        _submit(q2, pid, 0.4)

        # Create spread: long rates-rise, short rates-fall
        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Rate direction spread",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
        })
        inst_id = resp.json()["id"]

        # Second trader moves q1 up to ~0.8 (weighted avg of 0.6 and 0.95)
        _submit(q1, p2, 0.95)

        # Check MTM
        resp = client.get(f"/v1/book/instruments/{inst_id}")
        assert resp.status_code == 200
        data = resp.json()

        # Long leg should have positive P&L (probability went up)
        long_leg = [l for l in data["legs"] if l["side"] == "long"][0]
        assert long_leg["pnl"] > 0.0
        assert long_leg["current_probability"] > long_leg["entry_probability"]

    def test_close_locks_pnl(self):
        pid = _make_participant()
        q1 = _make_question()
        q2 = _make_question()
        _submit(q1, pid, 0.7)
        _submit(q2, pid, 0.3)

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Test spread",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
        })
        inst_id = resp.json()["id"]

        # Close it
        resp = client.post(f"/v1/book/instruments/{inst_id}/close")
        assert resp.status_code == 200
        assert resp.json()["closed"] is True
        assert resp.json()["closed_at"] is not None

        # Double-close fails
        resp = client.post(f"/v1/book/instruments/{inst_id}/close")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Book (positions)
# ---------------------------------------------------------------------------


class TestBook:
    def test_get_book(self):
        pid = _make_participant()
        q1 = _make_question("Q1")
        q2 = _make_question("Q2")
        q3 = _make_question("Q3")

        _submit(q1, pid, 0.6)
        _submit(q2, pid, 0.4)
        _submit(q3, pid, 0.5)

        # Create two instruments
        client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Spread 1",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
        })
        client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Basket 1",
            "type": "basket",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q3, "side": "long", "weight": 0.5},
            ],
        })

        resp = client.get(f"/v1/book/positions/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_instruments"] == 2
        assert data["summary"]["open_positions"] == 2

    def test_closed_excluded_by_default(self):
        pid = _make_participant()
        q1 = _make_question()
        q2 = _make_question()
        _submit(q1, pid, 0.5)
        _submit(q2, pid, 0.5)

        resp = client.post("/v1/book/instruments", json={
            "participant_id": pid,
            "name": "Closeable",
            "type": "spread",
            "legs": [
                {"question_id": q1, "side": "long", "weight": 1.0},
                {"question_id": q2, "side": "short", "weight": 1.0},
            ],
        })
        inst_id = resp.json()["id"]

        client.post(f"/v1/book/instruments/{inst_id}/close")

        # Default: closed excluded
        resp = client.get(f"/v1/book/positions/{pid}")
        assert resp.json()["summary"]["total_instruments"] == 0

        # With include_closed
        resp = client.get(f"/v1/book/positions/{pid}?include_closed=true")
        assert resp.json()["summary"]["total_instruments"] == 1


# ---------------------------------------------------------------------------
# Arbitrage scanner
# ---------------------------------------------------------------------------


class TestArbitrageScanner:
    def test_scan_returns_structure(self):
        resp = client.get("/v1/book/arbitrage/scan")
        assert resp.status_code == 200
        data = resp.json()
        assert "opportunities" in data
        assert "summary" in data
        assert data["summary"]["total"] == 0

    def test_exhaustive_violation(self):
        """Questions with same deadline summing to != 1.0 flags arb."""
        pid = _make_participant()
        now = time.time()
        deadline = now + 86400

        # Create 3 questions with identical deadline (simulating exhaustive outcomes)
        q1_resp = client.post("/v1/questions", json={
            "question": "Candidate A wins?",
            "resolution_criteria": "A wins election",
            "deadline": deadline,
            "resolution_deadline": deadline + 86400,
        })
        q2_resp = client.post("/v1/questions", json={
            "question": "Candidate B wins?",
            "resolution_criteria": "B wins election",
            "deadline": deadline,
            "resolution_deadline": deadline + 86400,
        })
        q3_resp = client.post("/v1/questions", json={
            "question": "Candidate C wins?",
            "resolution_criteria": "C wins election",
            "deadline": deadline,
            "resolution_deadline": deadline + 86400,
        })

        q1 = q1_resp.json()["question"]["id"]
        q2 = q2_resp.json()["question"]["id"]
        q3 = q3_resp.json()["question"]["id"]

        # Submit probabilities that sum to 1.3 (overpriced — arb exists)
        _submit(q1, pid, 0.6)
        _submit(q2, pid, 0.5)
        _submit(q3, pid, 0.2)
        # Sum = 1.3, should flag

        resp = client.get("/v1/book/arbitrage/scan")
        data = resp.json()
        assert data["summary"]["exhaustive_violations"] >= 1

        violation = [o for o in data["opportunities"] if o["type"] == "exhaustive_violation"][0]
        assert violation["implied_edge"] > 0.2  # 30pp off from 1.0


# ---------------------------------------------------------------------------
# Correlation map
# ---------------------------------------------------------------------------


class TestCorrelationMap:
    def test_correlation_map(self):
        p1 = _make_participant("Analyst A")
        p2 = _make_participant("Analyst B")
        p3 = _make_participant("Analyst C")
        q1 = _make_question("Will Fed hike?")
        q2 = _make_question("Will bond yields rise?")

        # All three submit to both questions
        _submit(q1, p1, 0.8)
        _submit(q1, p2, 0.7)
        _submit(q1, p3, 0.6)
        _submit(q2, p1, 0.75)
        _submit(q2, p2, 0.65)
        _submit(q2, p3, 0.55)

        resp = client.get("/v1/book/correlations?min_overlap=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

        pair = data["pairs"][0]
        assert pair["participant_overlap"] > 0
        assert pair["directional_agreement"] == 1.0  # All agree on direction

    def test_empty_when_no_overlap(self):
        p1 = _make_participant("Solo")
        q1 = _make_question()
        _submit(q1, p1, 0.5)

        resp = client.get("/v1/book/correlations")
        assert resp.json()["total"] == 0
