"""Integration tests for the dual-pool open submission system."""
import time

import pytest
from httpx import ASGITransport, AsyncClient

from augur.server import app
from augur.submissions import _participants, _questions, _submissions


@pytest.fixture(autouse=True)
def _clear_stores():
    """Reset in-memory stores between tests."""
    _questions.clear()
    _submissions.clear()
    _participants.clear()
    yield
    _questions.clear()
    _submissions.clear()
    _participants.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register_participant(client, name="Alice", ptype="human"):
    resp = await client.post("/v1/participants", json={"display_name": name, "type": ptype})
    assert resp.status_code == 200
    return resp.json()


async def _create_question(client, question="Will it rain tomorrow?"):
    now = time.time()
    resp = await client.post("/v1/questions", json={
        "question": question,
        "resolution_criteria": "Rain observed at station X",
        "deadline": now + 86400,
        "resolution_deadline": now + 172800,
    })
    assert resp.status_code == 200
    return resp.json()


class TestParticipants:
    async def test_register(self, client):
        p = await _register_participant(client, "Bob", "agent")
        assert p["display_name"] == "Bob"
        assert p["type"] == "agent"
        assert p["reputation_weight"] == 1.0

    async def test_get_participant(self, client):
        p = await _register_participant(client)
        resp = await client.get(f"/v1/participants/{p['id']}")
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Alice"

    async def test_invalid_type(self, client):
        resp = await client.post("/v1/participants", json={"display_name": "X", "type": "robot"})
        assert resp.status_code == 422

    async def test_not_found(self, client):
        resp = await client.get("/v1/participants/nonexistent")
        assert resp.status_code == 404


class TestQuestions:
    async def test_create_has_dual_pools(self, client):
        data = await _create_question(client)
        assert data["pools"] == ["open", "dark"]
        assert data["question"]["resolved"] is False

    async def test_list(self, client):
        await _create_question(client, "Q1")
        await _create_question(client, "Q2")
        resp = await client.get("/v1/questions")
        assert resp.json()["total"] == 2

    async def test_get_shows_both_pool_aggregates(self, client):
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.get(f"/v1/questions/{qid}")
        body = resp.json()
        assert "open" in body["pools"]
        assert "dark" in body["pools"]
        assert body["pools"]["open"]["aggregate_probability"] == 0.5
        # Dark is sealed (deadline hasn't passed)
        assert body["pools"]["dark"]["sealed"] is True
        assert body["pools"]["dark"]["aggregate_probability"] is None

    async def test_not_found(self, client):
        resp = await client.get("/v1/questions/nonexistent")
        assert resp.status_code == 404


class TestSubmissions:
    async def test_submit_open(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"],
            "probability": 0.7,
            "confidence": 0.8,
            "pool": "open",
        })
        assert resp.status_code == 200
        sub = resp.json()
        assert sub["pool"] == "open"
        assert sub["consensus_at_time"] == 0.5  # First open submission

    async def test_submit_dark_redacts_consensus(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"],
            "probability": 0.7,
            "confidence": 0.8,
            "pool": "dark",
        })
        assert resp.status_code == 200
        assert resp.json()["consensus_at_time"] == -1.0

    async def test_submit_defaults_to_open(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"],
            "probability": 0.5,
            "confidence": 0.5,
        })
        assert resp.json()["pool"] == "open"

    async def test_invalid_pool(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"],
            "probability": 0.5,
            "confidence": 0.5,
            "pool": "invalid",
        })
        assert resp.status_code == 422

    async def test_submit_both_pools(self, client):
        """Same participant can submit to both pools on the same question."""
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        r1 = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.8, "confidence": 0.9, "pool": "open",
        })
        r2 = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.6, "confidence": 0.7, "pool": "dark",
        })
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["pool"] == "open"
        assert r2.json()["pool"] == "dark"

    async def test_unregistered_participant(self, client):
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": "nobody", "probability": 0.5, "confidence": 0.5,
        })
        assert resp.status_code == 404

    async def test_submit_after_deadline(self, client):
        p = await _register_participant(client)
        now = time.time()
        resp = await client.post("/v1/questions", json={
            "question": "Past deadline?",
            "resolution_criteria": "test",
            "deadline": now - 1,
            "resolution_deadline": now + 86400,
        })
        qid = resp.json()["question"]["id"]
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.5, "confidence": 0.5,
        })
        assert resp.status_code == 409

    async def test_list_open_submissions_visible(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.6, "confidence": 0.7, "pool": "open",
        })
        resp = await client.get(f"/v1/questions/{qid}/submissions", params={"pool": "open"})
        assert resp.json()["count"] == 1

    async def test_list_dark_submissions_sealed(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.6, "confidence": 0.7, "pool": "dark",
        })
        resp = await client.get(f"/v1/questions/{qid}/submissions", params={"pool": "dark"})
        assert resp.status_code == 403

    async def test_list_all_excludes_dark_while_sealed(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.6, "confidence": 0.7, "pool": "open",
        })
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.8, "confidence": 0.9, "pool": "dark",
        })
        resp = await client.get(f"/v1/questions/{qid}/submissions")
        # Should only show the open submission
        assert resp.json()["count"] == 1
        assert resp.json()["submissions"][0]["pool"] == "open"


class TestResolution:
    async def test_resolve_scores_both_pools(self, client):
        alice = await _register_participant(client, "Alice", "human")
        bob = await _register_participant(client, "Bob", "agent")
        data = await _create_question(client, "Will BTC hit 100k?")
        qid = data["question"]["id"]

        # Alice in open pool
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.8, "confidence": 0.9, "pool": "open",
        })
        # Bob in dark pool
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": bob["id"], "probability": 0.9, "confidence": 0.95, "pool": "dark",
        })

        resp = await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})
        assert resp.status_code == 200
        result = resp.json()
        assert result["scored_submissions"] == 2
        assert result["pool_breakdown"]["open"] == 1
        assert result["pool_breakdown"]["dark"] == 1
        assert "final_spread" in result

    async def test_dark_high_confidence_vs_open(self, client):
        """Dark pool multiplier scales with confidence; open scales with timing."""
        alice = await _register_participant(client, "Alice")
        bob = await _register_participant(client, "Bob")
        data = await _create_question(client, "Test")
        qid = data["question"]["id"]

        # Alice: dark pool, high confidence → multiplier 1.9
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.8, "confidence": 0.9, "pool": "dark",
        })
        # Bob: open pool, submitted immediately → gets high early bonus (~2.0)
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": bob["id"], "probability": 0.8, "confidence": 0.9, "pool": "open",
        })

        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})

        subs_resp = await client.get(f"/v1/questions/{qid}/submissions")
        subs = subs_resp.json()["submissions"]
        alice_sub = next(s for s in subs if s["participant_id"] == alice["id"])
        bob_sub = next(s for s in subs if s["participant_id"] == bob["id"])

        # Dark multiplier: 1.0 + 0.9 confidence = 1.9
        assert alice_sub["pool_multiplier"] == 1.9
        # Open early multiplier: near 2.0 (submitted almost at creation)
        assert bob_sub["pool_multiplier"] > 1.9
        # Both have multiplied scores
        assert alice_sub["combined_score"] > 0
        assert bob_sub["combined_score"] > 0

    async def test_dark_unseals_after_resolution(self, client):
        p = await _register_participant(client)
        data = await _create_question(client, "Sealed test")
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.7, "confidence": 0.8, "pool": "dark",
        })
        # Sealed before resolution
        resp = await client.get(f"/v1/questions/{qid}/submissions", params={"pool": "dark"})
        assert resp.status_code == 403

        # Resolve
        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})

        # Unsealed after resolution
        resp = await client.get(f"/v1/questions/{qid}/submissions", params={"pool": "dark"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    async def test_resolve_already_resolved(self, client):
        data = await _create_question(client)
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})
        resp = await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": False})
        assert resp.status_code == 409

    async def test_submit_after_resolution(self, client):
        p = await _register_participant(client)
        data = await _create_question(client)
        qid = data["question"]["id"]
        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})
        resp = await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": p["id"], "probability": 0.5, "confidence": 0.5,
        })
        assert resp.status_code == 409


class TestSpread:
    async def test_spread_computed_after_resolution(self, client):
        alice = await _register_participant(client, "Alice")
        bob = await _register_participant(client, "Bob")
        data = await _create_question(client, "Spread test")
        qid = data["question"]["id"]

        # Divergent predictions: open says 0.8, dark says 0.3
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.8, "confidence": 0.9, "pool": "open",
        })
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": bob["id"], "probability": 0.3, "confidence": 0.9, "pool": "dark",
        })

        resp = await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})
        result = resp.json()
        # Spread should be |0.8 - 0.3| = 0.5
        assert result["final_spread"] == 0.5

    async def test_spread_hidden_while_sealed(self, client):
        data = await _create_question(client)
        qid = data["question"]["id"]
        resp = await client.get(f"/v1/questions/{qid}")
        assert resp.json()["spread"] is None  # Dark is sealed


class TestLeaderboard:
    async def test_empty(self, client):
        resp = await client.get("/v1/leaderboard")
        assert resp.json()["total_participants"] == 0

    async def test_ranked_after_resolution(self, client):
        alice = await _register_participant(client, "Alice")
        bob = await _register_participant(client, "Bob")
        data = await _create_question(client, "Test question")
        qid = data["question"]["id"]

        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.9, "confidence": 0.9, "pool": "open",
        })
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": bob["id"], "probability": 0.3, "confidence": 0.8, "pool": "open",
        })

        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})

        resp = await client.get("/v1/leaderboard")
        lb = resp.json()["leaderboard"]
        assert len(lb) == 2
        assert lb[0]["display_name"] == "Alice"
        assert lb[1]["display_name"] == "Bob"


class TestDualPoolArbitrage:
    async def test_same_participant_different_pools_scored_independently(self, client):
        """A participant who hedges between pools gets each scored with its own multiplier."""
        alice = await _register_participant(client, "Alice")
        data = await _create_question(client, "Hedge test")
        qid = data["question"]["id"]

        # Alice hedges: 0.8 in open, 0.6 in dark
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.8, "confidence": 0.9, "pool": "open",
        })
        await client.post(f"/v1/questions/{qid}/submit", json={
            "participant_id": alice["id"], "probability": 0.6, "confidence": 0.7, "pool": "dark",
        })

        await client.post(f"/v1/questions/{qid}/resolve", json={"outcome": True})

        subs_resp = await client.get(f"/v1/questions/{qid}/submissions")
        subs = subs_resp.json()["submissions"]
        open_sub = next(s for s in subs if s["pool"] == "open")
        dark_sub = next(s for s in subs if s["pool"] == "dark")

        # Different multipliers
        assert open_sub["pool_multiplier"] != dark_sub["pool_multiplier"]
        # Open (0.8) has better Brier than dark (0.6) on yes outcome
        assert open_sub["brier_score"] < dark_sub["brier_score"]
        # But dark has conviction multiplier
        assert dark_sub["pool_multiplier"] == 1.7  # 1.0 + 0.7
