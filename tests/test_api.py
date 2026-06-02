"""Offline tests for the FastAPI REST API.

These tests exercise the HTTP layer without loading the real sentence-transformers
model. They do this by injecting a lightweight FakeMapper into the application
state directly, so the request handlers run against a deterministic stand in.

A subtle but important detail: a plain TestClient(app) does not trigger the
application lifespan, so the real model is never loaded here. We populate
api._state["mapper"] ourselves to simulate a started up service.
"""
from __future__ import annotations

from typing import List, Tuple

import pytest
from fastapi.testclient import TestClient

import api
from nlp_mapper import RiskEntry


class FakeMapper:
    """A deterministic stand in for RiskMapper used to test the API offline."""

    def find_top_risks(
        self, description: str, n: int = 3
    ) -> List[Tuple[RiskEntry, float]]:
        """Return a fixed ranked list, truncated to n, ignoring the text.

        The scores descend so the ordering is unambiguous, which is all the API
        layer needs in order to assign ranks and serialise a response.
        """
        catalogue = [
            (RiskEntry("Privacy & Security", "Compromise of privacy", "Data exposure."), 0.62),
            (RiskEntry("Discrimination & Toxicity", "Unfair discrimination", "Biased outcomes."), 0.55),
            (RiskEntry("Misinformation", "False or misleading information", "Untrue claims."), 0.48),
        ]
        return catalogue[:n]


@pytest.fixture
def client():
    """Provide a TestClient with a FakeMapper pre loaded into app state.

    Using TestClient(app) without a context manager means the lifespan startup,
    which would load the real model, never runs. We inject the fake instead and
    clean it up afterwards so tests do not leak state into one another.
    """
    api._state["mapper"] = FakeMapper()
    test_client = TestClient(api.app)
    yield test_client
    api._state.clear()


def test_health_reports_model_loaded(client):
    """The health endpoint should report ok and a loaded model."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_analyze_returns_ranked_matches(client):
    """A valid request should return ranked matches with scores."""
    response = client.post("/analyze", json={"description": "A health prediction app."})
    assert response.status_code == 200
    body = response.json()
    assert body["description"] == "A health prediction app."
    assert body["count"] == 3
    assert len(body["matches"]) == 3
    first = body["matches"][0]
    assert first["rank"] == 1
    assert first["domain"] == "Privacy & Security"
    assert first["subdomain"] == "Compromise of privacy"
    assert first["score"] == pytest.approx(0.62)
    # Ranks must be a clean ascending sequence starting at 1.
    assert [m["rank"] for m in body["matches"]] == [1, 2, 3]


def test_analyze_respects_top_n(client):
    """The top_n field should limit how many matches are returned."""
    response = client.post(
        "/analyze", json={"description": "Some AI system.", "top_n": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert len(body["matches"]) == 1


def test_analyze_rejects_blank_description(client):
    """A whitespace only description should be rejected with a 400."""
    response = client.post("/analyze", json={"description": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_rejects_missing_description(client):
    """Omitting the required description should fail Pydantic validation (422)."""
    response = client.post("/analyze", json={"top_n": 3})
    assert response.status_code == 422


def test_analyze_rejects_out_of_range_top_n(client):
    """A top_n above the allowed maximum should fail validation (422)."""
    response = client.post(
        "/analyze", json={"description": "An AI tool.", "top_n": 999}
    )
    assert response.status_code == 422


def test_analyze_returns_503_when_model_not_loaded():
    """If the model is not loaded, analyze should report the service as unready."""
    # Build a client with empty state so get_mapper raises a 503.
    api._state.clear()
    test_client = TestClient(api.app)
    response = test_client.post("/analyze", json={"description": "An AI tool."})
    assert response.status_code == 503
