import pytest
from fastapi.testclient import TestClient
from sonapse_origin.observatory.server import app


def test_observatory_server_routes():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "SONAPSE-ORIGIN" in response.text

    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    data = state_resp.json()
    assert "step" in data
    assert "H" in data
    assert "A" in data

    interv_resp = client.post("/api/intervention", json={"replenish_energy": 0.5})
    assert interv_resp.status_code == 200
    assert interv_resp.json()["status"] == "Intervention queued"


def test_observatory_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as websocket:
        data = websocket.receive_json()
        assert "step" in data
        assert "homeostasis" in data
        assert "affect" in data
        assert "graph" in data
