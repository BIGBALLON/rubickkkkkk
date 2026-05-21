"""HTTP-level tests for the Nebula API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(isolated_data_dir: Path):
    """TestClient against the app with isolated data dir."""
    from rubick_backend.main import app

    return TestClient(app)


def test_get_map_empty(client: TestClient, isolated_data_dir: Path):
    """No map file → returns empty map structure."""
    resp = client.get("/nebula/map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == 1
    assert data["computed_at"] == 0
    assert data["total_points"] == 0
    assert data["points"] == []


def test_get_map_with_data(client: TestClient, isolated_data_dir: Path):
    """Persisted map file → served correctly."""
    from rubick_backend import settings

    test_map = {
        "version": 1,
        "computed_at": 9999,
        "total_points": 1,
        "points": [{"doc_id": "x", "chunk_id": "x-image-0",
                    "x": 0.5, "y": 0.5, "modality": "image",
                    "thumbnail_path": None, "filename": "test.jpg"}],
    }
    settings.NEBULA_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings.NEBULA_MAP_FILE.write_text(json.dumps(test_map))

    resp = client.get("/nebula/map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_points"] == 1
    assert data["points"][0]["doc_id"] == "x"


def test_get_status_idle(client: TestClient, isolated_data_dir: Path):
    """Default state → idle, not stale (no image/video chunks)."""
    resp = client.get("/nebula/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "idle"
    assert data["progress"] == 0.0


def test_recompute_returns_immediately(client: TestClient, isolated_data_dir: Path):
    """POST /nebula/recompute returns started status."""
    with patch("rubick_backend.nebula.compute.run_nebula_compute"):
        resp = client.post("/nebula/recompute")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("started", "already_computing")
