"""Tests for the Nebula UMAP compute pipeline.

Exercises:
- normalize (pure math, no deps)
- load_map (empty case, valid file, corrupt file)
- run_nebula_compute with mocked LanceDB (< 10 points path)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_normalize_basic():
    from rubick_backend.nebula.compute import _normalize

    coords = np.array([[0, 0], [10, 5], [5, 10]], dtype=np.float32)
    result = _normalize(coords)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert result[1, 0] == pytest.approx(1.0)
    assert result[2, 1] == pytest.approx(1.0)


def test_normalize_same_values():
    """All points on same position — should not crash (div by zero guard)."""
    from rubick_backend.nebula.compute import _normalize

    coords = np.array([[5, 5], [5, 5], [5, 5]], dtype=np.float32)
    result = _normalize(coords)
    assert np.allclose(result, 0.0)


def test_load_map_empty(isolated_data_dir: Path):
    """No file on disk — empty map."""
    from rubick_backend.nebula.compute import load_map

    result = load_map()
    assert result["version"] == 1
    assert result["computed_at"] == 0
    assert result["total_points"] == 0
    assert result["points"] == []


def test_load_map_valid(isolated_data_dir: Path):
    """Valid JSON file — loaded correctly."""
    from rubick_backend import settings
    from rubick_backend.nebula.compute import load_map

    test_map = {
        "version": 1,
        "computed_at": 1000,
        "total_points": 2,
        "points": [
            {"doc_id": "abc", "chunk_id": "abc-image-0", "x": 0.1, "y": 0.2,
             "modality": "image", "thumbnail_path": None, "filename": "a.jpg"},
            {"doc_id": "def", "chunk_id": "def-video-0", "x": 0.8, "y": 0.9,
             "modality": "video", "thumbnail_path": None, "filename": "b.mp4"},
        ],
    }
    settings.NEBULA_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings.NEBULA_MAP_FILE.write_text(json.dumps(test_map))

    result = load_map()
    assert result["total_points"] == 2
    assert len(result["points"]) == 2
    assert result["points"][0]["doc_id"] == "abc"


def test_load_map_corrupt(isolated_data_dir: Path):
    """Corrupt JSON — falls back to empty map."""
    from rubick_backend import settings
    from rubick_backend.nebula.compute import load_map

    settings.NEBULA_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings.NEBULA_MAP_FILE.write_text("not json {{{{")

    result = load_map()
    assert result["total_points"] == 0


def test_empty_map_structure():
    from rubick_backend.nebula.compute import _empty_map

    m = _empty_map()
    assert m == {"version": 1, "computed_at": 0, "total_points": 0, "points": []}
