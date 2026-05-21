"""API tests for ``DELETE /model/cache`` (v1.x #5 Re-download).

Same shape as ``test_healthz_model.py``: mount only the ``model``
router, point ``HUGGINGFACE_HUB_CACHE`` at a synthetic ``tmp_path``,
and verify the route round-trips against
``rubick_backend.model_status.delete_model_cache``. No MLX, no
LanceDB, no real ``~/.cache/huggingface/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rubick_backend import settings
from rubick_backend.api.model import router

REPO_MAIN = settings.MAIN_MODEL_REPO
SHA = "feedface0000111122223333444455556666777788889999aaaabbbbccccdddd"


def _make_complete_cache(cache_root: Path, repo_id: str, sha: str = SHA) -> Path:
    """Plant a minimal post-``snapshot_download`` layout for ``repo_id``.

    Same shape as ``test_model_status._build_complete_cache`` but inlined
    here so this file stays self-contained (and so importing test
    modules across packages doesn't kick in).
    """
    base = cache_root / ("models--" + repo_id.replace("/", "--"))
    (base / "blobs").mkdir(parents=True)
    (base / "snapshots" / sha).mkdir(parents=True)
    (base / "refs").mkdir(parents=True)
    blob = base / "blobs" / "deadbeef00"
    blob.write_bytes(b"x" * 4096)
    (base / "snapshots" / sha / "model.safetensors").symlink_to(blob)
    (base / "refs" / "main").write_text(sha)
    return base


@pytest.fixture()
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty HF cache rooted in ``tmp_path`` (override via env)."""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    """Bare FastAPI with only the ``/model`` router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_delete_cache_wipes_complete_cache(
    client: TestClient, fake_cache: Path
) -> None:
    base = _make_complete_cache(fake_cache, REPO_MAIN)
    assert base.is_dir()

    r = client.delete("/model/cache", params={"id": "embedding"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "embedding"
    assert body["repo"] == REPO_MAIN
    assert body["was_present"] is True
    assert body["deleted_bytes"] >= 4096
    assert body["path"] == str(base)
    assert not base.exists()


def test_delete_cache_idempotent_when_absent(
    client: TestClient, fake_cache: Path
) -> None:
    """A second click while the cache is already gone must not 404 —
    "make sure it's gone" semantics drive a calmer UX than "this op
    only works once"."""
    r = client.delete("/model/cache", params={"id": "embedding"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["was_present"] is False
    assert body["deleted_bytes"] == 0
    assert body["path"] is None


def test_delete_cache_404_for_unknown_id(
    client: TestClient, fake_cache: Path
) -> None:
    r = client.delete("/model/cache", params={"id": "transcript"})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "unknown model id" in detail
    # Echo the known ids in the error so the user sees the valid set.
    assert "embedding" in detail


def test_delete_cache_requires_id_query_param(
    client: TestClient, fake_cache: Path
) -> None:
    """``?id=`` is mandatory — no implicit "delete every model"
    default. Validates as a 422 from FastAPI (its standard "missing
    required query param" status), not 400."""
    r = client.delete("/model/cache")
    assert r.status_code == 422


def test_delete_cache_leaves_other_models_alone(
    client: TestClient, fake_cache: Path
) -> None:
    """A second model on the same cache root must survive the delete.
    Stand-in for an imaginary future second model id — proves the
    rmtree is scoped to the per-repo subdir.
    """
    target_base = _make_complete_cache(fake_cache, REPO_MAIN)
    bystander_base = _make_complete_cache(fake_cache, "someorg/other-model")

    r = client.delete("/model/cache", params={"id": "embedding"})
    assert r.status_code == 200, r.text
    assert not target_base.exists()
    assert bystander_base.is_dir()
