"""API tests for ``GET /healthz`` + ``GET /healthz/model``.

Runs in the **fast** suite (no ``slow`` mark): we mount the
``healthz`` router on a bare FastAPI app, point ``HUGGINGFACE_HUB_CACHE``
at a synthetic ``tmp_path``, and never touch MLX, the JobQueue
lifespan, or the real ``~/.cache/huggingface/`` tree. The endpoint
itself only stat()s directories, so this faithfully covers what
production sees.

Also a deliberate guard against a regression I almost introduced when
splitting ``healthz`` out of ``main.py``: the route used to be inline
in ``main.py``; the test ensures both paths are reachable through the
new ``api/healthz.py`` router and that the JSON shape matches what the
upcoming Settings → Model + Onboarding views consume.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rubick_backend import settings
from rubick_backend.api.healthz import router

REPO_MAIN = "jinaai/jina-embeddings-v5-omni-nano-retrieval-mlx"
SHA = "deadbeef0000111122223333444455556666777788889999aaaabbbbccccdddd"


@pytest.fixture()
def fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty HF cache rooted in ``tmp_path``. Tests opt into populating it."""
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    """Bare FastAPI mounting only the ``healthz`` router.

    Bypasses ``main.py``'s lifespan (which spins up the LanceDB-backed
    JobQueue) — out of scope for these tests, and would slow the suite
    down for no benefit. Production imports ``rubick_backend.main.app``
    which mounts the same router with the same paths.
    """
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def force_embed_unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``embed.is_loaded()`` to ``False`` so we can assert the
    "model not yet hydrated this process" branch without actually
    loading the 1.8 GB embedder.
    """
    import rubick_backend.embed as embed_mod

    monkeypatch.setattr(embed_mod, "is_loaded", lambda: False)
    # The healthz router did ``from ..embed import is_loaded as embed_is_loaded``
    # which captured the original symbol at import time — patch the binding
    # the route actually calls.
    import rubick_backend.api.healthz as healthz_mod

    monkeypatch.setattr(healthz_mod, "embed_is_loaded", lambda: False)


def _build_complete_layout(cache_root: Path, repo_id: str) -> Path:
    """Synthesize a post-snapshot_download layout for ``repo_id``."""
    base = cache_root / ("models--" + repo_id.replace("/", "--"))
    blobs = base / "blobs"
    snapshots = base / "snapshots" / SHA
    refs = base / "refs"
    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    refs.mkdir(parents=True)
    (blobs / "weights").write_bytes(b"x" * 2048)
    (blobs / "config").write_bytes(b"y" * 128)
    (snapshots / "model.safetensors").symlink_to(blobs / "weights")
    (snapshots / "config.json").symlink_to(blobs / "config")
    (refs / "main").write_text(SHA)
    return base


# === /healthz ===============================================================


def test_healthz_returns_status_ok(client: TestClient) -> None:
    """Liveness probe must always be cheap + correct."""
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # ``__version__`` is a simple semver string today; just make sure
    # we're surfacing *something* truthy so the Swift parent has a
    # breadcrumb for boot logs.
    assert body["version"]


# === /healthz/model =========================================================


def test_healthz_model_absent_when_cache_is_empty(
    client: TestClient,
    fake_cache: Path,  # noqa: ARG001
    force_embed_unloaded: None,  # noqa: ARG001
) -> None:
    """Fresh-machine boot — model not downloaded yet."""
    r = client.get("/healthz/model")
    assert r.status_code == 200
    body = r.json()

    assert "models" in body
    models = {m["id"]: m for m in body["models"]}
    assert set(models) == {"embedding"}

    embed = models["embedding"]
    assert embed["repo"] == REPO_MAIN
    assert embed["download_status"] == "absent"
    assert embed["cache_path"] is None
    assert embed["cache_bytes"] == 0
    assert embed["loaded_in_memory"] is False
    assert embed["declared_bytes"] == settings.MAIN_MODEL_DECLARED_BYTES
    assert embed["purpose"]  # truthy


def test_healthz_model_complete_when_main_repo_downloaded(
    client: TestClient,
    fake_cache: Path,
    force_embed_unloaded: None,  # noqa: ARG001
) -> None:
    """Embedding model finished snapshot_download but isn't in memory yet."""
    base = _build_complete_layout(fake_cache, REPO_MAIN)

    r = client.get("/healthz/model")
    assert r.status_code == 200
    models = {m["id"]: m for m in r.json()["models"]}

    embed = models["embedding"]
    assert embed["download_status"] == "complete"
    assert embed["cache_path"] == str(base)
    # 2048 + 128 (real blobs) + len(SHA) (refs/main text). The
    # snapshots/ symlinks contribute 0 because ``directory_size_bytes``
    # mirrors ``du`` and skips symlinks (otherwise we'd double-count).
    assert embed["cache_bytes"] == 2048 + 128 + len(SHA)
    assert embed["loaded_in_memory"] is False  # downloaded != loaded


def test_healthz_model_partial_when_only_incomplete_blob(
    client: TestClient,
    fake_cache: Path,
    force_embed_unloaded: None,  # noqa: ARG001
) -> None:
    """A single ``.incomplete`` blob means the last download was
    interrupted — UI should show "resuming" rather than "ready".
    """
    base = fake_cache / ("models--" + REPO_MAIN.replace("/", "--"))
    (base / "blobs").mkdir(parents=True)
    (base / "blobs" / "halfdone.incomplete").write_bytes(b"a" * 64)
    (base / "refs").mkdir()
    (base / "snapshots").mkdir()

    r = client.get("/healthz/model")
    embed = next(m for m in r.json()["models"] if m["id"] == "embedding")
    assert embed["download_status"] == "partial"
    # Cache path *is* surfaced even mid-download so the UI can show
    # "downloading to ~/.cache/...".
    assert embed["cache_path"] == str(base)
    assert embed["cache_bytes"] == 64


def test_healthz_model_reflects_loaded_in_memory_flag(
    client: TestClient,
    fake_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the embedder is actually hydrated, the API says so.

    We don't actually load MLX (slow + 1.8 GB); we monkey-patch the
    accessor the route calls.
    """
    _build_complete_layout(fake_cache, REPO_MAIN)
    import rubick_backend.api.healthz as healthz_mod

    monkeypatch.setattr(healthz_mod, "embed_is_loaded", lambda: True)

    r = client.get("/healthz/model")
    embed = next(m for m in r.json()["models"] if m["id"] == "embedding")
    assert embed["download_status"] == "complete"
    assert embed["loaded_in_memory"] is True


def test_healthz_model_payload_is_stable_for_consumers(
    client: TestClient,
    fake_cache: Path,  # noqa: ARG001
    force_embed_unloaded: None,  # noqa: ARG001
) -> None:
    """Lock down the exact key set the Swift Settings → Model + Onboarding
    views will hard-code against. Adding a new field is fine; *renaming*
    or *removing* one breaks the consumers, so this test fails first.
    """
    expected_per_model = {
        "id",
        "repo",
        "purpose",
        "declared_bytes",
        "cache_path",
        "cache_bytes",
        "download_status",
        "loaded_in_memory",
    }

    body = client.get("/healthz/model").json()
    assert list(body) == ["models"]  # top-level shape is just one key
    for entry in body["models"]:
        assert set(entry) == expected_per_model

    ids = [m["id"] for m in body["models"]]
    assert ids == ["embedding"]
