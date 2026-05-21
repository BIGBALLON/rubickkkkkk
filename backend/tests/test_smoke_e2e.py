"""End-to-end smokes (slow; gated on ``RUBICK_RUN_SLOW=1``).

This is the canonical wiring smoke — it verifies that the whole
chain (ingest dispatcher → MLX embedder → LanceDB → FastAPI ``/search``)
returns plausible cross-modal JSON results from a tiny text + image
corpus. Per-modality row-shape coverage lives in
``test_text_pipeline.py`` / ``test_image_pipeline.py`` /
``test_video_pipeline.py``; we only assert here what *integration*
adds on top (modality filter end-to-end, dedup re-ingest, the
``POST /index/job`` route).

Run::

    RUBICK_RUN_SLOW=1 pytest tests/test_smoke_e2e.py -v -s

The ``isolated_data_dir`` fixture (see ``conftest.py``) routes all
on-disk state into a tmp dir so we never touch the user's real
``~/Library/Application Support/Rubick``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


# A trio of distinct-topic notes so we can sanity-check that the right
# document floats to the top for the right query.
NOTES: dict[str, str] = {
    "mars-exploration.md": (
        "# Mars Exploration\n\n"
        "Notes on the red planet. The Perseverance rover landed in Jezero crater "
        "in February 2021 to search for ancient microbial life. Helicopter Ingenuity "
        "demonstrated the first powered flight on another world.\n\n"
        "## Atmosphere\n\n"
        "Mars has a thin atmosphere of carbon dioxide. Surface pressure is about "
        "0.6 percent of Earth's. Dust storms can engulf the entire planet."
    ),
    "espresso-recipes.md": (
        "# Espresso Recipes\n\n"
        "Dialing in a fresh bag of coffee beans takes practice. Start with an "
        "18 gram dose, aim for a 36 gram yield in 28 seconds. Adjust grind size "
        "first, then dose, then temperature.\n\n"
        "## Milk drinks\n\n"
        "Steam whole milk to about 60 degrees Celsius for cortado, latte, and "
        "cappuccino. Stretch the milk briefly, then submerge the wand to texture."
    ),
    "rust-async-notes.md": (
        "# Rust Async Notes\n\n"
        "tokio is the de-facto async runtime in Rust. Tasks are M:N scheduled "
        "onto a thread pool. The .await keyword yields control to the runtime "
        "when a future returns Pending.\n\n"
        "## Pinning\n\n"
        "Self-referential futures must be pinned in memory because moving them "
        "would invalidate internal pointers. Pin<&mut T> guarantees the pointee "
        "won't move."
    ),
}


@pytest.fixture()
def notes_folder(tmp_path: Path) -> Path:
    """Materialize the synthetic notes corpus inside ``tmp_path``."""
    d = tmp_path / "notes"
    d.mkdir()
    for name, body in NOTES.items():
        (d / name).write_text(body, encoding="utf-8")
    return d


def _reload_modules() -> None:
    """Reload backend modules so they pick up the patched ``RUBICK_DATA_DIR``.

    Order matters: settings → store.schema → store → embed.loader (no-op for
    paths) → main. We don't reload the embed loader because the singleton
    intentionally lives across test boundaries (a fresh load is too slow).
    """
    import rubick_backend.settings  # noqa: F401
    import rubick_backend.store  # noqa: F401
    import rubick_backend.store.schema  # noqa: F401

    for mod_name in (
        "rubick_backend.settings",
        "rubick_backend.store.schema",
        "rubick_backend.store",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


def test_ingest_and_search_end_to_end(
    isolated_data_dir: Path,
    notes_folder: Path,
) -> None:
    """Ingest three topical notes, then query each topic and confirm the
    matching document ranks #1 with positive cosine similarity.

    Also asserts the hybrid response surfaces ``score_rrf`` /
    ``score_vector`` / ``hit_count`` for every hit — Swift ignores
    extra fields, but server-side they must be present.
    """
    _reload_modules()
    from rubick_backend.ingest import ingest_path

    stats = ingest_path(notes_folder)
    assert stats["files"] == 3
    assert stats["chunks"] >= 3
    assert stats["skipped"] == 0

    from fastapi.testclient import TestClient

    from rubick_backend.main import app

    client = TestClient(app)

    cases = [
        ("rover on the red planet", "mars-exploration"),
        ("espresso milk drink steam wand", "espresso-recipes"),
        ("tokio futures pinning runtime", "rust-async-notes"),
    ]
    for query, expected_stem in cases:
        resp = client.get("/search", params={"q": query, "limit": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["query"] == query
        assert body["count"] >= 1
        top = body["results"][0]
        assert top["filename"] == expected_stem, (
            f"query={query!r} expected top filename={expected_stem!r}, "
            f"got {top['filename']!r} (sim={top['similarity']:.3f})"
        )
        assert top["similarity"] > 0.1
        assert top["modality"] == "text"
        assert top["doc_id"] and len(top["doc_id"]) == 16
        # took_ms should be populated and roughly sane
        assert body["took_ms"]["embed"] > 0
        assert body["took_ms"]["search"] >= 0
        # Hybrid retrieval fields (optional on older backends)
        assert top["score_rrf"] > 0
        assert top["score_vector"] is not None
        assert top["hit_count"] >= 1
        # RRF score must be monotonically non-increasing across results
        scores = [r["score_rrf"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True), f"results not sorted by score_rrf: {scores}"


def test_filename_keyword_hits_via_bm25(
    isolated_data_dir: Path,
    notes_folder: Path,
) -> None:
    """The BM25 leg should let a user find a file by its filename even
    when the vector path alone might not be strong enough.

    We query ``espresso-recipes`` (the literal filename stem) — a
    pure vector query on that exact token wouldn't necessarily place
    that doc on top, but BM25's filename index gives us a perfect
    match.
    """
    _reload_modules()
    from fastapi.testclient import TestClient

    from rubick_backend.ingest import ingest_path
    from rubick_backend.main import app

    ingest_path(notes_folder)
    client = TestClient(app)

    resp = client.get("/search", params={"q": "espresso-recipes", "limit": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    top = body["results"][0]
    assert top["filename"] == "espresso-recipes"
    # BM25 must have contributed a non-null score for this fixture.
    assert top["score_bm25"] is not None
    assert top["score_bm25"] > 0


def test_search_modality_filter(
    isolated_data_dir: Path,
    notes_folder: Path,
) -> None:
    """``modality=text`` must return only text rows; ``modality=image`` must
    return zero hits (we haven't ingested any images).
    """
    _reload_modules()
    from rubick_backend.ingest import ingest_path

    ingest_path(notes_folder)

    from fastapi.testclient import TestClient

    from rubick_backend.main import app

    client = TestClient(app)

    r_text = client.get("/search", params={"q": "rover", "modality": "text"})
    assert r_text.status_code == 200
    assert all(h["modality"] == "text" for h in r_text.json()["results"])

    r_img = client.get("/search", params={"q": "rover", "modality": "image"})
    assert r_img.status_code == 200
    assert r_img.json()["count"] == 0


def test_healthz_endpoint(isolated_data_dir: Path) -> None:
    """``/healthz`` should always return ``{"status":"ok", "version": ...}``."""
    _reload_modules()
    from fastapi.testclient import TestClient

    from rubick_backend.main import app

    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"]


def test_reingest_same_files_is_idempotent(
    isolated_data_dir: Path,
    notes_folder: Path,
) -> None:
    """Content dedup: ingesting the same folder twice must not
    duplicate rows.

    FSEvents on macOS happily fires on metadata-only changes (atime
    bumps, Spotlight-touches, even readlink), so re-ingest of the
    *same content* will happen routinely once FSEvents is wired up.
    The pipeline must dedupe on ``doc_id`` (= sha256 of bytes) and
    skip the embed pass for already-indexed content.
    """
    _reload_modules()
    from rubick_backend.ingest import ingest_path
    from rubick_backend.store import open_table

    first = ingest_path(notes_folder)
    table = open_table()
    rows_after_first = table.count_rows()
    assert first["files"] == 3
    assert first["chunks"] >= 3
    assert rows_after_first == first["chunks"]

    # Re-ingest the same folder. Every file should hit the dedup gate.
    second = ingest_path(notes_folder)
    rows_after_second = table.count_rows()
    assert rows_after_second == rows_after_first, (
        f"row count grew from {rows_after_first} to {rows_after_second} — "
        f"dedup did not catch the re-ingest"
    )
    # Pipeline reports skipped files for the dedup hits.
    assert second["files"] == 0
    assert second["chunks"] == 0
    assert second["skipped"] == 3


def test_index_job_end_to_end(
    isolated_data_dir: Path,
    notes_folder: Path,
) -> None:
    """POST /index/job → wait until succeeded → /search
    finds the new rows.

    Verifies the entire async pipeline end-to-end:

    1. The lifespan brings up a real ``JobQueue`` on ``app.state``.
    2. POST returns 202 with a ``queued`` job.
    3. The worker drains it, calling the real ``ingest_path`` which
       walks the folder, runs the embed model, writes to LanceDB.
    4. Polling GET converges to ``succeeded`` with sane stats.
    5. ``GET /search`` finds the freshly-ingested content — the
       LanceDB writes are visible to the search route running on
       the same connection.
    """
    _reload_modules()
    import time as _time

    from fastapi.testclient import TestClient

    from rubick_backend.main import app

    # ``TestClient`` as a context manager drives the lifespan, so
    # the queue is alive inside the ``with`` block and torn down
    # cleanly on exit.
    with TestClient(app) as client:
        resp = client.post("/index/job", json={"paths": [str(notes_folder)]})
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["id"]
        assert resp.json()["status"] == "queued"

        # Poll until done. Generous 60 s ceiling because the embed
        # model load is the slow part on first call.
        deadline = _time.time() + 60
        final = None
        while _time.time() < deadline:
            r = client.get(f"/index/job/{job_id}")
            assert r.status_code == 200
            final = r.json()
            if final["status"] in {"succeeded", "failed"}:
                break
            _time.sleep(0.25)

        assert final is not None
        assert final["status"] == "succeeded", final
        assert final["stats"]["files"] == 3
        assert final["stats"]["chunks"] >= 3
        assert final["stats"]["skipped"] == 0
        assert final["finished_at"] is not None

        # The newly-ingested data must show up in /search.
        r = client.get("/search", params={"q": "rover on the red planet"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1
        assert body["results"][0]["filename"] == "mars-exploration"


# === Image ingest smoke ======================================================


def _make_synthetic_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    """Materialize a textured RGB PNG so the byte gate doesn't fire and
    the model has something non-trivial to embed.

    We use a horizontal red→blue→green gradient overlay on a noise
    pattern. The exact content doesn't matter for the mechanical
    assertions — only that it parses, exceeds 5 KB, and isn't a solid
    color (which can collapse the embedding to a degenerate vector).
    """
    import numpy as np
    from PIL import Image as PILImage

    w, h = size
    rng = np.random.default_rng(seed=42)
    base = rng.integers(0, 64, size=(h, w, 3), dtype=np.uint8)
    # Horizontal gradient: red on the left, blue on the right.
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    base[:, :, 0] = np.clip(base[:, :, 0].astype(int) + (255 - grad), 0, 255).astype(np.uint8)
    base[:, :, 2] = np.clip(base[:, :, 2].astype(int) + grad, 0, 255).astype(np.uint8)
    # Vertical green ramp.
    vgrad = np.linspace(0, 255, h, dtype=np.uint8)
    base[:, :, 1] = np.clip(base[:, :, 1].astype(int) + vgrad[:, None], 0, 255).astype(np.uint8)
    PILImage.fromarray(base).save(path, format="PNG")


@pytest.fixture()
def mixed_folder(tmp_path: Path) -> Path:
    """A folder mixing text notes + a synthetic image."""
    d = tmp_path / "mixed"
    d.mkdir()
    for name, body in NOTES.items():
        (d / name).write_text(body, encoding="utf-8")
    _make_synthetic_image(d / "gradient.png")
    return d


def test_modality_filter_separates_text_and_image(
    isolated_data_dir: Path,
    mixed_folder: Path,
) -> None:
    """``GET /search?modality=image`` must only return image rows and
    vice versa. This is the integration-specific assertion — per-row
    shape (thumbnail, dimensions, embedding length) is covered in
    ``test_image_pipeline.py``.
    """
    _reload_modules()
    from fastapi.testclient import TestClient

    from rubick_backend.ingest import ingest_path
    from rubick_backend.main import app

    stats = ingest_path(mixed_folder)
    assert stats["files"] == 4  # 3 notes + 1 image
    client = TestClient(app)

    r_text = client.get("/search", params={"q": "rover", "modality": "text"})
    assert r_text.status_code == 200
    assert r_text.json()["count"] >= 1
    assert all(h["modality"] == "text" for h in r_text.json()["results"])

    r_image = client.get("/search", params={"q": "rover", "modality": "image"})
    assert r_image.status_code == 200
    assert r_image.json()["count"] == 1
    assert r_image.json()["results"][0]["filename"] == "gradient"
