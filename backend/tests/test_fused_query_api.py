"""Unit + API tests for the fused multimodal query path (``POST /search``).

Two surfaces under test:

- ``embed.loader.embed_query_fused`` argument validation (the loader
  keeps its video forward path intact for future use, but only the
  image branch is wired through to HTTP in v1.x).
- ``POST /search`` (multipart, image-only) — missing / empty /
  oversized handling, filter pass-through, and the basic image-as-
  query happy path with the embedder + retrieve layer monkeypatched
  out. The route now calls ``embed_query_fused_weighted``; the old
  single-pass ``embed_query_fused`` is kept for backward compat.

The slow end-to-end smokes (real MLX model + real LanceDB) live in
``tests/test_fused_query_e2e.py`` behind ``RUBICK_RUN_SLOW=1``.
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from rubick_backend import embed as embed_pkg
from rubick_backend.api.search import router

# === Loader argument validation =============================================


def test_embed_query_fused_rejects_no_attachment() -> None:
    """Refusing zero attachments is a guardrail against the FastAPI
    route silently calling the loader without an attachment when a
    future refactor forgets to validate at the HTTP layer."""
    from rubick_backend.embed.loader import embed_query_fused

    with pytest.raises(ValueError, match="requires one of"):
        embed_query_fused(text="hi")


def test_embed_query_fused_rejects_two_attachments() -> None:
    """The loader keeps a video forward path for future use, so this
    test exercises its multi-attachment guard. v1.x doesn't expose
    the video branch through HTTP."""
    from rubick_backend.embed.loader import embed_query_fused

    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    fake_video = [Image.new("RGB", (32, 32), color=(0, 0, 0)) for _ in range(2)]
    with pytest.raises(ValueError, match="exactly one attachment"):
        embed_query_fused(text=None, image=img, video_frames=fake_video)


# === HTTP route ============================================================


@pytest.fixture()
def fake_qvec() -> np.ndarray:
    """Deterministic stand-in for embed_query_fused's output. 768-dim
    L2-normalized."""
    rng = np.random.default_rng(seed=42)
    v = rng.standard_normal(768).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture()
def patched_app(
    monkeypatch: pytest.MonkeyPatch, fake_qvec: np.ndarray
) -> Iterator[FastAPI]:
    """FastAPI app with only the ``/search`` router mounted and
    embed + retrieve replaced with deterministic stubs.

    - ``embed_query_fused_weighted`` returns ``fake_qvec`` regardless
      of input, so the test exercises the route's plumbing without
      loading MLX or its 1.8 GB of weights.
    - ``hybrid_search`` returns an empty list — we only care that the
      route correctly composed its arguments and emitted the JSON
      envelope.
    """
    monkeypatch.setattr(
        "rubick_backend.api.search.embed_query_fused_weighted",
        lambda **kw: fake_qvec,
    )

    captured: dict[str, object] = {}

    def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "rubick_backend.api.search.hybrid_search", fake_hybrid_search
    )
    # Also stub image decode so we don't pull pillow-heif into the
    # fast suite. Signature is now (text, payload, text_weight).
    monkeypatch.setattr(
        "rubick_backend.api.search._embed_query_image_from_bytes",
        lambda text, payload, text_weight: fake_qvec,
    )

    app = FastAPI()
    app.include_router(router)
    app.state.captured = captured
    yield app


@pytest.fixture()
def client(patched_app: FastAPI) -> TestClient:
    return TestClient(patched_app)


def _png_bytes() -> bytes:
    """Tiny in-memory PNG for the upload field."""
    img = Image.new("RGB", (16, 16), color=(255, 128, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_post_search_requires_an_image(client: TestClient) -> None:
    """Text-only queries belong on GET /search — POST without an
    image attachment is a 400, not a fall-through to text-only
    behaviour."""
    r = client.post("/search", data={"q": "hello"})
    assert r.status_code == 400
    assert "image attachment" in r.json()["detail"]


def test_post_search_image_happy_path(
    client: TestClient,
    patched_app: FastAPI,
) -> None:
    """Image attachment → embed_query_fused stub → hybrid_search with
    qvec set + qtext=None (the v1.x design's "fused queries skip BM25"
    invariant)."""
    img_bytes = _png_bytes()
    r = client.post(
        "/search",
        data={"q": "like this but warmer", "limit": "20"},
        files={"image": ("photo.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 0  # stubbed hybrid_search returns []
    assert body["query"] == ""  # qtext was None on the wire

    captured = patched_app.state.captured
    # The fused leg must pass qvec, NEVER qtext.
    assert captured["qtext"] is None
    assert captured["qvec"] is not None
    assert captured["doc_limit"] == 20


def test_post_search_pure_image_query_no_text(
    client: TestClient, patched_app: FastAPI
) -> None:
    """Empty ``q`` + image is a valid "find me images like this"
    query — degrades to pure media embedding."""
    img_bytes = _png_bytes()
    r = client.post(
        "/search",
        data={"q": ""},
        files={"image": ("photo.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text


def test_post_search_empty_attachment_is_400(client: TestClient) -> None:
    r = client.post(
        "/search",
        data={"q": ""},
        files={"image": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_post_search_rejects_oversized_image(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 35 MB blob comes back as 413 with the cap surfaced; the
    embed stub is never called."""
    huge = b"\x00" * (35 * 1024 * 1024)
    r = client.post(
        "/search",
        data={"q": ""},
        files={"image": ("big.bin", huge, "application/octet-stream")},
    )
    assert r.status_code == 413
    assert "max" in r.json()["detail"]


def test_post_search_forwards_filters(
    client: TestClient, patched_app: FastAPI
) -> None:
    """Modality / path_prefix / mtime range / include_rejected all
    round-trip through the Form layer into hybrid_search kwargs.
    """
    img_bytes = _png_bytes()
    r = client.post(
        "/search",
        data={
            "q": "",
            "limit": "5",
            "modality": "image",
            "path_prefix": "/Users/foo/photos",
            "mtime_after": "1700000000",
            "mtime_before": "1800000000",
            "include_rejected": "true",
        },
        files={"image": ("a.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    captured = patched_app.state.captured
    assert captured["modality"] == "image"
    assert captured["path_prefix"] == "/Users/foo/photos"
    assert captured["mtime_after"] == 1700000000
    assert captured["mtime_before"] == 1800000000
    assert captured["include_rejected"] is True
    assert captured["doc_limit"] == 5


def test_get_search_still_works(
    client: TestClient,
    patched_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    fake_qvec: np.ndarray,
) -> None:
    """The text-only GET path is untouched by the fused additions —
    qtext is forwarded as-is so the BM25 leg still runs upstream.
    """
    monkeypatch.setattr(
        "rubick_backend.api.search.embed_query",
        lambda q: fake_qvec,
    )
    r = client.get("/search", params={"q": "hello"})
    assert r.status_code == 200, r.text
    captured = patched_app.state.captured
    assert captured["qtext"] == "hello"


def test_embed_pkg_exports_fused_loader() -> None:
    """Both fused helpers must be in the public surface."""
    assert hasattr(embed_pkg, "embed_query_fused")
    assert "embed_query_fused" in embed_pkg.__all__
    assert hasattr(embed_pkg, "embed_query_fused_weighted")
    assert "embed_query_fused_weighted" in embed_pkg.__all__
