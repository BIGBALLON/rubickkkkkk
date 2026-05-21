"""Unit tests for ``ingest/image.py``.

We don't load the embedding model — calls into ``embed_image`` are
monkeypatched to a constant 768-dim unit vector so we can verify the
gate logic, resize math, EXIF parsing, and thumbnail generation in
under a second.

The slow end-to-end smoke (real ``embed_image`` + LanceDB writes +
cross-modal search) lives in ``test_smoke_e2e.py`` behind the
``slow`` mark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rubick_backend.ingest import image as image_mod


@pytest.fixture(autouse=True)
def patch_embed_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``embed_image_preprocessed`` with a deterministic constant
    vector and drop ``MIN_FILE_BYTES`` to 0 so synthetic small PNGs are
    not rejected by the byte gate.

    The real embed path lazy-loads ~1.8 GB of weights — fine for the
    slow smoke, far too heavy for boundary-condition unit tests.
    The returned vector is unit-norm so the schema validator's
    "dim == 768" check still passes.

    Individual tests that need the production byte gate (e.g.
    ``test_skip_tiny_file_bytes``) re-monkeypatch ``MIN_FILE_BYTES``
    back to its spec value inside the test body.
    """
    vec = np.zeros(768, dtype=np.float32)
    vec[:384] = 1.0 / np.sqrt(384)  # unit-norm

    class _FakeState:
        tokenizer = None

    monkeypatch.setattr(image_mod, "load", lambda: _FakeState())
    monkeypatch.setattr(
        "rubick_backend.embed.preprocessing.preprocess_image",
        lambda img, **_: {"img": img},
    )
    monkeypatch.setattr(image_mod, "embed_image_preprocessed", lambda _p: vec)
    monkeypatch.setattr(image_mod, "MIN_FILE_BYTES", 0)


# === Helpers ================================================================


def _write_png(path: Path, size: tuple[int, int], color=(200, 30, 30)) -> None:
    """Materialize a real RGB PNG file at ``path``.

    We need an actual PNG on disk (not just bytes) because
    ``ingest_file`` re-reads the bytes to compute SHA256.
    """
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")


def _write_tiny_png(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """A < MIN_PIXEL_DIMENSION PNG used to exercise the pixel gate."""
    img = Image.new("RGB", size, (10, 20, 30))
    # Force a tiny on-disk size — Pillow's default compression keeps a
    # 32x32 PNG well under 5 KB anyway, but we pad to be explicit.
    img.save(path, format="PNG", optimize=True)


# === SUPPORTED_EXTENSIONS ====================================================


def test_supported_extensions_match_spec() -> None:
    """Supported image extensions (case variants included)."""
    expected = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".heic",
        ".heif",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
    }
    assert image_mod.SUPPORTED_EXTENSIONS == frozenset(expected)


# === Skip-rule gates =========================================================


def test_skip_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.pdf"
    p.write_bytes(b"\x00" * 10_000)
    assert image_mod.ingest_file(p) == []


def test_skip_directory(tmp_path: Path) -> None:
    d = tmp_path / "dir.png"
    d.mkdir()
    assert image_mod.ingest_file(d) == []


def test_skip_tiny_file_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-enable the production byte gate (autouse fixture relaxes it).
    monkeypatch.setattr(image_mod, "MIN_FILE_BYTES", 5 * 1024)
    p = tmp_path / "tiny.png"
    _write_png(p, (200, 200))
    p.write_bytes(p.read_bytes()[:100])
    assert image_mod.ingest_file(p) == []


def test_skip_tiny_pixels(tmp_path: Path) -> None:
    p = tmp_path / "tiny.png"
    _write_tiny_png(p, (32, 32))
    assert image_mod.ingest_file(p) == []


def test_skip_decode_failure(tmp_path: Path) -> None:
    """A file with a valid extension but invalid contents must be
    skipped with a warning, not raise."""
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 1000)
    assert image_mod.ingest_file(p) == []


# === Happy path: a single row, schema-valid =================================


def test_ingest_emits_one_row_with_expected_fields(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    p = tmp_path / "good.png"
    _write_png(p, (200, 300))
    rows = image_mod.ingest_file(p)
    assert len(rows) == 1
    row = rows[0]
    assert row["modality"] == "image"
    assert row["chunk_idx"] == 0
    assert row["width"] == 200
    assert row["height"] == 300
    assert row["filename"] == "good"
    assert row["file_paths"] == [str(p)]
    assert len(row["embedding"]) == 768
    assert row["thumbnail_path"] is not None
    # id format: <doc_id>-<modality>-<chunk_idx>
    assert row["id"] == f"{row['doc_id']}-image-0"
    # raw_text and chunk_n_tokens are text-only fields — must be None
    assert row["raw_text"] is None
    assert row["chunk_n_tokens"] is None


def test_ingest_writes_webp_thumbnail(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    """128-px short-edge WebP at
    ``<DATA_ROOT>/thumbnails/<doc_id>.webp``.
    """
    p = tmp_path / "good.jpg"
    _write_png(p.with_suffix(".png"), (640, 480))
    # Re-encode as JPEG to exercise the JPG branch.
    Image.open(p.with_suffix(".png")).save(p, format="JPEG", quality=85)

    rows = image_mod.ingest_file(p)
    assert rows
    thumb_path = Path(rows[0]["thumbnail_path"])
    assert thumb_path.exists()
    assert thumb_path.suffix == ".webp"
    assert thumb_path.parent == isolated_data_dir / "thumbnails"
    with Image.open(thumb_path) as t:
        assert t.format == "WEBP"
        # short edge == THUMBNAIL_SHORT_EDGE (128 px)
        assert min(t.size) == image_mod.THUMBNAIL_SHORT_EDGE


def test_oversize_image_is_resized_for_embedding(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long edge > 1280 → short edge resized to 768 before preprocess."""
    captured: list[tuple[int, int]] = []

    def fake_preprocess(img, **_) -> dict:
        captured.append(img.size)
        return {}

    monkeypatch.setattr(
        "rubick_backend.embed.preprocessing.preprocess_image",
        fake_preprocess,
    )

    # 5000x2500 → long-edge > 1280, short-edge lands at 768 →
    # post-resize long-edge is 5000 * 768 / 2500 = 1536.
    p = tmp_path / "huge.png"
    _write_png(p, (5000, 2500))

    rows = image_mod.ingest_file(p)
    assert rows
    row_dims = captured[0]
    assert min(row_dims) == image_mod.RESIZE_SHORT_EDGE
    assert max(row_dims) == 1536
    # Stored width/height are the *original* dimensions before resize.
    assert rows[0]["width"] == 5000
    assert rows[0]["height"] == 2500


def test_under_threshold_image_is_not_resized(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[int, int]] = []

    def fake_preprocess(img, **_) -> dict:
        captured.append(img.size)
        return {}

    monkeypatch.setattr(
        "rubick_backend.embed.preprocessing.preprocess_image",
        fake_preprocess,
    )

    p = tmp_path / "normal.png"
    _write_png(p, (1200, 800))
    image_mod.ingest_file(p)
    # Long edge 1200 <= 1280 → no resize.
    assert captured == [(1200, 800)]


def test_doc_id_is_deterministic_across_runs(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    p = tmp_path / "same.png"
    _write_png(p, (200, 300), color=(10, 20, 30))
    r1 = image_mod.ingest_file(p)
    r2 = image_mod.ingest_file(p)
    assert r1[0]["doc_id"] == r2[0]["doc_id"]
    assert r1[0]["sha256"] == r2[0]["sha256"]


# === EXIF parsing ============================================================


def test_exif_datetime_original_parsed(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    """A JPEG carrying EXIF DateTimeOriginal must surface as a Unix ts
    in the row's ``exif_taken_at`` field.
    """
    img = Image.new("RGB", (200, 200), (50, 100, 150))
    # Manually build a minimal EXIF block with DateTimeOriginal.
    exif = img.getexif()
    exif[image_mod._EXIF_DATETIME_ORIGINAL] = "2023:06:15 12:34:56"
    p = tmp_path / "exif.jpg"
    img.save(p, format="JPEG", exif=exif.tobytes())

    rows = image_mod.ingest_file(p)
    assert rows
    ts = rows[0]["exif_taken_at"]
    assert ts is not None
    # Local-time decode; we just sanity-check it's in 2023 and roughly
    # within the day, avoiding TZ-aware comparison flakiness.
    import datetime as _dt

    dt = _dt.datetime.fromtimestamp(ts)
    assert dt.year == 2023 and dt.month == 6 and dt.day == 15


def test_exif_missing_is_silently_null(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    p = tmp_path / "no_exif.png"
    _write_png(p, (200, 200))
    rows = image_mod.ingest_file(p)
    assert rows
    assert rows[0]["exif_taken_at"] is None


def test_exif_malformed_is_silently_null(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    img = Image.new("RGB", (200, 200), (50, 100, 150))
    exif = img.getexif()
    exif[image_mod._EXIF_DATETIME_ORIGINAL] = "this is not a date"
    p = tmp_path / "bad_exif.jpg"
    img.save(p, format="JPEG", exif=exif.tobytes())
    rows = image_mod.ingest_file(p)
    assert rows
    assert rows[0]["exif_taken_at"] is None


# === Helpers tested directly =================================================


@pytest.mark.parametrize(
    "size,target,expected_short,expected_long",
    [
        ((1000, 500), 200, 200, 400),  # landscape (2:1)
        ((500, 1000), 200, 200, 400),  # portrait (1:2)
    ],
)
def test_resize_short_edge_preserves_aspect(
    size: tuple[int, int],
    target: int,
    expected_short: int,
    expected_long: int,
) -> None:
    """The short edge lands on ``target``; the long edge scales to keep
    aspect, ±1 px from the round-trip through int rounding."""
    img = Image.new("RGB", size, (0, 0, 0))
    out = image_mod._resize_short_edge(img, target)
    assert min(out.size) == expected_short
    assert abs(max(out.size) - expected_long) <= 1
