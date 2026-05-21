"""Unit tests for ``ingest/video.py``.

We never load the embedding model here. The expensive callout
(``embed_video``) and the PyAV-heavy helpers (``_probe_video``,
``_decode_uniform_frames``) are monkeypatched per-test, so each case
runs in milliseconds against pure Python objects.

The slow end-to-end smoke that actually opens an MP4 with PyAV lives
in ``test_smoke_e2e.py`` behind the ``slow`` mark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rubick_backend.ingest import video as video_mod

# === Helpers ================================================================


def _make_video_placeholder(path: Path, n_bytes: int = 4096) -> None:
    """Create a non-empty placeholder file at ``path``.

    The PyAV-heavy helpers are monkeypatched, so the file's actual
    content doesn't have to be a real video container — we just need
    a file on disk so ``hashlib`` has bytes to chew on and the
    ``is_file`` gate passes.
    """
    path.write_bytes(b"\xff" * n_bytes)


def _fake_frames(n: int) -> list[Image.Image]:
    """A list of ``n`` distinct 64×64 RGB images (red, green, blue, …)."""
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
        (200, 100, 50),
    ]
    return [Image.new("RGB", (64, 64), palette[i % len(palette)]) for i in range(n)]


@pytest.fixture(autouse=True)
def patch_pyav_and_embedders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default stubs: 30-s video + canned 32 frames. Individual tests
    override these by re-monkeypatching after the autouse setup runs.
    """
    monkeypatch.setattr(video_mod, "_probe_video", lambda _p: 30.0)
    monkeypatch.setattr(
        video_mod,
        "_decode_uniform_frames",
        lambda _p, *, n_target, thumb_at_s: (_fake_frames(n_target), _fake_frames(1)[0]),
    )

    vec = np.zeros(768, dtype=np.float32)
    vec[:256] = 1.0 / np.sqrt(256)
    counters = {"embed_video": 0}

    def fake_embed_video(frames):
        counters["embed_video"] += 1
        return vec

    monkeypatch.setattr(video_mod, "embed_video", fake_embed_video)
    video_mod._test_counters = counters  # type: ignore[attr-defined]


# === SUPPORTED_EXTENSIONS ===================================================


def test_supported_extensions() -> None:
    """Video extension set — our v1 choice, not exhaustive."""
    expected = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
    assert video_mod.SUPPORTED_EXTENSIONS == frozenset(expected)


# === Skip gates =============================================================


def test_skip_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.xyz"
    _make_video_placeholder(p)
    assert video_mod.ingest_file(p) == []


def test_skip_not_a_file(tmp_path: Path) -> None:
    d = tmp_path / "dir.mp4"
    d.mkdir()
    assert video_mod.ingest_file(d) == []


def test_skip_probe_failure(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_mod, "_probe_video", lambda _p: None)
    p = tmp_path / "corrupt.mp4"
    _make_video_placeholder(p)
    assert video_mod.ingest_file(p) == []


def test_skip_when_decode_returns_too_few_frames(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degenerate video (e.g. corrupted such that only 1 frame
    decodes) must be skipped — encode_video requires even frame
    count ≥ 2.
    """
    monkeypatch.setattr(
        video_mod,
        "_decode_uniform_frames",
        lambda _p, *, n_target, thumb_at_s: (_fake_frames(1), _fake_frames(1)[0]),
    )
    p = tmp_path / "degenerate.mp4"
    _make_video_placeholder(p)
    assert video_mod.ingest_file(p) == []


# === Reject path ============================================================


def test_oversize_emits_single_rejected_row(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """> 2 min → exactly one row with modality="rejected"."""
    monkeypatch.setattr(video_mod, "_probe_video", lambda _p: 180.0)
    p = tmp_path / "long.mp4"
    _make_video_placeholder(p)
    rows = video_mod.ingest_file(p)
    assert len(rows) == 1
    r = rows[0]
    assert r["modality"] == "rejected"
    assert r["rejected_reason"] == "video_too_long"
    assert r["status"] == "rejected"
    assert r["duration_s"] == pytest.approx(180.0)
    # Critically: no embed forward pass for rejected.
    assert video_mod._test_counters["embed_video"] == 0  # type: ignore[attr-defined]


# === Happy path: visual-only output (no transcript track) ====================


def test_happy_path_emits_single_video_row(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    """Exactly one ``modality="video"`` row per file from the visual
    track. The Whisper transcript track was removed in v0.0.2; the
    audio-tower path was removed in the follow-up audio cleanup.
    """
    p = tmp_path / "good.mp4"
    _make_video_placeholder(p)
    rows = video_mod.ingest_file(p)

    assert len(rows) == 1
    video_row = rows[0]
    assert video_row["modality"] == "video"
    assert video_row["chunk_idx"] == 0
    assert video_row["duration_s"] == pytest.approx(30.0)
    assert video_row["n_frames_sampled"] == video_mod.N_FRAMES_SAMPLED
    assert video_row["thumbnail_path"] is not None
    assert Path(video_row["thumbnail_path"]).exists()
    assert len(video_row["embedding"]) == 768


def test_thumbnail_is_128px_webp(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    p = tmp_path / "thumb.mp4"
    _make_video_placeholder(p)
    rows = video_mod.ingest_file(p)
    thumb = Path(rows[0]["thumbnail_path"])
    assert thumb.exists()
    assert thumb.suffix == ".webp"
    assert thumb.parent == isolated_data_dir / "thumbnails"
    with Image.open(thumb) as t:
        assert t.format == "WEBP"
        assert min(t.size) == video_mod.THUMBNAIL_SHORT_EDGE


def test_frame_count_is_truncated_to_even(
    tmp_path: Path,
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """jina v5 omni's video path uses temporal_patch_size=2 — an odd
    frame list would crash, so ``ingest_file`` must trim the last
    one before calling embed_video.
    """
    monkeypatch.setattr(
        video_mod,
        "_decode_uniform_frames",
        lambda _p, *, n_target, thumb_at_s: (_fake_frames(31), _fake_frames(1)[0]),
    )

    captured: list[int] = []

    def fake_embed_video(frames):
        captured.append(len(frames))
        return np.zeros(768, dtype=np.float32) + (1.0 / np.sqrt(768))

    monkeypatch.setattr(video_mod, "embed_video", fake_embed_video)

    p = tmp_path / "odd_frames.mp4"
    _make_video_placeholder(p)
    rows = video_mod.ingest_file(p)
    assert captured == [30]
    assert rows[0]["n_frames_sampled"] == 30


def test_doc_id_is_deterministic_across_runs(
    tmp_path: Path,
    isolated_data_dir: Path,
) -> None:
    p = tmp_path / "same.mp4"
    _make_video_placeholder(p)
    r1 = video_mod.ingest_file(p)
    r2 = video_mod.ingest_file(p)
    assert r1[0]["doc_id"] == r2[0]["doc_id"]
    assert r1[0]["sha256"] == r2[0]["sha256"]
