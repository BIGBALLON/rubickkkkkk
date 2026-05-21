"""Tests for the cross-modality dispatcher in ``ingest/__init__.py``.

We verify file-walking (hidden / excluded dirs pruned, mixed text +
image extensions both collected) and extension-based routing
(``_pipeline_for`` picks the right sub-module). None of these tests
load the embedding model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rubick_backend import ingest, settings
from rubick_backend.ingest import (
    SUPPORTED_EXTENSIONS,
    _matches_user_pattern,
    _pipeline_for,
    _walk_supported_files,
)
from rubick_backend.ingest import image as image_mod
from rubick_backend.ingest import text as text_mod
from rubick_backend.ingest import video as video_mod


def _populate(root: Path) -> None:
    """A synthetic tree mixing text + image + non-supported files,
    plus the dirs we expect to be pruned.
    """
    # text
    (root / "a.md").write_text("# a\n\nhello")
    (root / "b.txt").write_text("plain text")
    (root / "c.markdown").write_text("# c")
    (root / "d.org").write_text("* org node")
    # image (just placeholders — the walk only checks extensions)
    (root / "photo1.jpg").write_bytes(b"\xff" * 16)
    (root / "photo2.PNG").write_bytes(b"\xff" * 16)  # case-insensitive
    (root / "photo3.heic").write_bytes(b"\xff" * 16)
    # video (placeholders)
    (root / "clip.mp4").write_bytes(b"\xff" * 16)
    (root / "screencast.MOV").write_bytes(b"\xff" * 16)
    # unsupported (audio extensions land here too — no longer ingested)
    (root / "memo.m4a").write_bytes(b"\xff" * 16)
    (root / "song.mp3").write_bytes(b"\xff" * 16)
    (root / "ignored.pdf").write_text("nope")
    (root / "ignored.py").write_text("print('hi')")
    # subdirectory contents
    (root / "sub").mkdir()
    (root / "sub" / "deep.md").write_text("# deep")
    (root / "sub" / "deep.jpg").write_bytes(b"\xff" * 16)
    (root / "sub" / "deep.wav").write_bytes(b"\xff" * 16)  # not collected
    (root / "sub" / "deep.webm").write_bytes(b"\xff" * 16)
    # pruning candidates
    (root / ".hidden_dir").mkdir()
    (root / ".hidden_dir" / "secret.md").write_text("nope")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "lib.md").write_text("nope")
    (root / ".hidden.md").write_text("nope")


# --- _walk_supported_files ----------------------------------------------------


def test_walk_collects_all_modality_extensions(tmp_path: Path) -> None:
    _populate(tmp_path)
    got = {p.name for p in _walk_supported_files(tmp_path)}
    assert got == {
        "a.md",
        "b.txt",
        "c.markdown",
        "d.org",
        "photo1.jpg",
        "photo2.PNG",
        "photo3.heic",
        "clip.mp4",
        "screencast.MOV",
        "deep.md",
        "deep.jpg",
        "deep.webm",
    }


def test_walk_skips_hidden_and_excluded_dirs(tmp_path: Path) -> None:
    _populate(tmp_path)
    paths = _walk_supported_files(tmp_path)
    for p in paths:
        for part in p.parts:
            assert not part.startswith(".")
            assert part != "node_modules"


def test_walk_returns_empty_for_empty_dir(tmp_path: Path) -> None:
    assert _walk_supported_files(tmp_path) == []


# --- user exclusion patterns (v1.x #3) ---------------------------------------


@pytest.fixture()
def _clean_user_patterns(monkeypatch: pytest.MonkeyPatch):
    """Snapshot + restore ``settings.EXCLUSION_PATTERNS`` around each
    test so the walker sees only what the test sets, without
    polluting siblings."""
    monkeypatch.setattr(settings, "EXCLUSION_PATTERNS", [])
    yield


def test_matches_user_pattern_helper() -> None:
    assert _matches_user_pattern("foo.tmp", ["*.tmp"]) is True
    assert _matches_user_pattern("foo.tmp", ["*.log"]) is False
    assert _matches_user_pattern("secrets", ["secrets"]) is True
    assert _matches_user_pattern("backup-2026", ["backup-*"]) is True
    assert _matches_user_pattern("anything", []) is False
    # ``fnmatchcase`` so the user can rely on case-sensitive matching
    # for repos that intentionally use mixed case.
    assert _matches_user_pattern("README.md", ["readme.md"]) is False


def test_walk_honours_user_pattern_on_filename(
    tmp_path: Path, _clean_user_patterns, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate(tmp_path)
    monkeypatch.setattr(settings, "EXCLUSION_PATTERNS", ["*.md"])
    paths = {p.name for p in _walk_supported_files(tmp_path)}
    for name in paths:
        assert not name.endswith(".md"), name
    # Non-matching extensions still come through.
    assert "b.txt" in paths
    assert "photo1.jpg" in paths


def test_walk_honours_user_pattern_on_dirname(
    tmp_path: Path, _clean_user_patterns, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate(tmp_path)
    # Add a 'secrets' dir with a supported file in it and verify the
    # user rule prunes the whole subtree.
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    (secret_dir / "private.md").write_text("nope")

    monkeypatch.setattr(settings, "EXCLUSION_PATTERNS", ["secrets"])
    paths = _walk_supported_files(tmp_path)
    for p in paths:
        assert "secrets" not in p.parts


def test_walk_honours_glob_dirname(
    tmp_path: Path, _clean_user_patterns, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backup-*`` should match any folder starting with that prefix."""
    _populate(tmp_path)
    bk = tmp_path / "backup-2026"
    bk.mkdir()
    (bk / "notes.md").write_text("ignored")

    monkeypatch.setattr(settings, "EXCLUSION_PATTERNS", ["backup-*"])
    paths = _walk_supported_files(tmp_path)
    for p in paths:
        assert "backup-2026" not in p.parts


def test_walk_with_empty_pattern_list_matches_baseline(
    tmp_path: Path, _clean_user_patterns
) -> None:
    """Setting ``EXCLUSION_PATTERNS=[]`` reproduces the v0.0.2
    behaviour (only the default deny-list applies)."""
    _populate(tmp_path)
    paths = {p.name for p in _walk_supported_files(tmp_path)}
    assert "a.md" in paths
    assert "secret.md" not in paths  # in .hidden_dir
    assert "lib.md" not in paths  # in node_modules


# --- combined SUPPORTED_EXTENSIONS -------------------------------------------


def test_supported_extensions_is_union_of_pipelines() -> None:
    expected = frozenset(
        text_mod.SUPPORTED_EXTENSIONS
        | image_mod.SUPPORTED_EXTENSIONS
        | video_mod.SUPPORTED_EXTENSIONS
    )
    assert SUPPORTED_EXTENSIONS == expected


def test_all_modality_extensions_are_pairwise_disjoint() -> None:
    """Sanity: a single extension must not be claimed by two pipelines —
    otherwise ``_pipeline_for`` order would matter for correctness, not
    just for import-cost ordering."""
    sets = {
        "text": text_mod.SUPPORTED_EXTENSIONS,
        "image": image_mod.SUPPORTED_EXTENSIONS,
        "video": video_mod.SUPPORTED_EXTENSIONS,
    }
    for a_name, a_set in sets.items():
        for b_name, b_set in sets.items():
            if a_name >= b_name:
                continue
            assert a_set & b_set == frozenset(), f"{a_name} & {b_name} overlap: {a_set & b_set}"


# --- _pipeline_for routing ----------------------------------------------------


def test_pipeline_for_text_extensions(tmp_path: Path) -> None:
    for name in ("a.md", "b.txt", "c.markdown", "d.org"):
        assert _pipeline_for(tmp_path / name) is text_mod, name


def test_pipeline_for_image_extensions(tmp_path: Path) -> None:
    for name in (
        "p.jpg",
        "p.jpeg",
        "P.PNG",
        "p.webp",
        "p.heic",
        "p.heif",
        "p.gif",
        "p.bmp",
        "p.tiff",
        "p.tif",
    ):
        assert _pipeline_for(tmp_path / name) is image_mod, name


def test_pipeline_for_audio_extensions_returns_none(tmp_path: Path) -> None:
    """Audio is no longer supported; every audio extension must hit the
    "no pipeline" branch so a regression that re-imports an audio
    pipeline can't quietly route through to the embedder."""
    for name in (
        "a.wav",
        "a.WAV",
        "a.mp3",
        "a.m4a",
        "a.flac",
        "a.aac",
        "a.ogg",
        "a.opus",
    ):
        assert _pipeline_for(tmp_path / name) is None, name


def test_pipeline_for_video_extensions(tmp_path: Path) -> None:
    for name in (
        "v.mp4",
        "v.MP4",
        "v.mov",
        "v.m4v",
        "v.webm",
        "v.mkv",
        "v.avi",
    ):
        assert _pipeline_for(tmp_path / name) is video_mod, name


def test_pipeline_for_unsupported_returns_none(tmp_path: Path) -> None:
    for name in ("p.pdf", "p.svg", "p.cr2", "p.exe", "p", "p.wmv", "p.flv"):
        assert _pipeline_for(tmp_path / name) is None, name


# --- ingest_file routing (with skip-rule short-circuit) -----------------------


def test_ingest_file_routes_unknown_extension_to_empty(tmp_path: Path) -> None:
    p = tmp_path / "x.pdf"
    p.write_text("pretend")
    assert ingest.ingest_file(p) == []


def test_ingest_file_routes_text_to_text_module(tmp_path: Path, monkeypatch) -> None:
    """``ingest.ingest_file`` must delegate to ``text.ingest_file`` for
    text extensions; we patch the leaf to confirm wiring without
    invoking the embedder.
    """
    captured: list[Path] = []
    monkeypatch.setattr(text_mod, "ingest_file", lambda p: captured.append(Path(p)) or [])
    p = tmp_path / "x.md"
    p.write_text("# hi")
    ingest.ingest_file(p)
    assert captured == [p]


def test_ingest_file_routes_image_to_image_module(tmp_path: Path, monkeypatch) -> None:
    captured: list[Path] = []
    monkeypatch.setattr(image_mod, "ingest_file", lambda p: captured.append(Path(p)) or [])
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff")
    ingest.ingest_file(p)
    assert captured == [p]


# --- ingest_path progress + pause hooks --------------------------------------


def test_ingest_path_emits_progress_per_file(tmp_path: Path, monkeypatch) -> None:
    """``ingest_path`` must fire ``progress_cb`` once with (0, total,
    None, 0) and then once per file with (idx, total, str(file),
    embedded). ``embedded`` is the running count of files that
    actually produced rows — a UI uses ``embedded == 0 and done > 0``
    to render "scanning" instead of "indexing".
    """
    # Three text files, no actual embedding (we monkeypatch the text
    # pipeline to a no-op so the fast suite stays fast).
    for i in range(3):
        (tmp_path / f"note{i}.md").write_text(f"# note {i}")
    monkeypatch.setattr(text_mod, "ingest_file", lambda _p: [])

    # Stub the table.add path so the function compiles without
    # opening a real LanceDB.
    class _NoTable:
        def add(self, _rows):  # noqa: ANN001
            pass

    progress: list[tuple[int, int, str | None, int]] = []
    ingest.ingest_path(
        tmp_path,
        table=_NoTable(),
        progress_cb=lambda done, total, current, embedded: progress.append(
            (done, total, current, embedded)
        ),
    )

    assert len(progress) == 4  # initial 0 + one per file
    assert progress[0] == (0, 3, None, 0)
    dones = [p[0] for p in progress[1:]]
    assert dones == [1, 2, 3]
    totals = {p[1] for p in progress}
    assert totals == {3}
    # All three files were stubbed to produce 0 rows → embedded stays 0.
    embeddeds = [p[3] for p in progress]
    assert embeddeds == [0, 0, 0, 0]
    # Last current must be a real path string, never None.
    assert progress[-1][2] is not None
    assert progress[-1][2].endswith(".md")


def test_ingest_path_blocks_on_cleared_pause_event(
    tmp_path: Path, monkeypatch
) -> None:
    """The pause-event mirror is what makes "Pause" effective mid-
    folder. Clear the event before ingest starts and assert that
    the loop never reaches the second file until we set it again.
    """
    import threading
    import time

    for i in range(2):
        (tmp_path / f"note{i}.md").write_text(f"# note {i}")
    monkeypatch.setattr(text_mod, "ingest_file", lambda _p: [])

    class _NoTable:
        def add(self, _rows):  # noqa: ANN001
            pass

    pause = threading.Event()
    pause.clear()  # paused before we even start

    progress_seen: list[int] = []

    def cb(done: int, _total: int, _current: str | None, _embedded: int) -> None:
        progress_seen.append(done)

    done_event = threading.Event()

    def runner() -> None:
        ingest.ingest_path(
            tmp_path,
            table=_NoTable(),
            progress_cb=cb,
            pause_event=pause,
        )
        done_event.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()

    # Even after a generous pause, the loop must not have advanced
    # past the initial ``progress_cb(0, total, None)`` emission.
    time.sleep(0.1)
    assert progress_seen == [0]
    assert not done_event.is_set()

    pause.set()
    done_event.wait(timeout=1.0)
    assert done_event.is_set()
    # After resume the loop drained both files.
    assert progress_seen[-1] == 2
