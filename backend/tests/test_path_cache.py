"""Unit tests for the ``store.path_cache`` fast-skip cache.

Covers the seven properties that matter for the fast-skip path on
ingest restart:

1. Round-trip: ``record`` → ``flush`` → reload → ``lookup`` hits.
2. mtime change → cache miss (correctness on edited files).
3. Same (path, mtime, doc_id) recorded twice → no rewrite (no flush
   noise, ``mtime``-stable cache file).
4. mtime advance → entry replaced.
5. Corrupt cache JSON → graceful empty start.
6. Atomic flush → ``Path.replace`` writes the real file, leaves no
   ``.tmp`` sibling.
7. Concurrent ``record`` from multiple threads → no lost updates.

Tests are independent of LanceDB: ``path_cache`` is a pure dict +
JSON file, so we point it at a tmp dir via ``_reset_for_tests``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from rubick_backend.store import path_cache


def test_record_flush_reload_roundtrip(tmp_path: Path) -> None:
    """End-to-end: record entries, flush, simulate process restart by
    re-loading the cache from disk, lookups still hit.  Also asserts the
    flush is atomic — no ``.tmp`` sibling after success."""
    path_cache._reset_for_tests(tmp_path)

    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")
    path_cache.record("/photos/b.jpg", mtime=2000, doc_id="bbbbbbbbbbbbbbbb")
    path_cache.flush()

    cache_file = tmp_path / "path_mtime_cache.json"
    assert cache_file.exists()
    assert not (tmp_path / "path_mtime_cache.tmp").exists()

    path_cache._reset_for_tests(tmp_path)  # simulate process restart
    assert path_cache.lookup("/notes/a.md", mtime=1000) == "aaaaaaaaaaaaaaaa"
    assert path_cache.lookup("/photos/b.jpg", mtime=2000) == "bbbbbbbbbbbbbbbb"


def test_lookup_misses_on_mtime_change_or_unknown_path(tmp_path: Path) -> None:
    """The two cache-miss paths the ingest pipeline relies on for
    correctness: edited file (mtime advanced) + brand-new file (path
    never seen)."""
    path_cache._reset_for_tests(tmp_path)
    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")

    assert path_cache.lookup("/notes/a.md", mtime=1001) is None
    assert path_cache.lookup("/notes/never-seen.md", mtime=1000) is None


def test_record_same_entry_does_not_re_dirty(tmp_path: Path) -> None:
    """Recording the exact same triple again must not flip the dirty
    flag — keeps the cache file's mtime stable so backup tools don't
    pointlessly re-snapshot it."""
    path_cache._reset_for_tests(tmp_path)
    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")
    path_cache.flush()
    cache_file = tmp_path / "path_mtime_cache.json"
    first_mtime_ns = cache_file.stat().st_mtime_ns

    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")
    path_cache.flush()  # nothing dirty → no rewrite
    assert cache_file.stat().st_mtime_ns == first_mtime_ns


def test_record_mtime_advance_replaces_entry(tmp_path: Path) -> None:
    """User edits a file → ingest re-embeds → record() with new mtime
    overwrites the old (path, mtime, doc_id) triple."""
    path_cache._reset_for_tests(tmp_path)
    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")
    path_cache.record("/notes/a.md", mtime=2000, doc_id="cccccccccccccccc")
    path_cache.flush()

    on_disk = json.loads((tmp_path / "path_mtime_cache.json").read_text())
    assert on_disk == {"/notes/a.md": {"mtime": 2000, "doc_id": "cccccccccccccccc"}}


def test_corrupt_cache_file_falls_back_to_empty(tmp_path: Path) -> None:
    """A truncated / non-JSON cache file must not crash ingest — we
    log + start empty, and subsequent record/lookup keep working."""
    (tmp_path / "path_mtime_cache.json").write_text("{not valid json", encoding="utf-8")
    path_cache._reset_for_tests(tmp_path)

    assert path_cache.lookup("/notes/a.md", mtime=1000) is None
    path_cache.record("/notes/a.md", mtime=1000, doc_id="aaaaaaaaaaaaaaaa")
    assert path_cache.lookup("/notes/a.md", mtime=1000) == "aaaaaaaaaaaaaaaa"


def test_concurrent_records_all_persist(tmp_path: Path) -> None:
    """Hammer the cache from 8 threads — every record must survive
    the eventual flush.  Defends against the lost-update race that
    would exist if ``record`` ever lost its lock."""
    path_cache._reset_for_tests(tmp_path)

    def worker(start: int) -> None:
        for i in range(50):
            path_cache.record(
                f"/p/{start}/{i}.md",
                mtime=1000 + i,
                doc_id=f"{start:08d}{i:08d}",
            )

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    path_cache.flush()
    on_disk = json.loads((tmp_path / "path_mtime_cache.json").read_text())
    assert len(on_disk) == 8 * 50
