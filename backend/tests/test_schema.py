"""Unit tests for ``store/schema.py``.

We exercise the schema construction, ``make_row`` validation, the
``make_doc_id`` deterministic hashing, and a smoke that ``open_table``
creates a fresh table the first time and reopens it the second time —
all against a tmp ``RUBICK_DATA_DIR``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pyarrow as pa
import pytest


def _reload_store() -> tuple:
    """Reload settings + store so module-level constants pick up env vars.

    Order matters: ``store.schema`` snapshots ``settings.EMBED_DIM`` etc.
    at import time, so we must reload it after settings.
    """
    import rubick_backend.settings as s

    importlib.reload(s)
    import rubick_backend.store.schema as sch

    importlib.reload(sch)
    return s, sch


# --- doc id -------------------------------------------------------------------


def test_make_doc_id_is_first_16_of_sha256_and_deterministic() -> None:
    """``make_doc_id`` is the spec's content-hash key. One test pins
    both the algorithm (sha256[:16]) and the equality / inequality
    properties — ingest dedup relies on both."""
    import hashlib

    _, sch = _reload_store()
    payload = b"hello world"
    got = sch.make_doc_id(payload)
    assert got == hashlib.sha256(payload).hexdigest()[:16]
    assert len(got) == 16
    assert sch.make_doc_id(b"a") == sch.make_doc_id(b"a")
    assert sch.make_doc_id(b"a") != sch.make_doc_id(b"b")


# --- schema fields ------------------------------------------------------------


def test_schema_required_fields_present_and_typed() -> None:
    _, sch = _reload_store()
    schema: pa.Schema = sch.SCHEMA
    for required in (
        "id",
        "doc_id",
        "file_paths",
        "modality",
        "chunk_idx",
        "embedding",
        "created_at",
        "mtime",
        "sha256",
        "filename",
    ):
        assert required in schema.names, f"missing required field {required}"

    emb_field = schema.field("embedding")
    assert pa.types.is_fixed_size_list(emb_field.type)
    assert emb_field.type.list_size == 768


def test_modalities_match_spec() -> None:
    """Audio + the legacy ``*_transcript`` siblings are retired from the
    write-side enum. Old indexes may still carry rows with those values
    — see ``store/schema.py`` module docstring.
    """
    _, sch = _reload_store()
    assert sch.MODALITIES == frozenset(
        {
            "text",
            "image",
            "video",
            "rejected",
        }
    )


def test_make_row_rejects_audio_modality() -> None:
    """Audio is retired. ``make_row`` must refuse new rows with the
    ``audio`` modality so a regression that re-introduces an audio
    pipeline can't quietly write into the table.
    """
    _, sch = _reload_store()
    with pytest.raises(ValueError, match="unknown modality 'audio'"):
        sch.make_row(
            doc_id="0" * 16,
            modality="audio",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/tmp/x.wav",
            sha256="0" * 64,
            mtime=0,
            filename="x",
        )


# --- make_row -----------------------------------------------------------------


def _good_row_kwargs(sch) -> dict:
    return dict(
        doc_id="0123456789abcdef",
        modality="text",
        chunk_idx=0,
        embedding=[0.0] * 768,
        file_path="/tmp/foo.md",
        sha256="0123456789abcdef" * 4,
        mtime=1_700_000_000,
        filename="foo",
        raw_text="hello",
        chunk_n_tokens=12,
    )


def test_make_row_happy_path() -> None:
    _, sch = _reload_store()
    row = sch.make_row(**_good_row_kwargs(sch))
    assert row["id"] == "0123456789abcdef-text-0"
    assert row["file_paths"] == ["/tmp/foo.md"]
    assert len(row["embedding"]) == 768
    assert row["created_at"] > 0
    # every schema-required field is populated
    for name in sch.SCHEMA.names:
        assert name in row


def test_make_row_rejects_unknown_modality() -> None:
    _, sch = _reload_store()
    kw = _good_row_kwargs(sch)
    kw["modality"] = "pdf"
    with pytest.raises(ValueError, match="unknown modality"):
        sch.make_row(**kw)


def test_make_row_rejects_wrong_embedding_dim() -> None:
    _, sch = _reload_store()
    kw = _good_row_kwargs(sch)
    kw["embedding"] = [0.0] * 384
    with pytest.raises(ValueError, match="embedding has dim"):
        sch.make_row(**kw)


def test_make_row_created_at_override_respected() -> None:
    _, sch = _reload_store()
    kw = _good_row_kwargs(sch)
    kw["created_at"] = 1_234_567
    row = sch.make_row(**kw)
    assert row["created_at"] == 1_234_567


# --- on-disk: connect + open_table --------------------------------------------


def test_connect_creates_data_dirs(
    isolated_data_dir: Path,
) -> None:
    _, sch = _reload_store()
    sch.connect()
    expected = isolated_data_dir / "lancedb"
    assert expected.is_dir()


def test_open_table_create_then_reopen_roundtrip(
    isolated_data_dir: Path,
) -> None:
    _, sch = _reload_store()
    t1 = sch.open_table()
    # round-trip: insert one row, reopen, count
    row = sch.make_row(**_good_row_kwargs(sch))
    t1.add([row])

    t2 = sch.open_table()
    df = t2.to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["id"] == "0123456789abcdef-text-0"
    assert df.iloc[0]["modality"] == "text"


# --- dedup cache (warm path for ``ingest_path`` scanning) ---------------------


def test_warm_dedup_cache_loads_doc_ids(isolated_data_dir: Path) -> None:
    """``warm_dedup_cache`` must actually populate the in-memory set —
    a quiet API regression here (e.g. an unsupported ``columns=``
    kwarg in a newer lancedb) drops every ingest run back to per-call
    LanceDB queries, which is the slow path users perceive as
    "scanning is taking forever".  Pin the happy path so we notice.
    """
    _, sch = _reload_store()
    table = sch.open_table()
    # Two distinct doc_ids → set of size 2 after warm.
    kw_a = _good_row_kwargs(sch)
    kw_b = _good_row_kwargs(sch)
    kw_b["doc_id"] = "fedcba9876543210"
    kw_b["chunk_idx"] = 0  # both first-chunk
    table.add([sch.make_row(**kw_a), sch.make_row(**kw_b)])

    n = sch.warm_dedup_cache(table=table)
    assert n == 2

    # Fast-path: ``is_doc_indexed`` now hits the in-memory set.
    assert sch.is_doc_indexed("0123456789abcdef", table=table) is True
    assert sch.is_doc_indexed("fedcba9876543210", table=table) is True
    assert sch.is_doc_indexed("0000000000000000", table=table) is False

    sch.clear_dedup_cache()


def test_note_indexed_keeps_warm_cache_consistent(
    isolated_data_dir: Path,
) -> None:
    """A row added *after* the warm pass must still register as
    indexed — otherwise two files with identical content in the same
    ``ingest_path`` run would both pay the embed cost."""
    _, sch = _reload_store()
    table = sch.open_table()
    table.add([sch.make_row(**_good_row_kwargs(sch))])

    sch.warm_dedup_cache(table=table)
    assert sch.is_doc_indexed("0123456789abcdef", table=table) is True

    # Simulate a newly-added doc within the same run.
    sch.note_indexed("aaaaaaaaaaaaaaaa")
    assert sch.is_doc_indexed("aaaaaaaaaaaaaaaa", table=table) is True

    sch.clear_dedup_cache()


def test_clear_dedup_cache_falls_back_to_query(
    isolated_data_dir: Path,
) -> None:
    """After ``clear_dedup_cache`` ``is_doc_indexed`` must go back to
    the per-call query path so a single-file ingest after a directory
    walk doesn't operate on stale doc_ids."""
    _, sch = _reload_store()
    table = sch.open_table()
    table.add([sch.make_row(**_good_row_kwargs(sch))])

    sch.warm_dedup_cache(table=table)
    sch.clear_dedup_cache()

    # The set is gone — fall back via LanceDB query, which still finds
    # the row that's actually in the table.
    assert sch.is_doc_indexed("0123456789abcdef", table=table) is True
    # Rows that were never added are still missing.
    assert sch.is_doc_indexed("0000000000000000", table=table) is False


# --- index_stats --------------------------------------------------------------


def test_index_stats_empty_table(isolated_data_dir: Path) -> None:
    """Fresh table → all zeros; every modality key still present so
    the SwiftUI Index tab can render a stable row order from cold."""
    _, sch = _reload_store()
    sch.open_table()  # materialize the empty table

    stats = sch.index_stats()
    assert stats["total_chunks"] == 0
    assert stats["total_docs"] == 0
    assert set(stats["by_modality"].keys()) == sch.MODALITIES
    assert all(v == 0 for v in stats["by_modality"].values())
    # Per-modality docs counter sits alongside chunk counts.
    # counter so the SwiftUI status bar can render "N items · M chunks".
    assert set(stats["by_modality_docs"].keys()) == sch.MODALITIES
    assert all(v == 0 for v in stats["by_modality_docs"].values())


def test_index_stats_distinct_docs_and_modality_counts(
    isolated_data_dir: Path,
) -> None:
    """Insert a hand-crafted mix and verify the totals roll up
    correctly: two text chunks of doc-A + one image of doc-B + one
    video of doc-C → 4 chunks, 3 distinct docs, modality counts
    {text: 2, image: 1, video: 1, ...rest 0}.

    Also verifies ``by_modality_docs`` — for
    text it should be 1 doc (both chunks share doc-A), so the
    chunks-vs-items distinction is observable end-to-end.
    """
    _, sch = _reload_store()
    table = sch.open_table()

    rows = [
        sch.make_row(
            doc_id="aaaaaaaaaaaaaaaa",
            modality="text",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/tmp/a.md",
            sha256="a" * 64,
            mtime=1_700_000_000,
            filename="a",
        ),
        sch.make_row(
            doc_id="aaaaaaaaaaaaaaaa",
            modality="text",
            chunk_idx=1,
            embedding=[0.0] * 768,
            file_path="/tmp/a.md",
            sha256="a" * 64,
            mtime=1_700_000_000,
            filename="a",
        ),
        sch.make_row(
            doc_id="bbbbbbbbbbbbbbbb",
            modality="image",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/tmp/b.jpg",
            sha256="b" * 64,
            mtime=1_700_000_000,
            filename="b",
        ),
        sch.make_row(
            doc_id="cccccccccccccccc",
            modality="video",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/tmp/c.mp4",
            sha256="c" * 64,
            mtime=1_700_000_000,
            filename="c",
        ),
    ]
    table.add(rows)

    stats = sch.index_stats()
    assert stats["total_chunks"] == 4
    assert stats["total_docs"] == 3
    assert stats["by_modality"]["text"] == 2
    assert stats["by_modality"]["image"] == 1
    assert stats["by_modality"]["video"] == 1
    # Untouched modalities still report zero rather than missing keys.
    assert stats["by_modality"]["rejected"] == 0

    # Items (distinct docs) per modality.
    # text has 2 chunks but only 1 doc (both rows share doc-A);
    # image and video each have 1 chunk → 1 doc.
    assert stats["by_modality_docs"]["text"] == 1
    assert stats["by_modality_docs"]["image"] == 1
    assert stats["by_modality_docs"]["video"] == 1
    assert stats["by_modality_docs"]["rejected"] == 0


def test_index_stats_path_prefix_filters_to_subset(
    isolated_data_dir: Path,
) -> None:
    """``path_prefix`` restricts counts to docs whose canonical first
    path starts with the prefix. v1.x relies on this so the
    Watched-folders sidebar can re-derive each folder's
    ``items · chunks`` line after a restart wipes the in-memory
    per-folder stats."""
    _, sch = _reload_store()
    table = sch.open_table()

    rows = [
        sch.make_row(
            doc_id="aaaaaaaaaaaaaaaa",
            modality="text",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/Users/me/notes/a.md",
            sha256="a" * 64,
            mtime=1_700_000_000,
            filename="a",
        ),
        sch.make_row(
            doc_id="bbbbbbbbbbbbbbbb",
            modality="text",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/Users/me/notes/b.md",
            sha256="b" * 64,
            mtime=1_700_000_000,
            filename="b",
        ),
        sch.make_row(
            doc_id="cccccccccccccccc",
            modality="image",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path="/Users/me/photos/c.jpg",
            sha256="c" * 64,
            mtime=1_700_000_000,
            filename="c",
        ),
    ]
    table.add(rows)

    # /Users/me/notes/ → 2 text docs.
    stats = sch.index_stats(path_prefix="/Users/me/notes")
    assert stats["total_chunks"] == 2
    assert stats["total_docs"] == 2
    assert stats["by_modality"]["text"] == 2
    assert stats["by_modality"]["image"] == 0
    assert stats["by_modality_docs"]["text"] == 2

    # /Users/me/photos/ → 1 image doc.
    stats = sch.index_stats(path_prefix="/Users/me/photos")
    assert stats["total_chunks"] == 1
    assert stats["total_docs"] == 1
    assert stats["by_modality"]["image"] == 1
    assert stats["by_modality"]["text"] == 0

    # Disjoint prefix → empty stats (but well-formed).
    stats = sch.index_stats(path_prefix="/somewhere/else")
    assert stats["total_chunks"] == 0
    assert stats["total_docs"] == 0
    assert stats["by_modality"]["text"] == 0


def test_index_stats_modality_keys_are_sorted(
    isolated_data_dir: Path,
) -> None:
    """SwiftUI re-renders this dict as a list — alphabetized order
    keeps the Settings → Index rows from jumping around between
    refreshes."""
    _, sch = _reload_store()
    sch.open_table()
    stats = sch.index_stats()
    keys = list(stats["by_modality"].keys())
    assert keys == sorted(keys)
    docs_keys = list(stats["by_modality_docs"].keys())
    assert docs_keys == sorted(docs_keys)


# --- delete_by_path_prefix ---------------------------------------------------


def test_delete_by_path_prefix_rejects_empty_or_root(
    isolated_data_dir: Path,
) -> None:
    """A 1-char prefix or ``/`` would wipe the entire index — refuse
    so a UI bug or rogue caller can't nuke everything by mistake."""
    _, sch = _reload_store()
    sch.open_table()
    for bad in ("", " ", "x", "/"):
        with pytest.raises(ValueError):
            sch.delete_by_path_prefix(bad)


def test_delete_by_path_prefix_removes_matching_rows_and_thumbnails(
    isolated_data_dir: Path,
    tmp_path: Path,
) -> None:
    """Insert text + image rows under two distinct prefixes, delete one,
    verify the other survives, and that the deleted image's thumbnail
    file on disk is unlinked."""
    _, sch = _reload_store()
    table = sch.open_table()

    # Materialize a fake thumbnail so the unlink branch exercises a
    # real file (not just a missing path).
    thumb = tmp_path / "thumb.webp"
    thumb.write_bytes(b"webp-bytes")
    assert thumb.is_file()

    rows = [
        sch.make_row(
            doc_id="aaaaaaaaaaaaaaaa",
            modality="text",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path=f"{tmp_path}/keep/note.md",
            sha256="a" * 64,
            mtime=1_700_000_000,
            filename="note",
        ),
        sch.make_row(
            doc_id="bbbbbbbbbbbbbbbb",
            modality="text",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path=f"{tmp_path}/demo/notes/x.md",
            sha256="b" * 64,
            mtime=1_700_000_000,
            filename="x",
        ),
        sch.make_row(
            doc_id="cccccccccccccccc",
            modality="image",
            chunk_idx=0,
            embedding=[0.0] * 768,
            file_path=f"{tmp_path}/demo/images/y.jpg",
            sha256="c" * 64,
            mtime=1_700_000_000,
            filename="y",
            thumbnail_path=str(thumb),
        ),
    ]
    table.add(rows)

    # Pass the same ``table`` handle so we observe the post-delete
    # state on the same snapshot. (LanceDB tables are versioned;
    # opening a fresh handle inside the helper would return a
    # different snapshot for our test-side reads.)
    result = sch.delete_by_path_prefix(f"{tmp_path}/demo", table=table)
    assert result["deleted_chunks"] == 2
    assert result["deleted_docs"] == 2
    assert result["deleted_thumbnails"] == 1
    assert not thumb.is_file()  # actually unlinked

    df = table.to_pandas()
    assert len(df) == 1
    assert df.iloc[0]["doc_id"] == "aaaaaaaaaaaaaaaa"


def test_delete_by_path_prefix_no_match_returns_zeros(
    isolated_data_dir: Path,
) -> None:
    """A prefix that matches nothing isn't an error — returns
    well-formed zeros so the SwiftUI UI can render a friendly
    "nothing to remove" message."""
    _, sch = _reload_store()
    table = sch.open_table()
    table.add(
        [
            sch.make_row(
                doc_id="dddddddddddddddd",
                modality="text",
                chunk_idx=0,
                embedding=[0.0] * 768,
                file_path="/Users/x/keep.md",
                sha256="d" * 64,
                mtime=1_700_000_000,
                filename="keep",
            ),
        ]
    )
    result = sch.delete_by_path_prefix("/no/such/folder/anywhere", table=table)
    assert result == {"deleted_chunks": 0, "deleted_docs": 0, "deleted_thumbnails": 0}
    assert table.count_rows() == 1


def test_delete_by_path_prefix_escapes_single_quote(
    isolated_data_dir: Path,
) -> None:
    """A folder named ``Bob's`` must not break the WHERE clause —
    same SQL-doubling trick as ``_build_where``."""
    _, sch = _reload_store()
    table = sch.open_table()
    table.add(
        [
            sch.make_row(
                doc_id="eeeeeeeeeeeeeeee",
                modality="text",
                chunk_idx=0,
                embedding=[0.0] * 768,
                file_path="/Users/Bob's Stuff/notes/a.md",
                sha256="e" * 64,
                mtime=1_700_000_000,
                filename="a",
            ),
        ]
    )
    result = sch.delete_by_path_prefix("/Users/Bob's Stuff", table=table)
    assert result["deleted_chunks"] == 1
    assert table.count_rows() == 0


def test_delete_index_by_prefix_endpoint_round_trip(
    isolated_data_dir: Path,
    tmp_path: Path,
) -> None:
    """``DELETE /index/by-path-prefix`` + a follow-up ``GET /index/stats``
    confirm the wiring (FastAPI route → helper → table) is intact."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _, sch = _reload_store()
    table = sch.open_table()
    table.add(
        [
            sch.make_row(
                doc_id="ffffffffffffffff",
                modality="text",
                chunk_idx=0,
                embedding=[0.0] * 768,
                file_path=f"{tmp_path}/demo/n.md",
                sha256="f" * 64,
                mtime=1_700_000_000,
                filename="n",
            ),
        ]
    )

    from rubick_backend.api import index as index_api

    app = FastAPI()
    app.include_router(index_api.router)

    with TestClient(app) as client:
        # Refusal path.
        r = client.delete("/index/by-path-prefix", params={"prefix": "/"})
        assert r.status_code == 400
        # Happy path.
        r = client.delete("/index/by-path-prefix", params={"prefix": f"{tmp_path}/demo"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["deleted_chunks"] == 1
        assert body["deleted_docs"] == 1

        # Verify via the stats endpoint.
        r = client.get("/index/stats")
        assert r.json()["total_chunks"] == 0


def test_get_index_stats_endpoint_returns_payload(
    isolated_data_dir: Path,
) -> None:
    """``GET /index/stats`` round-trip through FastAPI returns the
    same shape ``index_stats()`` produces; smoke that the route is
    wired up at the correct URL."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _, sch = _reload_store()
    table = sch.open_table()
    table.add(
        [
            sch.make_row(
                doc_id="dddddddddddddddd",
                modality="image",
                chunk_idx=0,
                embedding=[0.0] * 768,
                file_path="/tmp/d.jpg",
                sha256="d" * 64,
                mtime=1_700_000_000,
                filename="d",
            ),
        ]
    )

    from rubick_backend.api import index as index_api

    app = FastAPI()
    app.include_router(index_api.router)

    with TestClient(app) as client:
        r = client.get("/index/stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_chunks"] == 1
    assert body["total_docs"] == 1
    assert body["by_modality"]["image"] == 1
    assert body["by_modality"]["text"] == 0
