"""Tests for the text ingest pipeline (``ingest/text.py``).

Walks and dispatching are now owned by ``ingest/__init__.py`` and have
their own test file (``test_ingest_dispatch.py``). What's left here
are the parts of ``ingest_file`` that operate purely on the filesystem
and on the chunker output — no model load required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rubick_backend.ingest import text as text_mod
from rubick_backend.ingest.text import (
    SUPPORTED_EXTENSIONS,
    _read_text,
    ingest_file,
)

# --- _read_text encoding fallback --------------------------------------------


def test_read_text_strips_utf8_bom(tmp_path: Path) -> None:
    p = tmp_path / "bom.md"
    p.write_bytes(b"\xef\xbb\xbf# hello")
    assert _read_text(p) == "# hello"


def test_read_text_falls_back_to_latin1(tmp_path: Path) -> None:
    p = tmp_path / "latin1.txt"
    p.write_bytes(b"caf\xe9")  # 'café' in latin-1, invalid utf-8
    out = _read_text(p)
    assert out is not None
    assert "caf" in out


def test_read_text_handles_pure_utf8(tmp_path: Path) -> None:
    p = tmp_path / "utf8.md"
    p.write_text("hello 你好 🌍", encoding="utf-8")
    assert _read_text(p) == "hello 你好 🌍"


# --- ingest_file skip rules ---------------------------------------------------


def test_ingest_file_skips_unsupported_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.pdf"
    p.write_text("pretend this is a pdf")
    assert ingest_file(p) == []


def test_ingest_file_skips_oversize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_mod, "MAX_FILE_BYTES", 16)
    p = tmp_path / "big.md"
    p.write_text("x" * 1000)
    assert ingest_file(p) == []


def test_ingest_file_skips_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.md"
    p.write_text("   \n\n   \n")
    assert ingest_file(p) == []


def test_ingest_file_skips_directory(tmp_path: Path) -> None:
    d = tmp_path / "subdir"
    d.mkdir()
    assert ingest_file(d) == []


def test_supported_extensions_covers_spec_set() -> None:
    """v1 text inputs: md / markdown / txt / org."""
    assert SUPPORTED_EXTENSIONS == frozenset({".md", ".markdown", ".txt", ".org"})
