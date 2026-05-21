"""Unit tests for the text chunker (``ingest/text.py``).

These tests intentionally **never load the embedding model**: any path
that would call ``_count_tokens`` is fed via the ``patch_token_counter``
fixture below, which monkeypatches the tokenizer call with a cheap
char-based approximation. This keeps the suite fast on cold caches.
"""

from __future__ import annotations

import pytest

from rubick_backend import settings as backend_settings
from rubick_backend.ingest import text as text_mod
from rubick_backend.ingest.text import (
    SHORT_CHAR_THRESHOLD,
    chunk_text,
)


@pytest.fixture(autouse=True)
def patch_token_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``_count_tokens`` with a char/4 approximation.

    The real implementation calls ``embed.load()`` which downloads /
    loads 1.8 GB of weights — fine for end-to-end smokes, far too heavy
    for chunker unit tests. char/4 is a well-known cheap proxy for BPE
    token counts and is plenty for boundary-condition testing.
    """
    monkeypatch.setattr(text_mod, "_count_tokens", lambda s: max(1, len(s) // 4))


# --- short-circuit path (< 1000 chars → single chunk, no tokenizer) -----------


def test_short_text_returns_single_chunk_stripped() -> None:
    raw = "   hello world\n\nthis is short   "
    out = chunk_text(raw, is_markdown=False)
    assert out == ["hello world\n\nthis is short"]
    assert len(out) == 1


def test_short_markdown_also_short_circuits() -> None:
    md = "# header\n\nbody"
    assert chunk_text(md, is_markdown=True) == ["# header\n\nbody"]


# --- markdown chunking --------------------------------------------------------


def _make_section(heading: str, body_chars: int) -> str:
    """A markdown heading + a body of ``body_chars`` characters."""
    body = "x" * body_chars
    return f"# {heading}\n\n{body}"


def test_markdown_splits_on_h1_boundaries() -> None:
    # Each section must exceed the default target (2048 tokens ≈ 8k chars)
    # so three H1 blocks become three chunks.
    md = "\n\n".join(_make_section(f"S{i}", 8500) for i in range(3))
    assert len(md) >= SHORT_CHAR_THRESHOLD  # confirm we leave the short path
    chunks = chunk_text(md, is_markdown=True)
    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        assert c.startswith(f"# S{i}")


def test_markdown_h2_does_not_split() -> None:
    """Only top-level ``#`` boundaries split; ``##`` stays inside a chunk."""
    md = _make_section("Top", 100) + "\n\n## Sub\n\n" + "y" * 1500
    chunks = chunk_text(md, is_markdown=True)
    assert len(chunks) == 1
    assert "## Sub" in chunks[0]


def test_markdown_fenced_hash_is_not_a_heading() -> None:
    """``# foo`` inside a fenced code block must not split."""
    md = (
        "# Real heading\n\n"
        + "z" * 800
        + "\n\n```python\n# this is a comment\nprint(1)\n```\n\n"
        + "more body " * 50
    )
    chunks = chunk_text(md, is_markdown=True)
    assert len(chunks) == 1
    assert "# this is a comment" in chunks[0]


def test_markdown_greedy_packs_small_sections_together() -> None:
    """Several tiny sections should pack into one chunk under TARGET_TOKENS."""
    md = "\n\n".join(_make_section(f"S{i}", 200) for i in range(8))
    assert len(md) >= SHORT_CHAR_THRESHOLD
    chunks = chunk_text(md, is_markdown=True)
    assert len(chunks) <= 3
    assert all(c.startswith("# S") for c in chunks)


# --- plaintext chunking -------------------------------------------------------


def test_plaintext_paragraph_boundary() -> None:
    paragraphs = ["para " * 80 + str(i) for i in range(6)]
    raw = "\n\n".join(paragraphs)
    chunks = chunk_text(raw, is_markdown=False)
    assert len(chunks) >= 1
    rejoined = "\n\n".join(chunks)
    needle = "para " * 10
    for i in range(6):
        assert needle in rejoined or str(i) in rejoined


def test_plaintext_drops_blank_paragraphs() -> None:
    raw = "hello world.\n\n\n\nworld hello.\n\n\n\n" * 60
    chunks = chunk_text(raw, is_markdown=False)
    text_blob = "\n\n".join(chunks)
    assert "\n\n\n" not in text_blob


# --- hard-cap backstop --------------------------------------------------------


def test_hard_max_tokens_caps_runaway_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single oversize block must still be emitted (no infinite buffer)."""
    huge = "# Solo\n\n" + ("x" * 16000)
    # Override the counter so this block clearly exceeds HARD_MAX_TOKENS.
    monkeypatch.setattr(text_mod, "_count_tokens", lambda s: len(s))
    chunks = chunk_text(huge, is_markdown=True)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Solo")
    assert backend_settings.TARGET_TOKENS < backend_settings.HARD_MAX_TOKENS


# --- whitespace robustness ----------------------------------------------------


def test_empty_string_short_circuit() -> None:
    assert chunk_text("", is_markdown=False) == [""]
    assert chunk_text("    \n  \n", is_markdown=True) == [""]
