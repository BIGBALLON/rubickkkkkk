"""Unit tests for the WHERE-clause builder used by ``hybrid_search``.

Pure-Python — no LanceDB, no model. End-to-end coverage (a real
filtered search against a populated index) lives in ``test_smoke_e2e.py``
behind ``@slow``.
"""

from __future__ import annotations

import pytest

from rubick_backend.retrieve.hybrid import _build_where

# === Defaults / no-op cases =================================================


def test_no_filters_returns_none() -> None:
    """No ``modality``, no path/time, ``include_rejected=True`` →
    truly empty WHERE; the caller should pass ``where=None`` to LanceDB.
    """
    assert _build_where(modality=None, include_rejected=True) is None


def test_default_hides_rejected() -> None:
    """``include_rejected=False`` (the default) injects the
    ``modality != 'rejected'`` guard even when no other filter is
    supplied — UI never wants to surface rejected placeholders."""
    where = _build_where(modality=None, include_rejected=False)
    assert where == "modality != 'rejected'"


# === Modality ==============================================================


def test_modality_exact_match() -> None:
    where = _build_where(modality="image", include_rejected=True)
    assert where == "modality = 'image'"


def test_modality_with_underscore_passes_sanitizer() -> None:
    """``audio_transcript`` / ``video_transcript`` (and ``audio``
    itself, retired with the audio modality removal) are legacy values
    that may still appear in old indexes; the sanitizer must accept
    underscores so a forensic query against a pre-removal database
    still composes without raising."""
    where = _build_where(modality="audio_transcript", include_rejected=True)
    assert where == "modality = 'audio_transcript'"


def test_modality_rejects_sql_injection() -> None:
    """Anything that's not ``[a-zA-Z0-9_]+`` is refused. We sanitize
    here even though the FastAPI layer is also typed, defense in
    depth."""
    with pytest.raises(ValueError, match="illegal modality"):
        _build_where(modality="text' OR '1'='1", include_rejected=True)


def test_modality_comma_separated_emits_in_clause() -> None:
    """Multi-select sidebar facets want ``image OR video`` in one round-trip."""
    where = _build_where(modality="image,video", include_rejected=True)
    assert where == "modality IN ('image', 'video')"


def test_modality_comma_separated_with_legacy_values() -> None:
    """Backward-compat for old indexes: pre-removal databases may
    contain ``audio`` rows, and legacy databases may contain
    ``audio_transcript`` / ``video_transcript`` rows. The builder
    must handle 3+ tokens cleanly so a forensic search across all
    legacy modality strings still composes (new ingest never
    produces these — see ``store/schema.py`` MODALITIES)."""
    where = _build_where(
        modality="audio,audio_transcript,video,video_transcript",
        include_rejected=True,
    )
    assert where == ("modality IN ('audio', 'audio_transcript', 'video', 'video_transcript')")


@pytest.mark.parametrize(
    "raw",
    [
        " image , video ",  # surrounding + interior spaces
        "image,,video,",  # repeated / trailing commas
        " image , , video ",  # both
    ],
)
def test_modality_normalises_messy_csv(raw: str) -> None:
    """Whitespace + empty tokens get stripped before sanitisation, so
    a user composing a filter URL by hand still gets the intended
    ``IN`` clause."""
    where = _build_where(modality=raw, include_rejected=True)
    assert where == "modality IN ('image', 'video')"


def test_modality_all_empty_tokens_raises() -> None:
    """A filter that's entirely commas / whitespace is a caller bug —
    we'd rather raise than silently drop the filter and return
    everything."""
    with pytest.raises(ValueError, match="empty modality"):
        _build_where(modality=" , , ", include_rejected=True)


def test_modality_in_clause_still_sanitizes_each_token() -> None:
    """Per-token sanitizer applies even inside a comma-separated list
    — one bad apple should reject the whole filter."""
    with pytest.raises(ValueError, match="illegal modality"):
        _build_where(modality="image,video' OR 1=1 --", include_rejected=True)


# === Path prefix ============================================================


def test_path_prefix_basic() -> None:
    where = _build_where(
        modality=None,
        include_rejected=True,
        path_prefix="/Users/x/notes",
    )
    assert where == "starts_with(file_paths[1], '/Users/x/notes')"


def test_path_prefix_escapes_single_quote() -> None:
    """Single quotes get doubled per SQL standard. Without this a
    folder name like ``Bob's Stuff`` would terminate the literal early
    and turn the rest into syntax errors (or worse)."""
    where = _build_where(
        modality=None,
        include_rejected=True,
        path_prefix="/Users/Bob's/notes",
    )
    assert where == "starts_with(file_paths[1], '/Users/Bob''s/notes')"


def test_path_prefix_empty_string_is_ignored() -> None:
    """An empty / falsy ``path_prefix`` should NOT emit a clause —
    otherwise we'd inject ``starts_with(file_paths[1], '')`` which
    matches everything but adds runtime cost."""
    where = _build_where(
        modality=None,
        include_rejected=True,
        path_prefix="",
    )
    assert where is None


# === Time range =============================================================


def test_mtime_after_only() -> None:
    where = _build_where(
        modality=None,
        include_rejected=True,
        mtime_after=1_700_000_000,
    )
    assert where == "mtime >= 1700000000"


def test_mtime_before_only() -> None:
    where = _build_where(
        modality=None,
        include_rejected=True,
        mtime_before=1_800_000_000,
    )
    assert where == "mtime <= 1800000000"


def test_mtime_range_both_sides() -> None:
    where = _build_where(
        modality=None,
        include_rejected=True,
        mtime_after=1_700_000_000,
        mtime_before=1_800_000_000,
    )
    # Order is fixed by ``_build_where`` insertion order — guard
    # against it silently flipping (humans memorize WHERE clauses).
    assert where == "mtime >= 1700000000 AND mtime <= 1800000000"


def test_mtime_zero_is_legal_lower_bound() -> None:
    """Epoch 0 is "1970-01-01"; legitimate as a "no lower bound" knob
    for callers that always pass both ends. We must not reject it."""
    where = _build_where(
        modality=None,
        include_rejected=True,
        mtime_after=0,
    )
    assert where == "mtime >= 0"


def test_negative_mtime_after_raises() -> None:
    with pytest.raises(ValueError, match="mtime_after must be >= 0"):
        _build_where(modality=None, include_rejected=True, mtime_after=-1)


def test_negative_mtime_before_raises() -> None:
    with pytest.raises(ValueError, match="mtime_before must be >= 0"):
        _build_where(modality=None, include_rejected=True, mtime_before=-1)


def test_inverted_mtime_range_raises() -> None:
    """Catch the obvious user error of swapping the two ends —
    LanceDB would happily run ``mtime >= 1000 AND mtime <= 500`` and
    return zero rows; that's a confusing failure mode."""
    with pytest.raises(ValueError, match="must be >= mtime_after"):
        _build_where(
            modality=None,
            include_rejected=True,
            mtime_after=1_800_000_000,
            mtime_before=1_700_000_000,
        )


# === Composition =============================================================


def test_all_filters_compose_with_and() -> None:
    """Realistic worst case: every knob set. Verifies the order is
    stable so logged WHERE clauses are diffable across versions, and
    that all clauses are joined by `` AND ``."""
    where = _build_where(
        modality="image",
        include_rejected=False,
        path_prefix="/Users/x/Pictures",
        mtime_after=1_700_000_000,
        mtime_before=1_800_000_000,
    )
    assert where == (
        "modality = 'image' AND modality != 'rejected' AND "
        "starts_with(file_paths[1], '/Users/x/Pictures') AND "
        "mtime >= 1700000000 AND mtime <= 1800000000"
    )


def test_modality_and_rejected_default_compose() -> None:
    """``include_rejected=False`` always adds its guard, even when
    ``modality`` is set. (We could optimize away the second clause
    when ``modality != 'rejected'``, but the redundancy is harmless
    and keeps the builder dumb.)"""
    where = _build_where(modality="text", include_rejected=False)
    assert where == "modality = 'text' AND modality != 'rejected'"
