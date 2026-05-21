"""Unit tests for ``rubick_backend.__main__._parse_mtime_arg``.

The CLI accepts either a POSIX epoch second integer (``--mtime-after
1700000000``) or a calendar date in ``YYYY-MM-DD`` form (``--mtime-after
2024-01-01``), so the user doesn't need to think in epochs for the
common "files since Jan 1" case.

Calendar dates are interpreted in the *local* timezone — that's what
users mean by "after Jan 1" without reading a tz table. The actual
epoch this maps to therefore depends on the runner's TZ; tests assert
the parser's promise (start-of-day local) instead of pinning a magic
number.
"""

from __future__ import annotations

import argparse
import datetime

import pytest

from rubick_backend.__main__ import _parse_mtime_arg


def test_epoch_passthrough() -> None:
    """Bare digits round-trip as an int."""
    assert _parse_mtime_arg("1700000000") == 1_700_000_000


def test_zero_is_legal() -> None:
    """Epoch 0 (1970-01-01 UTC) is a legitimate "no lower bound"."""
    assert _parse_mtime_arg("0") == 0


def test_yyyy_mm_dd_maps_to_local_midnight() -> None:
    """Calendar dates snap to start-of-day in the **local** timezone.
    We compare against ``datetime`` itself rather than a hardcoded
    epoch so the test passes regardless of where CI runs.
    """
    parsed = _parse_mtime_arg("2024-01-01")
    expected = int(datetime.datetime(2024, 1, 1).timestamp())
    assert parsed == expected


def test_yyyy_mm_dd_strips_whitespace() -> None:
    """A trailing newline / leading space (e.g. shell expansion of a
    file containing the date) should be tolerated."""
    parsed = _parse_mtime_arg("  2024-06-15  ")
    expected = int(datetime.datetime(2024, 6, 15).timestamp())
    assert parsed == expected


def test_invalid_string_raises_argparse_error() -> None:
    """Random garbage produces an ``argparse.ArgumentTypeError`` — that's
    the contract argparse expects from custom ``type=`` callables; it
    formats the message as part of the standard usage error."""
    with pytest.raises(argparse.ArgumentTypeError, match="POSIX epoch seconds or YYYY-MM-DD"):
        _parse_mtime_arg("January 1st")


def test_invalid_date_raises_argparse_error() -> None:
    """Well-formed but impossible dates (Feb 30) also raise."""
    with pytest.raises(argparse.ArgumentTypeError, match="POSIX epoch seconds or YYYY-MM-DD"):
        _parse_mtime_arg("2024-02-30")
