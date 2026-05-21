"""Shared pytest fixtures for the rubick-backend test suite.

Two big concerns the fixtures here address:

1. **Don't touch the user's real Library** — every test that materializes
   on-disk state (LanceDB / models / thumbnails) gets a fresh ``tmp_path``
   wired in via ``RUBICK_DATA_DIR``, and we reset the cached module-level
   path constants in ``rubick_backend.settings`` to point at it.

2. **Don't load the 1.8 GB embedding model** unless the test explicitly
   opts in (e.g. via the ``slow`` mark). Loader access is funneled
   through ``embed/__init__.py``, so tests that only exercise pure
   helpers (chunker, schema) can run in <1 s on a cold cache.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks used in this suite."""
    config.addinivalue_line(
        "markers",
        "slow: end-to-end smokes that load the 1.8 GB embedding model "
        "(skipped unless RUBICK_RUN_SLOW=1 is set).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.slow`` tests unless ``RUBICK_RUN_SLOW=1``."""
    if os.environ.get("RUBICK_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set RUBICK_RUN_SLOW=1 to run slow smokes")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``settings.DATA_ROOT`` (and friends) at ``tmp_path`` for the test.

    Reloads ``rubick_backend.settings`` so the module-level ``DATA_ROOT`` /
    ``LANCEDB_DIR`` / ... constants pick up the new env var. The store
    package re-imports them lazily via ``rubick_backend.settings``, so
    reloading ``settings`` alone is enough.
    """
    monkeypatch.setenv("RUBICK_DATA_DIR", str(tmp_path))
    import rubick_backend.settings as s

    importlib.reload(s)
    yield tmp_path
    # No teardown needed — tmp_path is auto-cleaned and the next test
    # gets its own fixture invocation with a fresh reload.
