"""Unit tests for user-editable chunking settings.

Two surfaces under test:

- ``rubick_backend.settings`` runtime API
  (``get_chunking_settings`` / ``update_chunking_settings`` /
  ``_resolve_initial_chunking``)
- ``GET /settings`` + ``PATCH /settings`` HTTP routes
  (``rubick_backend.api.settings``)

The settings module reads ``settings.json`` + env vars at import
time, so we use ``importlib.reload`` to re-bind the module under a
freshly-isolated ``RUBICK_DATA_DIR`` per test. The autouse
``isolated_data_dir`` fixture in ``conftest.py`` already injects
the env var; we just need the reload to pick it up.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _reload_settings_module():
    """Force a fresh import of ``rubick_backend.settings`` so the
    module-level chunking values reflect the current env / file
    state. Returns the reloaded module so the caller can poke at
    ``TARGET_TOKENS`` directly.
    """
    import rubick_backend.settings as settings_mod  # noqa: F401

    if "rubick_backend.settings" in sys.modules:
        return importlib.reload(sys.modules["rubick_backend.settings"])
    return settings_mod


# === Module-level resolution chain (file > env > default) =====================


def test_defaults_when_no_overrides(
    isolated_data_dir: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh data dir + clean env → fall back to compile-time defaults."""
    monkeypatch.delenv("RUBICK_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == sm._DEFAULT_TARGET_TOKENS
    assert sm.HARD_MAX_TOKENS == sm._DEFAULT_HARD_MAX_TOKENS


def test_env_var_overrides_defaults(
    isolated_data_dir: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env vars beat defaults when ``settings.json`` is absent."""
    monkeypatch.setenv("RUBICK_TARGET_TOKENS", "1024")
    monkeypatch.setenv("RUBICK_HARD_MAX_TOKENS", "3072")
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == 1024
    assert sm.HARD_MAX_TOKENS == 3072


def test_settings_file_overrides_env_and_defaults(
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``settings.json`` on disk wins the resolution chain."""
    monkeypatch.setenv("RUBICK_TARGET_TOKENS", "1024")
    monkeypatch.setenv("RUBICK_HARD_MAX_TOKENS", "3072")
    (isolated_data_dir / "settings.json").write_text(
        json.dumps({"target_tokens": 2048, "hard_max_tokens": 4096})
    )
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == 2048
    assert sm.HARD_MAX_TOKENS == 4096


def test_invalid_env_var_falls_back_to_default(
    isolated_data_dir: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd env var (``RUBICK_TARGET_TOKENS=foo``) shouldn't take
    the backend down — it should log + fall back."""
    monkeypatch.setenv("RUBICK_TARGET_TOKENS", "not-a-number")
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == sm._DEFAULT_TARGET_TOKENS


def test_malformed_settings_file_falls_back(
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted ``settings.json`` shouldn't crash the boot path."""
    monkeypatch.delenv("RUBICK_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    (isolated_data_dir / "settings.json").write_text("{not valid json")
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == sm._DEFAULT_TARGET_TOKENS


# === Validation =============================================================


@pytest.mark.parametrize(
    "raw,bound_idx",
    [
        (99_999, 1),  # above max → clamps up
        (0, 0),  # below min → clamps down
    ],
)
def test_update_clamps_out_of_range(
    isolated_data_dir: Path,  # noqa: ARG001
    raw: int,
    bound_idx: int,
) -> None:
    """Out-of-range target tokens silently clamp to the nearest bound;
    no exception. The Swift slider sends raw ints so hardcoding a
    raise here would surface a UX bug as a 500."""
    sm = _reload_settings_module()
    result = sm.update_chunking_settings(target_tokens=raw, persist=False)
    assert result["target_tokens"] == sm._TARGET_TOKENS_BOUNDS[bound_idx]


def test_update_bumps_hard_max_when_below_target(
    isolated_data_dir: Path,  # noqa: ARG001
) -> None:
    """Backwards pair (hard_max < target) bumps hard_max up to target
    instead of accepting an unworkable config."""
    sm = _reload_settings_module()
    result = sm.update_chunking_settings(
        target_tokens=2000, hard_max_tokens=500, persist=False
    )
    assert result["target_tokens"] == 2000
    assert result["hard_max_tokens"] == 2000


def test_update_one_field_keeps_other(
    isolated_data_dir: Path,  # noqa: ARG001
) -> None:
    """``target_tokens=None`` means ``leave it alone`` — required for
    the Swift client to flip just one knob."""
    sm = _reload_settings_module()
    sm.reset_chunking_for_tests()
    sm.update_chunking_settings(
        target_tokens=1500, hard_max_tokens=4000, persist=False
    )
    sm.update_chunking_settings(target_tokens=800, persist=False)
    assert sm.TARGET_TOKENS == 800
    assert sm.HARD_MAX_TOKENS == 4000


# === Persistence ============================================================


def test_update_writes_to_settings_json(
    isolated_data_dir: Path,
) -> None:
    """``persist=True`` (the default) writes the file atomically."""
    sm = _reload_settings_module()
    sm.update_chunking_settings(target_tokens=1024, hard_max_tokens=3072)
    on_disk = json.loads((isolated_data_dir / "settings.json").read_text())
    assert on_disk == {
        "target_tokens": 1024,
        "hard_max_tokens": 3072,
        "exclusion_patterns": [],
    }


def test_settings_survive_module_reload(
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence's payoff — the next boot picks up the user's choice."""
    monkeypatch.delenv("RUBICK_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    sm = _reload_settings_module()
    sm.update_chunking_settings(target_tokens=1500, hard_max_tokens=3500)
    sm2 = _reload_settings_module()
    assert sm2.TARGET_TOKENS == 1500
    assert sm2.HARD_MAX_TOKENS == 3500


# === HTTP surface ===========================================================


@pytest.fixture()
def client(
    isolated_data_dir: Path,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Bare FastAPI app mounting only the settings router. Avoids
    spinning up the JobQueue lifespan, which is irrelevant here."""
    monkeypatch.delenv("RUBICK_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    _reload_settings_module()
    # Reload the router so its module-level binding picks up the
    # freshly-reloaded settings module.
    import rubick_backend.api.settings as settings_route_mod

    importlib.reload(settings_route_mod)
    app = FastAPI()
    app.include_router(settings_route_mod.router)
    return TestClient(app)


def test_get_settings_returns_metadata_envelope(client: TestClient) -> None:
    """GET /settings returns current values + defaults + bounds so the
    Swift UI can render preset cards without a second round-trip."""
    r = client.get("/settings")
    assert r.status_code == 200
    body = r.json()
    assert "target_tokens" in body
    assert "hard_max_tokens" in body
    assert body["defaults"] == {"target_tokens": 2048, "hard_max_tokens": 6144}
    bounds = body["bounds"]
    assert bounds["target_tokens"] == [100, 8192]
    assert bounds["hard_max_tokens"] == [200, 8192]


def test_patch_settings_round_trips(client: TestClient) -> None:
    """PATCH writes through to settings + persists; subsequent GET
    reflects the new state."""
    r1 = client.patch(
        "/settings",
        json={"target_tokens": 1024, "hard_max_tokens": 3072},
    )
    assert r1.status_code == 200
    assert r1.json() == {
        "target_tokens": 1024,
        "hard_max_tokens": 3072,
        "exclusion_patterns": [],
    }

    r2 = client.get("/settings")
    body = r2.json()
    assert body["target_tokens"] == 1024
    assert body["hard_max_tokens"] == 3072


def test_patch_only_one_field(client: TestClient) -> None:
    """Optional pydantic fields → caller can flip just target."""
    client.patch(
        "/settings",
        json={"target_tokens": 1024, "hard_max_tokens": 3000},
    )
    r = client.patch("/settings", json={"target_tokens": 800})
    assert r.status_code == 200
    assert r.json() == {
        "target_tokens": 800,
        "hard_max_tokens": 3000,
        "exclusion_patterns": [],
    }


def test_patch_clamps_out_of_range(client: TestClient) -> None:
    """The HTTP layer surfaces the clamped value; no 4xx for an
    over-eager request."""
    r = client.patch("/settings", json={"target_tokens": 999_999})
    assert r.status_code == 200
    assert r.json()["target_tokens"] == 8192


# === Exclusion patterns (v1.x #3) ===========================================


def test_exclusion_patterns_default_empty(
    isolated_data_dir: Path,  # noqa: ARG001
) -> None:
    """A fresh data dir surfaces no user-defined exclusions — the
    default deny-list is separate and stays opaque to this field."""
    sm = _reload_settings_module()
    assert sm.EXCLUSION_PATTERNS == []


def test_sanitize_drops_invalid_entries() -> None:
    """The sanitiser drops non-strings, empty / whitespace-only, and
    overlong entries; coalesces duplicates; trims to the cap."""
    sm = _reload_settings_module()
    raw: list[object] = [
        "  *.tmp  ",       # whitespace-stripped
        "",                # empty → drop
        "   ",             # whitespace → drop
        "*.tmp",           # dup of #1 → drop
        42,                # non-string → drop
        "x" * 300,         # overlong → drop
        "secrets",
    ]
    clean = sm._sanitize_exclusion_patterns(raw)
    assert clean == ["*.tmp", "secrets"]


def test_sanitize_trims_to_max_count() -> None:
    sm = _reload_settings_module()
    raw = [f"pat-{i}" for i in range(sm._MAX_EXCLUSION_PATTERNS + 10)]
    clean = sm._sanitize_exclusion_patterns(raw)
    assert len(clean) == sm._MAX_EXCLUSION_PATTERNS
    assert clean[0] == "pat-0"


def test_update_persists_exclusion_patterns(
    isolated_data_dir: Path,
) -> None:
    """``update_chunking_settings`` round-trips exclusion patterns
    through settings.json."""
    sm = _reload_settings_module()
    sm.update_chunking_settings(
        exclusion_patterns=["*.tmp", "secrets"], persist=True
    )
    on_disk = json.loads((isolated_data_dir / "settings.json").read_text())
    assert on_disk["exclusion_patterns"] == ["*.tmp", "secrets"]

    sm2 = _reload_settings_module()
    assert sm2.EXCLUSION_PATTERNS == ["*.tmp", "secrets"]


def test_update_empty_list_clears_patterns(
    isolated_data_dir: Path,  # noqa: ARG001
) -> None:
    """Sending ``[]`` is the explicit "clear all rules" signal —
    distinct from ``None`` (which means "leave untouched")."""
    sm = _reload_settings_module()
    sm.update_chunking_settings(exclusion_patterns=["foo"], persist=False)
    assert sm.EXCLUSION_PATTERNS == ["foo"]
    sm.update_chunking_settings(exclusion_patterns=[], persist=False)
    assert sm.EXCLUSION_PATTERNS == []


def test_update_none_leaves_patterns_alone(
    isolated_data_dir: Path,  # noqa: ARG001
) -> None:
    sm = _reload_settings_module()
    sm.update_chunking_settings(exclusion_patterns=["foo"], persist=False)
    sm.update_chunking_settings(target_tokens=900, persist=False)
    assert sm.EXCLUSION_PATTERNS == ["foo"]


def test_metadata_includes_default_dir_names_and_limits(
    client: TestClient,
) -> None:
    """The Privacy UI needs both the always-on dir names AND the
    pattern-list limits so it can render the "always excluded" list
    plus enforce the same cap as the backend."""
    r = client.get("/settings")
    body = r.json()
    assert "default_exclusion_dir_names" in body
    assert "node_modules" in body["default_exclusion_dir_names"]
    limits = body["exclusion_pattern_limits"]
    assert limits["max_count"] > 0
    assert limits["max_length"] > 0


def test_patch_round_trips_exclusion_patterns(client: TestClient) -> None:
    r = client.patch(
        "/settings",
        json={"exclusion_patterns": ["*.log", "scratch", "*.log"]},
    )
    assert r.status_code == 200
    # Dedup → 2 entries; order preserved.
    assert r.json()["exclusion_patterns"] == ["*.log", "scratch"]

    r2 = client.get("/settings")
    assert r2.json()["exclusion_patterns"] == ["*.log", "scratch"]


def test_settings_file_malformed_exclusion_list_falls_back(
    isolated_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken patterns list in settings.json shouldn't take the
    backend down — bad fields drop to ``[]`` while the rest of the
    settings still load."""
    monkeypatch.delenv("RUBICK_TARGET_TOKENS", raising=False)
    monkeypatch.delenv("RUBICK_HARD_MAX_TOKENS", raising=False)
    (isolated_data_dir / "settings.json").write_text(
        json.dumps(
            {
                "target_tokens": 800,
                "hard_max_tokens": 2400,
                "exclusion_patterns": "not-a-list",
            }
        )
    )
    sm = _reload_settings_module()
    assert sm.TARGET_TOKENS == 800
    assert sm.EXCLUSION_PATTERNS == []
