"""Unit + API tests for ``rubick_backend.permissions`` and
``GET /healthz/permissions`` (v1.x #2).

The probe path is parameterised so we drive every branch
(granted / denied / absent / non-Darwin) under ``tmp_path`` without
touching the real ``/Library/Application Support/com.apple.TCC/``.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rubick_backend.api.healthz import router
from rubick_backend.permissions import (
    FullDiskAccessProbe,
    probe_full_disk_access,
)

# === Helpers ===============================================================


def _force_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``probe_full_disk_access`` take the Darwin branch even on a
    Linux CI runner. We patch ``platform.system`` at the module the
    probe imports it through, not the global, so the test doesn't
    affect any other code.
    """
    monkeypatch.setattr(
        "rubick_backend.permissions.platform.system", lambda: "Darwin"
    )


# === Module-level probe tests =============================================


def test_probe_returns_granted_when_path_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_darwin(monkeypatch)
    probe_file = tmp_path / "TCC.db"
    probe_file.write_bytes(b"fake db bytes")

    got = probe_full_disk_access(str(probe_file))

    assert got.granted is True
    assert got.probe_error is None
    assert got.probe_path == str(probe_file)
    assert got.platform == "Darwin"


def test_probe_returns_denied_with_error_text_on_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PermissionError`` (TCC denial) → ``granted=False`` + verbatim
    OS error in ``probe_error`` so the UI can show the user what we
    tried."""
    _force_darwin(monkeypatch)
    probe_file = tmp_path / "TCC.db"
    probe_file.write_bytes(b"x")
    os.chmod(probe_file, 0o000)  # no-permission for anyone
    try:
        # On macOS or Linux non-root this open will EACCES. The CI
        # runner is non-root so we're safe; skip if somehow root.
        if os.geteuid() == 0:
            pytest.skip("running as root — chmod 000 is ignored")
        got = probe_full_disk_access(str(probe_file))
        assert got.granted is False
        assert got.probe_error is not None
        # ``Permission denied`` is the POSIX phrasing; we keep the test
        # tolerant in case future Python phrasing changes.
        assert "denied" in got.probe_error.lower() or "permission" in got.probe_error.lower()
    finally:
        # Restore perms so pytest tmp_path cleanup can rmtree it.
        os.chmod(probe_file, 0o600)


def test_probe_handles_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-existent probe path is reported as denied + a clear
    ``probe path missing`` message — neither a crash nor a misleading
    "granted=True". Useful on stripped macOS installs or if Apple ever
    moves the canary file."""
    _force_darwin(monkeypatch)
    nonexistent = tmp_path / "does-not-exist" / "TCC.db"
    got = probe_full_disk_access(str(nonexistent))

    assert got.granted is False
    assert got.probe_error is not None
    assert "missing" in got.probe_error.lower()
    assert got.platform == "Darwin"


def test_probe_returns_not_applicable_on_non_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux dev VMs / CI runners should get a clean "n/a" rather than
    a misleading "denied" — the UI hides the section entirely when
    ``platform != "Darwin"``."""
    monkeypatch.setattr(
        "rubick_backend.permissions.platform.system", lambda: "Linux"
    )
    got = probe_full_disk_access("/nonexistent/probe.db")
    assert got.granted is False
    assert got.platform == "Linux"
    assert got.probe_error is not None
    assert "not applicable" in got.probe_error.lower()


def test_probe_dataclass_to_dict_is_json_safe() -> None:
    """The route serialises via FastAPI's default encoder — round-trip
    the dataclass through ``to_dict`` to make sure no non-JSON-safe
    types sneak in (e.g. a future ``Path`` field)."""
    p = FullDiskAccessProbe(
        granted=True, probe_path="/x", probe_error=None, platform="Darwin"
    )
    d = p.to_dict()
    import json

    assert json.loads(json.dumps(d)) == d


def test_probe_uses_real_platform_when_not_monkeypatched() -> None:
    """Sanity: when nothing's monkey-patched, the probe runs against
    the host platform. This is the production path; we only assert
    we get a coherent record (no exception, ``platform`` matches the
    host) — not what ``granted`` is, since that depends on whether
    the CI host has FDA-protected TCC.db readable.
    """
    got = probe_full_disk_access("/nonexistent/intentional/path/TCC.db")
    assert got.platform == platform.system()
    assert isinstance(got.granted, bool)


# === HTTP layer ============================================================


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_healthz_permissions_granted_branch(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force Darwin + a readable probe file; route returns
    ``granted=true``."""
    _force_darwin(monkeypatch)
    probe_file = tmp_path / "TCC.db"
    probe_file.write_bytes(b"fake")
    monkeypatch.setattr(
        "rubick_backend.permissions.DEFAULT_FDA_PROBE_PATH", str(probe_file)
    )

    r = client.get("/healthz/permissions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "full_disk_access" in body
    fda = body["full_disk_access"]
    assert fda["granted"] is True
    assert fda["probe_path"] == str(probe_file)
    assert fda["probe_error"] is None
    assert fda["platform"] == "Darwin"


def test_healthz_permissions_missing_probe_path(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path doesn't exist; route returns granted=false + a clear
    ``probe_error``. Mirrors what a stripped macOS install would
    show."""
    _force_darwin(monkeypatch)
    monkeypatch.setattr(
        "rubick_backend.permissions.DEFAULT_FDA_PROBE_PATH",
        str(tmp_path / "absent" / "TCC.db"),
    )

    r = client.get("/healthz/permissions")
    body = r.json()
    assert body["full_disk_access"]["granted"] is False
    assert "missing" in body["full_disk_access"]["probe_error"].lower()


def test_healthz_permissions_non_darwin_route(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On non-Darwin, the route returns a coherent ``not applicable``
    response. UI hides the permissions section in this state."""
    monkeypatch.setattr(
        "rubick_backend.permissions.platform.system", lambda: "Linux"
    )
    r = client.get("/healthz/permissions")
    body = r.json()
    assert body["full_disk_access"]["platform"] == "Linux"
    assert body["full_disk_access"]["granted"] is False
    assert "not applicable" in body["full_disk_access"]["probe_error"].lower()
