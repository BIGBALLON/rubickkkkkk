"""Unit tests for ``rubick_backend.model_status``.

Synthesizes a HuggingFace-style cache layout in ``tmp_path`` and walks
the helpers; never touches the real ``~/.cache/huggingface/``. Fast
enough to run on every ``pytest`` (no ``slow`` marker).

The layout we mimic is the one HF Hub produces for any
``snapshot_download(repo_id)`` call::

    <cache_root>/
        models--<owner>--<repo>/
            blobs/                  # actual file content
            refs/<branch>           # text file with the resolved sha
            snapshots/<sha>/<file>  # symlinks to ../../blobs/<hash>

Three download states get exercised: ``absent`` (no dir at all),
``partial`` (``.incomplete`` files left over from an interrupted
download), and ``complete`` (refs + snapshots populated).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rubick_backend.model_status import (
    DownloadStatus,
    ModelSnapshot,
    delete_model_cache,
    directory_size_bytes,
    download_status,
    hf_cache_root,
    model_cache_dir,
    snapshot,
)

REPO_ID = "demoorg/demo-model"
FAKE_SHA = "deadbeef0000111122223333444455556666777788889999aaaabbbbccccdddd"


# === Helpers to build fake HF cache layouts ================================


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _build_complete_cache(cache_root: Path, repo_id: str, sha: str = FAKE_SHA) -> Path:
    """Mirror the post-snapshot_download layout for ``repo_id``.

    Two blobs (one big-ish, one small) plus a ``refs/main`` pointer and
    a populated ``snapshots/<sha>/`` of symlinks. Returns the per-repo
    ``models--<o>--<r>/`` dir for further mutation by callers.
    """
    base = cache_root / ("models--" + repo_id.replace("/", "--"))
    blobs = base / "blobs"
    snapshots = base / "snapshots" / sha
    refs = base / "refs"

    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    refs.mkdir(parents=True)

    big = blobs / "deadbeef00"
    small = blobs / "cafef00d11"
    big.write_bytes(b"x" * 4096)
    small.write_bytes(b"y" * 256)

    (snapshots / "model.safetensors").symlink_to(big)
    (snapshots / "config.json").symlink_to(small)
    (refs / "main").write_text(sha)
    return base


# === Tests =================================================================


def test_hf_cache_root_respects_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/some/where/hub")
    monkeypatch.delenv("HF_HOME", raising=False)
    assert hf_cache_root() == Path("/some/where/hub")


def test_hf_cache_root_falls_back_to_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "/tmp/hf-test")
    # Always ``$HF_HOME/hub`` — the parent dir holds non-cache state too.
    assert hf_cache_root() == Path("/tmp/hf-test/hub")


def test_hf_cache_root_xdg_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert hf_cache_root() == Path.home() / ".cache" / "huggingface" / "hub"


def test_model_cache_dir_uses_owner_repo_folder_form(tmp_path: Path) -> None:
    got = model_cache_dir("acme/super-cool-model", cache_root=tmp_path)
    assert got == tmp_path / "models--acme--super-cool-model"


def test_directory_size_bytes_sums_recursive_files(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b.bin", b"abc" * 100)  # 300 B
    _write(tmp_path / "a" / "c" / "d.bin", b"x" * 1024)  # 1024 B
    _write(tmp_path / "stray.txt", b"!" * 7)  # 7 B
    assert directory_size_bytes(tmp_path) == 300 + 1024 + 7


def test_directory_size_bytes_returns_zero_for_missing(tmp_path: Path) -> None:
    assert directory_size_bytes(tmp_path / "nope") == 0


def test_directory_size_bytes_skips_symlinks_to_avoid_double_counting(
    tmp_path: Path,
) -> None:
    """The HF layout has the same blob reachable via ``blobs/<hash>``
    AND via ``snapshots/<sha>/<file>`` (a symlink). Following symlinks
    would report ~3.6 GB for the 1.8 GB jina cache. We match ``du -sh``
    instead (skip symlinks; count the blob exactly once via its real
    location).
    """
    real = tmp_path / "blobs" / "weights.bin"
    _write(real, b"z" * 8192)
    link_dir = tmp_path / "snapshots" / FAKE_SHA
    link_dir.mkdir(parents=True)
    (link_dir / "model.safetensors").symlink_to(real)

    # Whole tree: 8192 bytes of real blob, period.
    assert directory_size_bytes(tmp_path) == 8192

    # Snapshot dir alone: only symlinks live there → 0 bytes.
    # That's correct behaviour: nothing was actually allocated for the
    # snapshot dir, all the bytes are in blobs/.
    assert directory_size_bytes(link_dir) == 0


def test_download_status_absent_when_dir_missing(tmp_path: Path) -> None:
    assert download_status(REPO_ID, cache_root=tmp_path) == "absent"


def test_download_status_complete_for_full_layout(tmp_path: Path) -> None:
    _build_complete_cache(tmp_path, REPO_ID)
    assert download_status(REPO_ID, cache_root=tmp_path) == "complete"


def test_download_status_partial_when_incomplete_blob_present(tmp_path: Path) -> None:
    base = _build_complete_cache(tmp_path, REPO_ID)
    # A leftover ``.incomplete`` is unambiguous evidence that the
    # last ``snapshot_download`` was interrupted; the resolution is to
    # call snapshot_download again — until then we should not report
    # ``complete`` even if the rest of the tree is fine.
    (base / "blobs" / "newfile.incomplete").write_bytes(b"00")
    assert download_status(REPO_ID, cache_root=tmp_path) == "partial"


def test_download_status_partial_when_refs_missing(tmp_path: Path) -> None:
    base = _build_complete_cache(tmp_path, REPO_ID)
    # Nuke ``refs/main`` while leaving the snapshot dir behind — this
    # mimics the case where a future HF version stores refs differently
    # or the user manually deleted them.
    (base / "refs" / "main").unlink()
    # No alternate ref files left either.
    assert download_status(REPO_ID, cache_root=tmp_path) == "partial"


def test_download_status_partial_when_snapshot_dir_empty(tmp_path: Path) -> None:
    base = tmp_path / ("models--" + REPO_ID.replace("/", "--"))
    (base / "blobs").mkdir(parents=True)
    (base / "refs").mkdir()
    (base / "snapshots" / FAKE_SHA).mkdir(parents=True)
    (base / "refs" / "main").write_text(FAKE_SHA)
    # Refs file points at an empty snapshot dir — interrupted at the
    # symlink-creation step.
    assert download_status(REPO_ID, cache_root=tmp_path) == "partial"


def test_download_status_complete_via_alt_ref_branch(tmp_path: Path) -> None:
    """Some HF repos pin a non-main branch (mlx-community sometimes
    does). The status check should accept *any* refs file pointing at a
    populated snapshot, not insist on ``main`` specifically.
    """
    base = _build_complete_cache(tmp_path, REPO_ID)
    (base / "refs" / "main").unlink()
    (base / "refs" / "mlx-q4").write_text(FAKE_SHA)
    assert download_status(REPO_ID, cache_root=tmp_path) == "complete"


def test_snapshot_assembles_full_payload_for_complete_repo(tmp_path: Path) -> None:
    _build_complete_cache(tmp_path, REPO_ID)
    snap = snapshot(
        model_id="demo",
        repo=REPO_ID,
        purpose="Demo model used by the test suite.",
        declared_bytes=10_000,
        loaded_in_memory=True,
        cache_root=tmp_path,
    )
    assert isinstance(snap, ModelSnapshot)
    assert snap.id == "demo"
    assert snap.repo == REPO_ID
    assert snap.purpose.startswith("Demo")
    assert snap.declared_bytes == 10_000
    assert snap.cache_path == str(tmp_path / "models--demoorg--demo-model")
    # 4096 (big blob) + 256 (small blob) + len(SHA) (refs/main).
    # snapshots/ tree is symlinks → 0 contribution, matching ``du -sh``.
    assert snap.cache_bytes == 4096 + 256 + len(FAKE_SHA)
    assert snap.download_status == "complete"
    assert snap.loaded_in_memory is True


def test_snapshot_for_absent_repo_emits_zero_size_and_null_path(tmp_path: Path) -> None:
    snap = snapshot(
        model_id="missing",
        repo=REPO_ID,
        purpose="Not on disk yet.",
        declared_bytes=1234,
        loaded_in_memory=False,
        cache_root=tmp_path,
    )
    assert snap.download_status == "absent"
    assert snap.cache_path is None
    assert snap.cache_bytes == 0
    assert snap.loaded_in_memory is False


def test_snapshot_to_dict_round_trip(tmp_path: Path) -> None:
    """``ModelSnapshot.to_dict`` is what FastAPI serializes — make sure
    every field lands in the JSON payload with the documented key.
    """
    _build_complete_cache(tmp_path, REPO_ID)
    snap = snapshot(
        model_id="demo",
        repo=REPO_ID,
        purpose="x",
        declared_bytes=42,
        loaded_in_memory=None,
        cache_root=tmp_path,
    )
    d = snap.to_dict()
    assert set(d) == {
        "id",
        "repo",
        "purpose",
        "declared_bytes",
        "cache_path",
        "cache_bytes",
        "download_status",
        "loaded_in_memory",
    }
    assert d["loaded_in_memory"] is None
    assert d["download_status"] == "complete"


def test_download_status_type_alias_is_what_we_advertise() -> None:
    """Sanity check: the ``Literal`` alias has exactly the three values
    callers (Settings → Model UI, Onboarding) hard-code against. If we
    ever add a fourth, the API consumers need updating in lockstep —
    failing this assertion is the breadcrumb.
    """
    # ``Literal`` types expose their values via ``__args__``.
    assert set(DownloadStatus.__args__) == {"absent", "partial", "complete"}


def test_directory_size_bytes_skips_unreadable_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable files (perm-denied, missing race) are silently
    skipped — a 1.4 GB cache shouldn't crash the API just because one
    file lost its perms during a stat race.
    """
    good = tmp_path / "ok.bin"
    good.write_bytes(b"a" * 100)
    # We can't easily create a chmod-000 file in CI (test runs as the
    # owner so they're still readable). Instead, monkeypatch ``Path.stat``
    # to raise PermissionError on a sentinel path so we exercise the
    # except branch deterministically.
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"b" * 50)

    real_stat = os.stat

    def _stat(path, *args, **kwargs):
        if str(path).endswith("bad.bin"):
            raise PermissionError("test-only")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", _stat)
    assert directory_size_bytes(tmp_path) == 100


# === delete_model_cache (v1.x #5) ==========================================


def test_delete_model_cache_wipes_complete_cache(tmp_path: Path) -> None:
    """Happy path: a fully-populated cache disappears off disk and the
    helper reports the freed bytes + the path it removed."""
    base = _build_complete_cache(tmp_path, REPO_ID)
    assert base.is_dir()
    expected_bytes = directory_size_bytes(base)

    result = delete_model_cache(REPO_ID, cache_root=tmp_path)

    assert result["was_present"] is True
    assert result["deleted_bytes"] == expected_bytes
    assert result["path"] == str(base)
    assert not base.exists()


def test_delete_model_cache_is_idempotent_on_absent(tmp_path: Path) -> None:
    """Calling against a cache that never existed must succeed quietly —
    "make sure it's gone" semantics, not "fail if you can't find it"."""
    result = delete_model_cache(REPO_ID, cache_root=tmp_path)
    assert result == {"deleted_bytes": 0, "path": None, "was_present": False}


def test_delete_model_cache_only_touches_its_own_repo(tmp_path: Path) -> None:
    """A second repo sharing the same cache root must survive the
    delete — we ``rmtree`` the per-repo dir, never the cache root.
    """
    _build_complete_cache(tmp_path, REPO_ID)
    other_repo = "someorg/other-model"
    other_base = _build_complete_cache(tmp_path, other_repo)

    delete_model_cache(REPO_ID, cache_root=tmp_path)

    assert other_base.is_dir()
    assert download_status(other_repo, cache_root=tmp_path) == "complete"


def test_delete_model_cache_flips_download_status_to_absent(tmp_path: Path) -> None:
    """After delete, the existing snapshot helper must report ``absent``
    — this is the signal the Swift UI watches to drive the post-delete
    "model card now shows 'Not downloaded'" transition."""
    _build_complete_cache(tmp_path, REPO_ID)
    assert download_status(REPO_ID, cache_root=tmp_path) == "complete"
    delete_model_cache(REPO_ID, cache_root=tmp_path)
    assert download_status(REPO_ID, cache_root=tmp_path) == "absent"
