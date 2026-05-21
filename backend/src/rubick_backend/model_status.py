"""Backend-side view of HuggingFace model cache state.

Powers ``GET /healthz/model`` (and any caller that wants to know "did
the 1.8 GB embedding model finish downloading yet?" without having to
talk to ``huggingface_hub`` itself or duplicate the cache-walk logic
that the Settings → Model SwiftUI tab already does).

Why a separate module instead of folding into ``embed/``:

- The status check must be safe to call **before** the embedder is
  loaded — Onboarding and Settings both render this view on launch,
  long before any search happens. Keeping the helpers here means we
  don't drag MLX (~189 MB native) into the import graph of a route
  that just stat()s a directory.
- Tests can monkey-patch ``HF_HOME`` and lay out a synthetic cache in
  ``tmp_path`` without touching the real ``~/.cache/huggingface/``.

The cache layout we walk is HuggingFace's standard hub format::

    <cache_root>/
        models--<owner>--<repo>/
            blobs/                 # actual file content (or ``.incomplete``
            refs/<branch>          # text file containing the resolved sha
            snapshots/<sha>/<file> # symlinks into ``blobs/``

Download states map onto that structure as:

- ``absent``    no ``models--<o>--<r>/`` dir at all.
- ``partial``   dir exists but ``refs/main`` is missing **or** any
                ``blobs/*.incomplete`` file is still on disk (the
                rename-on-finish marker the hub uses).
- ``complete``  ``refs/main`` resolves to a sha that has a populated
                ``snapshots/<sha>/`` and no ``.incomplete`` files.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

DownloadStatus = Literal["absent", "partial", "complete"]


# === Cache root resolution ==================================================


def hf_cache_root() -> Path:
    """The on-disk root HuggingFace's hub uses for this user.

    Resolution mirrors ``huggingface_hub``'s own precedence and the
    ``ModelTab`` Swift code so the API + UI never disagree:

    1. ``HUGGINGFACE_HUB_CACHE`` env var — direct override of the
       ``hub/`` dir.
    2. ``HF_HOME`` env var — fallback root; cache is ``$HF_HOME/hub``.
    3. ``~/.cache/huggingface/hub`` (Linux/macOS XDG default).
    """
    direct = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if direct:
        return Path(direct).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(repo_id: str, cache_root: Path | None = None) -> Path:
    """Where ``repo_id`` would live in the cache, whether or not it's there.

    HF stores ``owner/repo`` under ``models--<owner>--<repo>/``. We
    return the would-be path even when the dir doesn't exist yet so the
    UI can show a stable ``cache_path`` to copy into Finder once the
    download lands.
    """
    if cache_root is None:
        cache_root = hf_cache_root()
    folder = "models--" + repo_id.replace("/", "--")
    return cache_root / folder


# === Size + status helpers ==================================================


def directory_size_bytes(path: Path) -> int:
    """Physical on-disk size under ``path``, matching ``du -sh`` behavior.

    HF's cache layout has the same content reachable two ways:

        models--<o>--<r>/blobs/<hash>           # the real file
        models--<o>--<r>/snapshots/<sha>/foo    # symlink pointing at it

    Following symlinks would double-count every blob (once via
    ``blobs/`` walk and once via ``snapshots/`` walk). ``du`` skips
    symlinks by default; so do we. Net: the number we report matches
    the one a user sees from ``du -sh ~/.cache/huggingface/hub/...``,
    which is what the Settings → Model "cache size" copy shows.

    Returns 0 for a missing directory so the API always emits an int.
    """
    if not path.is_dir():
        return 0
    total = 0
    # ``followlinks=False`` (default) keeps ``os.walk`` from descending
    # into symlinked dirs; ``Path.is_symlink()`` then drops symlinked
    # files from the per-entry sum. The combination is what ``du -sh``
    # does (modulo block-vs-byte rounding, which we accept).
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            full = root_path / name
            try:
                if full.is_symlink():
                    continue
                total += full.stat().st_size
            except (FileNotFoundError, PermissionError, OSError):
                continue
    return total


def download_status(repo_id: str, cache_root: Path | None = None) -> DownloadStatus:
    """``"absent"`` / ``"partial"`` / ``"complete"`` for ``repo_id``.

    See module docstring for the layout. We tolerate either ``main`` or
    any single-file under ``refs/`` because some repos pin a non-main
    branch (mlx-community sometimes uses ``revision``-named refs); the
    presence of *any* refs file plus a non-empty matching snapshot dir
    is the actual completion signal.
    """
    base = model_cache_dir(repo_id, cache_root)
    if not base.is_dir():
        return "absent"

    blobs = base / "blobs"
    if blobs.is_dir():
        # ``snapshot_download`` writes ``<hash>.incomplete`` while the
        # download is in flight, then renames on success. Any leftover
        # is unambiguous evidence of a partial state — even if other
        # files in the same repo finished, the next ``snapshot_download``
        # call will redo the partial one.
        for entry in blobs.iterdir():
            if entry.name.endswith(".incomplete"):
                return "partial"

    refs_dir = base / "refs"
    if not refs_dir.is_dir():
        return "partial"

    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        return "partial"

    # Any ref pointing at a non-empty snapshot dir = good enough.
    for ref_file in refs_dir.iterdir():
        if not ref_file.is_file():
            continue
        try:
            sha = ref_file.read_text().strip()
        except (OSError, UnicodeDecodeError):
            continue
        if not sha:
            continue
        snap = snapshots / sha
        if snap.is_dir() and any(snap.iterdir()):
            return "complete"

    return "partial"


# === Cache delete ===========================================================


def delete_model_cache(
    repo_id: str, cache_root: Path | None = None
) -> dict[str, object]:
    """Wipe the on-disk HuggingFace cache subtree for ``repo_id``.

    Returns ``{"deleted_bytes", "path", "was_present"}`` so the API
    layer can echo it straight to the client without re-stat-ing. The
    helper is **idempotent** — calling against an absent cache returns
    ``deleted_bytes=0`` + ``was_present=False`` without raising. This
    matches the Settings UI's "I'll let the user re-click safely"
    convention; a 404 here would be more annoying than helpful.

    The in-process model singleton (if hydrated) is intentionally
    **left alone**: the user's current session keeps working until
    they restart Rubick, at which point the next ``snapshot_download``
    fetches fresh weights. That separation is what lets us treat
    "Re-download" as a pure file-system operation — no inter-thread
    MLX dance, no stop-the-world during an in-flight ``embed_query``.
    The cost is a small UX hand-wave ("quit + reopen for the
    re-download to actually start"), which the Swift side surfaces
    in the confirmation copy.
    """
    base = model_cache_dir(repo_id, cache_root)
    if not base.is_dir():
        return {"deleted_bytes": 0, "path": None, "was_present": False}
    bytes_freed = directory_size_bytes(base)
    # ``ignore_errors=False`` so a permission/IO failure surfaces as a
    # real exception and the API layer can return 5xx — silently
    # eating an EACCES would look like success to the user.
    shutil.rmtree(base)
    log.info(
        "deleted HF cache for %s (%d bytes, %s)",
        repo_id,
        bytes_freed,
        base,
    )
    return {
        "deleted_bytes": bytes_freed,
        "path": str(base),
        "was_present": True,
    }


# === Public per-model snapshot ==============================================


@dataclass(frozen=True)
class ModelSnapshot:
    """Everything a UI needs to render one model card or progress meter."""

    id: str
    repo: str
    purpose: str
    declared_bytes: int
    cache_path: str | None
    cache_bytes: int
    download_status: DownloadStatus
    loaded_in_memory: bool | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "repo": self.repo,
            "purpose": self.purpose,
            "declared_bytes": self.declared_bytes,
            "cache_path": self.cache_path,
            "cache_bytes": self.cache_bytes,
            "download_status": self.download_status,
            "loaded_in_memory": self.loaded_in_memory,
        }


def snapshot(
    *,
    model_id: str,
    repo: str,
    purpose: str,
    declared_bytes: int,
    loaded_in_memory: bool | None,
    cache_root: Path | None = None,
) -> ModelSnapshot:
    """Build a ``ModelSnapshot`` by stat()-ing the cache for ``repo``.

    ``loaded_in_memory`` is the caller's responsibility (only the
    owner of the model singleton knows whether it's hydrated). For a
    model with no long-lived in-process handle, callers pass
    ``None`` so the UI renders a neutral status.
    """
    base = model_cache_dir(repo, cache_root)
    status = download_status(repo, cache_root)
    cache_path = str(base) if base.is_dir() else None
    cache_bytes = directory_size_bytes(base)
    return ModelSnapshot(
        id=model_id,
        repo=repo,
        purpose=purpose,
        declared_bytes=declared_bytes,
        cache_path=cache_path,
        cache_bytes=cache_bytes,
        download_status=status,
        loaded_in_memory=loaded_in_memory,
    )
