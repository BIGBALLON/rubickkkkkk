"""Centralized configuration for the Rubick backend.

On-disk layout under ``RUBICK_DATA_DIR`` (defaults to
``~/Library/Application Support/Rubick/``):

    ~/Library/Application Support/Rubick/
      ├── models/      # ML model weights cache (future: redirect HF_HOME here)
      ├── lancedb/     # vector + metadata store
      ├── thumbnails/  # image / video thumbnails (WebP)
      ├── logs/        # rotating ingestion logs
      └── settings.json  # user-editable runtime knobs

The data root can be overridden via the ``RUBICK_DATA_DIR`` environment
variable; this is what tests and dev smokes use so they don't write into
the user's real Library.

Two flavours of configuration coexist here:

- **Build-time / boot-time constants** (``MAIN_MODEL_REPO``,
  ``EMBED_DIM``, etc.). Module-level finals; never mutated.
- **User-editable runtime knobs** (``TARGET_TOKENS`` /
  ``HARD_MAX_TOKENS``). Stored as plain module-level
  ``int`` so any caller can do ``settings.TARGET_TOKENS`` to read
  the current value, but mutated through ``update_chunking_settings``
  which persists to ``settings.json`` so the choice survives
  restart. Resolution priority (highest wins):

    1. ``settings.json`` on disk
    2. ``RUBICK_TARGET_TOKENS`` / ``RUBICK_HARD_MAX_TOKENS`` env vars
    3. Compile-time defaults (``_DEFAULT_TARGET_TOKENS`` etc.)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# === Model repositories (HuggingFace Hub) ===================================

MAIN_MODEL_REPO = "jinaai/jina-embeddings-v5-omni-nano-retrieval-mlx"

# HuggingFace endpoint — override with HF_ENDPOINT env var or settings.json.
# Users in China should set this to "https://hf-mirror.com".
HF_ENDPOINT: str = os.environ.get("HF_ENDPOINT", "")

EMBED_DIM = 768

# Declared on-disk size for the embedding model, used by
# ``GET /healthz/model`` (and any UI consuming it) to render
# ``downloaded 1.7 / 2.0 GB``-style progress before the snapshot is
# fully resolved. The number comes from ``du -sh`` of a freshly-
# completed snapshot at the pinned revision, rounded *up* to the next
# 100 MB so an in-progress download never shows >100 % even with the
# inevitable HF blob-vs-config-file slop.
#
# Real measurement 2026-05-14 against the currently-pinned revision:
#   embedding (jinaai/jina-embeddings-v5-omni-nano-retrieval-mlx):
#     1,936,323,190 B  (≈1.80 GiB; ``du -sh`` rounds to 1.8G)
MAIN_MODEL_DECLARED_BYTES = 2_000_000_000  # ~1.86 GiB headroom

# Human-readable purpose string shown on the single model card. Sourced
# from the same docs the Settings → Model tab already shows; centralizing
# it here means the API + Swift UI can't drift out of sync.
MAIN_MODEL_PURPOSE = (
    "Multimodal embedding (text / image / video → 768-dim, Apple Silicon MLX)."
)

# === On-disk data layout ====================================================


def _resolve_data_root() -> Path:
    override = os.environ.get("RUBICK_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / "Library" / "Application Support" / "Rubick"


DATA_ROOT: Path = _resolve_data_root()
MODELS_DIR: Path = DATA_ROOT / "models"
LANCEDB_DIR: Path = DATA_ROOT / "lancedb"
THUMBNAILS_DIR: Path = DATA_ROOT / "thumbnails"
LOGS_DIR: Path = DATA_ROOT / "logs"
SETTINGS_FILE: Path = DATA_ROOT / "settings.json"

# LanceDB table that stores all modalities.
LANCEDB_TABLE = "documents"

# Nebula M3 — precomputed 3-D UMAP map of image/video embeddings.
NEBULA_MAP_FILE: Path = DATA_ROOT / "nebula_map.json"


def ensure_data_dirs() -> None:
    """Create the on-disk data directory tree if it doesn't exist yet."""
    for d in (DATA_ROOT, MODELS_DIR, LANCEDB_DIR, THUMBNAILS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# === User-editable runtime knobs =============================================
#
# Default chunk-size targets (overridable via settings.json / PATCH /settings).
# v0.0.1 supported (hard-coded inside ``ingest/text.py``). v0.0.2 lifts
# them here so:
#  - env vars can override on a per-process basis (CI, dev smokes)
#  - the upcoming Settings → Index "Text chunking" section can mutate
#    them at runtime via ``PATCH /settings`` (api/settings.py)
#  - the choice persists across restarts via SETTINGS_FILE
#
# Bounds picked empirically: anything below 100 tokens fragments
# semantic units smaller than the model's effective context; anything
# above 8192 risks blowing past the model's max input length on long
# heading blocks. The Swift UI clamps to a tighter user-facing range.
_DEFAULT_TARGET_TOKENS: int = 2048
_DEFAULT_HARD_MAX_TOKENS: int = 6144
_TARGET_TOKENS_BOUNDS: tuple[int, int] = (100, 8192)
_HARD_MAX_TOKENS_BOUNDS: tuple[int, int] = (200, 8192)

# === Exclusion-pattern caps + sanitiser (v1.x #3) ===========================
#
# Hoisted **above** ``_read_settings_file`` because the file-reading
# helper calls ``_sanitize_exclusion_patterns`` to coerce the persisted
# list at module import time. Keeping all three at module scope (rather
# than re-defining the function later in the file) avoids a forward-
# reference NameError when a real ``settings.json`` happens to carry an
# ``exclusion_patterns`` field.

_MAX_EXCLUSION_PATTERNS: int = 64
_MAX_PATTERN_LENGTH: int = 200


def _sanitize_exclusion_patterns(raw: list[object]) -> list[str]:
    """Coerce-and-dedupe a list of pattern strings.

    Rejects: non-strings, empty / whitespace-only, > 200 chars (a
    hard ceiling so a runaway paste doesn't blow up the walker's
    per-entry fnmatch loop). De-dupes preserving order. Caps total
    list size at ``_MAX_EXCLUSION_PATTERNS`` and logs if we trim.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or len(s) > _MAX_PATTERN_LENGTH:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) > _MAX_EXCLUSION_PATTERNS:
        log.warning(
            "exclusion_patterns has %d entries; trimming to %d",
            len(out),
            _MAX_EXCLUSION_PATTERNS,
        )
        out = out[:_MAX_EXCLUSION_PATTERNS]
    return out


def _read_int_env(name: str) -> int | None:
    """Parse a positive int from ``os.environ[name]``, or None if
    unset / unparseable. Surfaces a warning on bad data instead of
    silently dropping it — a typo'd env var is a user-actionable
    problem, not something to hide.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring invalid %s=%r (not an int)", name, raw)
        return None


def _read_settings_file() -> dict[str, object]:
    """Load the on-disk settings JSON, returning ``{}`` on any failure.

    Failures (missing file, permission denied, malformed JSON,
    unexpected schema) all degrade to defaults rather than raising —
    a corrupted user-edit shouldn't take the backend down. Bad
    fields are simply ignored.
    """
    if not SETTINGS_FILE.is_file():
        return {}
    try:
        raw = SETTINGS_FILE.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("settings.json read failed: %s", e)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("settings.json malformed (%s); ignoring", e)
        return {}
    if not isinstance(data, dict):
        log.warning("settings.json top-level is not an object; ignoring")
        return {}
    out: dict[str, object] = {}
    for key in ("target_tokens", "hard_max_tokens"):
        if isinstance(data.get(key), int):
            out[key] = data[key]
    # ``exclusion_patterns`` is a list[str] of fnmatch globs that the
    # ingest walker AND-s with the always-on default deny-list.
    raw_patterns = data.get("exclusion_patterns")
    if isinstance(raw_patterns, list):
        clean = _sanitize_exclusion_patterns(raw_patterns)
        if clean:
            out["exclusion_patterns"] = clean
    return out


def _resolve_initial_chunking() -> tuple[int, int]:
    """Return ``(target, hard_max)`` after applying the resolution
    chain (file → env → default) and validation. Called exactly
    once at module import.
    """
    file_overrides = _read_settings_file()
    env_target = _read_int_env("RUBICK_TARGET_TOKENS")
    env_hard_max = _read_int_env("RUBICK_HARD_MAX_TOKENS")

    file_target = file_overrides.get("target_tokens")
    file_hard_max = file_overrides.get("hard_max_tokens")

    target_raw = (
        file_target if isinstance(file_target, int)
        else env_target or _DEFAULT_TARGET_TOKENS
    )
    hard_max_raw = (
        file_hard_max if isinstance(file_hard_max, int)
        else env_hard_max or _DEFAULT_HARD_MAX_TOKENS
    )
    return _validate_chunking(target_raw, hard_max_raw)


def _resolve_initial_exclusions() -> list[str]:
    """Return the persisted user exclusion list, sanitised. Empty
    when the user hasn't added any rules.
    """
    file_overrides = _read_settings_file()
    raw = file_overrides.get("exclusion_patterns")
    if isinstance(raw, list):
        return _sanitize_exclusion_patterns(raw)
    return []


def _validate_chunking(target: int, hard_max: int) -> tuple[int, int]:
    """Clamp + sanity-check a (target, hard_max) pair.

    Rules:
    - Each value must lie in its respective bounds (clamp silently).
    - hard_max must be ≥ target — otherwise the chunker's "soft
      target then hard backstop" semantics fall apart. We bump
      hard_max up to target if the user passed a backwards pair,
      and log so the discrepancy is visible.
    """
    t_lo, t_hi = _TARGET_TOKENS_BOUNDS
    h_lo, h_hi = _HARD_MAX_TOKENS_BOUNDS
    target = max(t_lo, min(t_hi, target))
    hard_max = max(h_lo, min(h_hi, hard_max))
    if hard_max < target:
        log.warning(
            "hard_max_tokens (%d) < target_tokens (%d); bumping to %d",
            hard_max, target, target,
        )
        hard_max = target
    return target, hard_max


# Module-level mutables: read by ``ingest/text.py`` on every chunk pass
# (intentionally not snapshotted into the function's locals so a
# ``PATCH /settings`` takes effect on the very next ingest job
# without a restart).
TARGET_TOKENS, HARD_MAX_TOKENS = _resolve_initial_chunking()

# === User-editable exclusion patterns (v1.x #3) — runtime list ============
#
# fnmatch globs the ingest walker AND-s with the always-on default
# deny-list on every walk pass. The sanitiser + caps live further up
# the file (right after the chunking defaults) because
# ``_read_settings_file`` needs them at import time; this section is
# just the live runtime list, populated from the persisted file by
# ``_resolve_initial_exclusions``.

EXCLUSION_PATTERNS: list[str] = _resolve_initial_exclusions()


def get_chunking_settings() -> dict[str, object]:
    """Snapshot of the current text-chunking parameters + exclusion
    patterns. Used by ``PATCH /settings`` callers to echo what was
    actually applied (post-clamp / post-sanitise).
    """
    return {
        "target_tokens": TARGET_TOKENS,
        "hard_max_tokens": HARD_MAX_TOKENS,
        "exclusion_patterns": list(EXCLUSION_PATTERNS),
    }


def get_chunking_metadata() -> dict[str, object]:
    """Same as ``get_chunking_settings`` plus the bound / default
    metadata the Swift UI uses to drive its stepper, preset cards,
    and the Privacy → "Always excluded" list. Kept as a separate
    accessor so simple consumers don't have to skip past the UI
    hint fields.
    """
    from .ingest import EXCLUDED_DIR_NAMES

    return {
        **get_chunking_settings(),
        "defaults": {
            "target_tokens": _DEFAULT_TARGET_TOKENS,
            "hard_max_tokens": _DEFAULT_HARD_MAX_TOKENS,
        },
        "bounds": {
            "target_tokens": list(_TARGET_TOKENS_BOUNDS),
            "hard_max_tokens": list(_HARD_MAX_TOKENS_BOUNDS),
        },
        # Always-on deny-list (hidden dirs are handled separately in
        # the walker; we surface the named ones so the UI can show
        # the user what runs unconditionally).
        "default_exclusion_dir_names": sorted(EXCLUDED_DIR_NAMES),
        "exclusion_pattern_limits": {
            "max_count": _MAX_EXCLUSION_PATTERNS,
            "max_length": _MAX_PATTERN_LENGTH,
        },
    }


def update_chunking_settings(
    *,
    target_tokens: int | None = None,
    hard_max_tokens: int | None = None,
    exclusion_patterns: list[str] | None = None,
    persist: bool = True,
) -> dict[str, object]:
    """Mutate the runtime chunking / exclusion parameters and (optionally) persist.

    ``None`` values keep the current setting. Validation runs on
    every call — out-of-bounds tokens are clamped silently and a
    backwards pair (hard_max < target) bumps hard_max up. Exclusion
    patterns are sanitised (dropped if non-string, empty, or too
    long; de-duped; capped at ``_MAX_EXCLUSION_PATTERNS``).

    Returns the post-update snapshot so the caller (typically the
    ``PATCH /settings`` route) can echo the actually-applied values
    back to the client. ``persist=False`` is the test hook —
    production always persists so the choice survives restart.
    """
    global TARGET_TOKENS, HARD_MAX_TOKENS, EXCLUSION_PATTERNS
    new_target = TARGET_TOKENS if target_tokens is None else target_tokens
    new_hard_max = HARD_MAX_TOKENS if hard_max_tokens is None else hard_max_tokens
    new_target, new_hard_max = _validate_chunking(new_target, new_hard_max)

    TARGET_TOKENS = new_target
    HARD_MAX_TOKENS = new_hard_max
    if exclusion_patterns is not None:
        EXCLUSION_PATTERNS = _sanitize_exclusion_patterns(
            list(exclusion_patterns)
        )

    if persist:
        _persist_chunking_settings()
    return get_chunking_settings()


def _persist_chunking_settings() -> None:
    """Write the current chunking + exclusion settings to ``SETTINGS_FILE``.

    Best-effort: a write failure logs but doesn't raise — the
    runtime change still took effect, the user just won't see it
    after a restart. Atomic via temp-rename so a crash mid-write
    can't truncate the file.
    """
    ensure_data_dirs()
    payload = get_chunking_settings()
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(SETTINGS_FILE)
    except OSError as e:
        log.warning("settings.json write failed: %s", e)


def reset_chunking_for_tests() -> None:
    """Test hook: drop runtime overrides back to compile-time defaults
    without touching ``SETTINGS_FILE``. Production never calls this.
    """
    global TARGET_TOKENS, HARD_MAX_TOKENS, EXCLUSION_PATTERNS
    TARGET_TOKENS = _DEFAULT_TARGET_TOKENS
    HARD_MAX_TOKENS = _DEFAULT_HARD_MAX_TOKENS
    EXCLUSION_PATTERNS = []
