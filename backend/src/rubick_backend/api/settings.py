"""User-editable runtime knobs (chunking, etc.).

Two endpoints:

- ``GET  /settings`` — return the current chunking parameters plus
  the metadata the Settings → Index "Text chunking" UI uses to drive
  its preset cards (defaults + bounds).
- ``PATCH /settings`` — update target / hard-max tokens; persist to
  ``RUBICK_DATA_DIR/settings.json`` so the choice survives restart.

Why a dedicated router rather than folding into ``healthz``: the
``/healthz/*`` family is for read-only liveness / readiness; mutating
state belongs on its own surface so we can attach auth or rate
limiting later without disturbing health probes.

Validation lives entirely in ``settings.update_chunking_settings`` —
out-of-bounds values are clamped silently (matching the Swift stepper)
rather than raising 400s, so a user dragging the slider past the
limit gets a snappy "I'll cap it for you" instead of an error toast.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import settings as backend_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/settings")


class ChunkingPatch(BaseModel):
    """Body of ``PATCH /settings``. All fields optional so the Swift
    client can flip one knob at a time without re-sending the rest.
    """

    target_tokens: int | None = Field(
        default=None,
        description="Soft target — chunks bigger than this get split.",
    )
    hard_max_tokens: int | None = Field(
        default=None,
        description="Hard ceiling — never let a chunk exceed this.",
    )
    exclusion_patterns: list[str] | None = Field(
        default=None,
        description=(
            "v1.x #3 — fnmatch globs applied to dir + file basenames "
            "by the ingest walker, AND-ed with the always-on default "
            "deny-list. Pass ``[]`` to clear all user rules; omit "
            "(``null``) to leave them unchanged. Sanitised server-side: "
            "non-strings, empty / whitespace-only, and overlong entries "
            "are dropped; duplicates are coalesced; the list is capped "
            "at the limit advertised by ``GET /settings``."
        ),
    )


@router.get("")
async def get_settings() -> dict[str, object]:
    """Snapshot of the current chunking + exclusion settings + UI metadata.

    Wrapped in ``asyncio.to_thread`` for symmetry with the rest of
    the API surface even though the underlying read is just two
    attribute lookups + a small file read on first call. Cheap.
    """
    return await asyncio.to_thread(backend_settings.get_chunking_metadata)


@router.patch("")
async def patch_settings(patch: ChunkingPatch) -> dict[str, object]:
    """Mutate chunking parameters / exclusion patterns and persist to
    ``settings.json``.

    Returns the post-update snapshot so the client can echo the
    actually-applied values back to the UI — important because we
    silently clamp out-of-bounds tokens and silently drop invalid /
    duplicate exclusion entries rather than erroring (the Swift
    stepper / list editor behaves the same way).
    """
    return await asyncio.to_thread(
        backend_settings.update_chunking_settings,
        target_tokens=patch.target_tokens,
        hard_max_tokens=patch.hard_max_tokens,
        exclusion_patterns=patch.exclusion_patterns,
    )
