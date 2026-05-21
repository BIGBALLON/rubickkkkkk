"""Video ingestion pipeline (visual embedding only; no transcript track).

Per file we emit a single visual embedding row: 32 frames sampled
uniformly across the timeline, fed to ``encode_video`` for one
768-dim vector (one ``modality="video"`` row, ``chunk_idx=0``). The
jina v5 omni video path requires an *even* frame count
(``temporal_patch_size=2``); 32 stays well under the model's
context budget and matches what spike-0 verified.

The original spec called for a parallel transcript track
(``modality="video_transcript"``) produced by extracting the audio
stream with PyAV and running it through Whisper; v0.0.2 dropped it
once the visual embedding was confirmed to land in the same joint
space as text. Text queries match video files via the visual
embedding alone (the model's ``encode_video`` is contrastively
trained against text). Transcript rows were removed in v0.0.2; see
``store/schema.py`` for the legacy ``video_transcript`` note.

Thumbnail: grab the frame at the 1-second mark (or the first frame if
shorter), resize to 128-px short edge, and write
``<DATA_ROOT>/thumbnails/<doc_id>.webp`` — same directory the image
pipeline uses, so the UI can render any document type uniformly.

Files longer than ``MAX_DURATION_S`` (2 min) get a
single ``modality="rejected"`` placeholder so the next index pass
doesn't re-scan them.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import settings
from ..embed import embed_video
from ..store import is_doc_indexed, make_row, path_cache

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# === Tunables ===============================================================

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})

# 2-minute hard ceiling for indexing.
MAX_DURATION_S: float = 2 * 60.0

# 32 uniform frames (even count required by temporal_patch_size=2).
N_FRAMES_SAMPLED: int = 32

# 128-px short edge WebP thumbnail (same as image pipeline).
THUMBNAIL_SHORT_EDGE: int = 128
THUMBNAIL_QUALITY: int = 75
THUMBNAIL_SEED_TIME_S: float = 1.0

# Placeholder embedding for rejected rows.
_REJECTED_EMBEDDING: list[float] = [0.0] * settings.EMBED_DIM


# === Public API =============================================================


def ingest_file(path: Path | str) -> list[dict[str, Any]]:
    """Process one video file; return rows ready for ``table.add``.

    Skip rules (return ``[]``):
    - unsupported extension
    - not a regular file
    - PyAV can't open the file or duration probe fails
    - frame decode produces fewer than 2 usable frames (degenerate
      video — encode_video would crash on that anyway)

    Reject rule (return single ``modality="rejected"`` row):
    - duration > 2 min
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        log.warning("skip %s — not a regular file", p)
        return []
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.warning("skip %s — unsupported extension", p)
        return []

    mtime = int(p.stat().st_mtime)

    # Fast-path dedup: skip without reading the (potentially large) video file.
    cached_doc_id = path_cache.lookup(str(p), mtime)
    if cached_doc_id and is_doc_indexed(cached_doc_id):
        log.debug("fast-skip %s — path+mtime cache hit", p)
        return []

    # Slow-path: read + sha256 before paying PyAV probe + frame-decode costs.
    file_bytes = p.read_bytes()
    sha = hashlib.sha256(file_bytes).hexdigest()
    doc_id = sha[:16]
    if is_doc_indexed(doc_id):
        log.info("skip %s — doc_id=%s already indexed", p, doc_id)
        path_cache.record(str(p), mtime, doc_id)
        return []

    duration = _probe_video(p)
    if duration is None:
        log.warning("skip %s — PyAV probe failed", p)
        return []
    filename = p.stem

    if duration > MAX_DURATION_S:
        log.info(
            "reject %s — duration %.1fs exceeds %.0fs cap",
            p,
            duration,
            MAX_DURATION_S,
        )
        path_cache.record(str(p), mtime, doc_id)
        return [
            make_row(
                doc_id=doc_id,
                modality="rejected",
                chunk_idx=0,
                embedding=_REJECTED_EMBEDDING,
                file_path=str(p),
                sha256=sha,
                mtime=mtime,
                filename=filename,
                duration_s=duration,
                status="rejected",
                rejected_reason="video_too_long",
            )
        ]

    frames, thumb_frame = _decode_uniform_frames(
        p, n_target=N_FRAMES_SAMPLED, thumb_at_s=THUMBNAIL_SEED_TIME_S
    )
    if not frames or len(frames) < 2:
        log.warning("skip %s — fewer than 2 usable frames after decode", p)
        return []
    if len(frames) % 2:
        frames = frames[:-1]
    n_used = len(frames)

    thumbnail_path = _write_thumbnail(thumb_frame, doc_id) if thumb_frame else None

    try:
        vec = embed_video(frames)
    except Exception as e:
        log.warning("video embed failure on %s: %s", p, e)
        return []

    path_cache.record(str(p), mtime, doc_id)
    return [
        make_row(
            doc_id=doc_id,
            modality="video",
            chunk_idx=0,
            embedding=vec.tolist(),
            file_path=str(p),
            sha256=sha,
            mtime=mtime,
            filename=filename,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            duration_s=duration,
            n_frames_sampled=n_used,
        )
    ]


# === PyAV helpers ===========================================================


def _probe_video(path: Path) -> float | None:
    """Return the container's duration in seconds, or None on failure.

    We read the container metadata only — no full decode. Some
    containers store duration on the format-level rather than per-
    stream; ``_container_duration`` coalesces.
    """
    try:
        import av

        with av.open(str(path)) as c:
            return _container_duration(c)
    except Exception as e:
        log.debug("PyAV probe failed on %s: %s", path, e)
        return None


def _container_duration(container) -> float | None:
    """Best-effort container duration in seconds.

    Tries the format-level ``container.duration`` first (microseconds),
    then falls back to the first video stream's ``duration * time_base``.
    """
    if container.duration is not None and container.duration > 0:
        return float(container.duration) / 1_000_000.0
    for stream in container.streams.video:
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    return None


def _decode_uniform_frames(
    path: Path,
    *,
    n_target: int,
    thumb_at_s: float,
) -> tuple[list, PILImage | None]:
    """Decode ``path``, returning (sampled_frames, thumbnail_frame).

    The frame list contains at most ``n_target`` PIL RGB images,
    sampled uniformly across the timeline. The thumbnail frame is
    the first decoded frame whose presentation timestamp lands at
    or after ``thumb_at_s`` (or the very first frame for sub-1-s
    clips).

    Empty list is a valid return — caller decides what to do with it.
    """
    try:
        import av

        with av.open(str(path)) as c:
            video_streams = list(c.streams.video)
            if not video_streams:
                return [], None
            stream = video_streams[0]
            total = stream.frames or 0
            if total <= 0:
                # Some MP4 containers don't store the stream frame count;
                # fall back to a duration × fps estimate so we still hit
                # uniform sampling.
                dur = _container_duration(c) or 0.0
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                total = max(int(round(dur * fps)), n_target)
            target_indices = sorted(
                set(int(round(i * (total - 1) / max(n_target - 1, 1))) for i in range(n_target))
            )

            frames: list = []
            thumb: PILImage | None = None
            i = 0
            t_idx = 0
            time_base = float(stream.time_base) if stream.time_base else 0.0

            for frame in c.decode(video=0):
                # Thumbnail seed
                if thumb is None:
                    pts = frame.pts
                    if pts is not None and time_base > 0:
                        t = pts * time_base
                        if t >= thumb_at_s or i == 0:
                            thumb = frame.to_image().convert("RGB")
                    elif i == 0:
                        thumb = frame.to_image().convert("RGB")
                # Uniform sample
                while t_idx < len(target_indices) and i >= target_indices[t_idx]:
                    frames.append(frame.to_image().convert("RGB"))
                    t_idx += 1
                if t_idx >= len(target_indices):
                    break
                i += 1

            # If the inner-most while didn't catch the last bucket (e.g.
            # round-up index exceeded the real frame count) pad with
            # the last frame we did capture.
            if frames and len(frames) < n_target:
                frames.extend([frames[-1]] * (n_target - len(frames)))
            if thumb is None and frames:
                thumb = frames[0]
            return frames, thumb
    except Exception as e:
        log.warning("PyAV decode failed on %s: %s", path, e)
        return [], None


# === Thumbnail ==============================================================


def _write_thumbnail(frame: PILImage, doc_id: str) -> Path | None:
    """Persist the seed frame as a 128-px short-edge WebP."""
    from . import image as image_mod  # reuse the resize helper

    settings.ensure_data_dirs()
    target = settings.THUMBNAILS_DIR / f"{doc_id}.webp"
    thumb = image_mod._resize_short_edge(frame, THUMBNAIL_SHORT_EDGE)
    try:
        thumb.save(target, format="WEBP", quality=THUMBNAIL_QUALITY)
    except OSError as e:
        log.warning("video thumbnail write failed for %s: %s", doc_id, e)
        return None
    return target
