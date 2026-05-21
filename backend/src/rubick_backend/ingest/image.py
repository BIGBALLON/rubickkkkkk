"""Image ingestion pipeline.

Pipeline per file::

    detect → read bytes → sha256 / doc_id
           → tiny-file gate (< 5 KB)
           → PIL.Image.open (decode failures skipped + warned)
           → exif_transpose so the embedding sees the image upright
           → tiny-pixel gate (< 64×64)
           → record (orig_w, orig_h)  — *original* dimensions
           → if long edge > 1280 → resize so short edge == 768 (Lanczos)
           → write 128-px short-edge WebP thumbnail to ``thumbnails/<doc_id>.webp``
           → extract EXIF DateTimeOriginal (best-effort, silently skipped on failure)
           → embed_image(rgb_image)
           → emit a single LanceDB row with modality="image", chunk_idx=0

Design choices that aren't obvious:

- We auto-rotate via ``ImageOps.exif_transpose`` *before* both thumbnail
  generation and embedding so a portrait iPhone photo doesn't appear
  rotated 90° to the model. (jina v5 omni was trained on already-
  oriented natural images.)
- The filename is **not** part of the embedding input — we avoid mixing
  text tokens with image patches in the same forward pass. Filename
  searchability comes from the BM25 FTS index on ``filename``, not
  from the embedding forward.
- The thumbnail is generated from the *post-resize* image (same source
  as the embedding input after oversize shrink).
- HEIC support requires ``pillow-heif`` (Pillow 12 still doesn't ship
  native HEIF). The opener is registered lazily, so text-only ingest
  paths don't pay the import cost.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .. import settings
from ..embed import embed_image_preprocessed, load
from ..store import is_doc_indexed, make_row, path_cache

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger(__name__)

# === Tunables ===============================================================

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".heic",
        ".heif",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
    }
)

# Tiny-image filter — knocks out icons, emoji placeholders, tracking pixels, etc.
MIN_PIXEL_DIMENSION: int = 64
MIN_FILE_BYTES: int = 5 * 1024

# Oversize handling: shrink before embedding when the long edge exceeds the threshold.
# The model's preprocessing caps at ~1.3M pixels (1144×1144); we pre-shrink
# to avoid wasting time on large-image preprocessing + more GPU tokens.
LARGE_LONG_EDGE_THRESHOLD: int = 1280
RESIZE_SHORT_EDGE: int = 768

# WebP thumbnail (128-px short edge).
THUMBNAIL_SHORT_EDGE: int = 128
THUMBNAIL_QUALITY: int = 75

# EXIF DateTimeOriginal only — no GPS / camera / lens metadata in v1.
_EXIF_DATETIME_ORIGINAL: int = 0x9003

# === HEIF opener (lazy, one-time) ===========================================

_heif_lock = threading.Lock()
_heif_registered: bool = False


def _ensure_heif_opener_registered() -> None:
    """Register the pillow-heif HEIF opener exactly once per process.

    Idempotent and thread-safe. Failure (e.g. pillow-heif not
    installed) downgrades to a warning rather than an exception so a
    text-only ingest still completes — the actual HEIC file in the
    walk will simply hit ``UnidentifiedImageError`` later and be
    skipped with a log line.
    """
    global _heif_registered
    if _heif_registered:
        return
    with _heif_lock:
        if _heif_registered:
            return
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            _heif_registered = True
            log.info("pillow-heif opener registered (HEIC/HEIF enabled)")
        except ImportError:
            log.warning("pillow_heif not installed; HEIC/HEIF files will be skipped")
            # Don't flip the flag — re-trying once another HEIC shows
            # up is a cheap no-op, and if the user pip-installs
            # pillow-heif while the app is running we'll pick it up
            # next call.


# === Public API =============================================================


def ingest_file(path: Path | str) -> list[dict[str, Any]]:
    """Process one image file; return a list of rows ready for ``table.add``.

    Returns ``[]`` (with a log line) for any skip path — wrong
    extension, too-small file, too-small pixels, decode failure,
    embed failure. Never raises for per-file errors — the writer
    coroutine expects ``[]`` on skip paths, not exceptions.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        log.warning("skip %s — not a regular file", p)
        return []
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.warning("skip %s — unsupported extension", p)
        return []

    st = p.stat()
    size = st.st_size
    if size < MIN_FILE_BYTES:
        log.info("skip %s — too small (%d bytes)", p, size)
        return []

    mtime = int(st.st_mtime)
    filename = p.stem

    # Fast-path dedup: if path+mtime match a cached entry that is still in
    # the LanceDB table, skip the file entirely — no bytes read, no sha256.
    cached_doc_id = path_cache.lookup(str(p), mtime)
    if cached_doc_id and is_doc_indexed(cached_doc_id):
        log.debug("fast-skip %s — path+mtime cache hit", p)
        return []

    # Lazily register the HEIF opener only when we might actually
    # need it. Cheap once flipped on.
    if p.suffix.lower() in {".heic", ".heif"}:
        _ensure_heif_opener_registered()

    file_bytes = p.read_bytes()
    sha = hashlib.sha256(file_bytes).hexdigest()
    doc_id = sha[:16]

    # Content dedup: skip if this sha256/doc_id is already indexed.
    if is_doc_indexed(doc_id):
        log.info("skip %s — doc_id=%s already indexed", p, doc_id)
        path_cache.record(str(p), mtime, doc_id)  # warm cache for next restart
        return []

    try:
        img = Image.open(io.BytesIO(file_bytes))
        # ``exif_transpose`` returns None for "no orientation tag" in
        # some Pillow versions; coalesce back to the original.
        oriented = ImageOps.exif_transpose(img)
        if oriented is not None:
            img = oriented
    except (UnidentifiedImageError, OSError) as e:
        log.warning("decode failure on %s: %s", p, e)
        return []

    orig_w, orig_h = img.size
    if orig_w < MIN_PIXEL_DIMENSION or orig_h < MIN_PIXEL_DIMENSION:
        log.info("skip %s — too small (%dx%d)", p, orig_w, orig_h)
        return []

    exif_taken_at = _extract_exif_taken_at(img)

    # The model expects 3-channel RGB; this also lets WebP encode the
    # thumbnail without complaining about palette / RGBA quirks.
    if img.mode != "RGB":
        img = img.convert("RGB")

    if max(orig_w, orig_h) > LARGE_LONG_EDGE_THRESHOLD:
        img = _resize_short_edge(img, RESIZE_SHORT_EDGE)

    thumbnail_path = _make_thumbnail(img, doc_id)

    try:
        # Preprocess on this thread (CPU work), then submit only the
        # GPU forward pass to the model executor. This enables overlap:
        # while the executor runs encode_image for the previous file,
        # this thread can already be reading + preprocessing the next.
        from ..embed.preprocessing import preprocess_image

        state = load()
        preprocessed = preprocess_image(img, text_prefix="", tokenizer=state.tokenizer)
        vec = embed_image_preprocessed(preprocessed)
    except Exception as e:
        log.warning("embed failure on %s: %s", p, e)
        return []

    path_cache.record(str(p), mtime, doc_id)
    return [
        make_row(
            doc_id=doc_id,
            modality="image",
            chunk_idx=0,
            embedding=vec.tolist(),
            file_path=str(p),
            sha256=sha,
            mtime=mtime,
            filename=filename,
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            width=orig_w,
            height=orig_h,
            exif_taken_at=exif_taken_at,
        )
    ]


# === Helpers ================================================================


def _resize_short_edge(img: PILImage, target_short: int) -> PILImage:
    """Lanczos resize so the *shorter* edge equals ``target_short``.

    Aspect-preserving; both edges scale by the same factor. Used for
    oversize shrink and thumbnail generation (``target_short`` varies).
    """
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if w < h:
        new_w = target_short
        new_h = max(1, int(round(h * target_short / w)))
    else:
        new_h = target_short
        new_w = max(1, int(round(w * target_short / h)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _make_thumbnail(img: PILImage, doc_id: str) -> Path | None:
    """Generate + persist a 128-px short-edge WebP thumbnail.

    Returns the on-disk path on success, or ``None`` if the write
    failed (logged at warn). Failure is non-fatal — the row still
    embeds and stores, we just won't have a thumbnail to render.
    """
    settings.ensure_data_dirs()
    target = settings.THUMBNAILS_DIR / f"{doc_id}.webp"
    thumb = _resize_short_edge(img, THUMBNAIL_SHORT_EDGE)
    try:
        thumb.save(target, format="WEBP", quality=THUMBNAIL_QUALITY)
    except OSError as e:
        log.warning("thumbnail write failed for %s: %s", doc_id, e)
        return None
    return target


def _extract_exif_taken_at(img: PILImage) -> int | None:
    """Best-effort EXIF ``DateTimeOriginal`` → Unix timestamp.

    Parse failure / missing tag / un-parseable date all silently return
    silently return None. We deliberately don't catch a narrow
    exception class because Pillow's EXIF parser raises a moving
    target across versions (ValueError, KeyError, struct errors…).
    """
    try:
        exif = img.getexif()
        if not exif:
            return None
        dt_str = exif.get(_EXIF_DATETIME_ORIGINAL)
        if not dt_str:
            return None
        # EXIF DateTimeOriginal format is "YYYY:MM:DD HH:MM:SS".
        # Some cameras emit subsecond suffixes — strip them via slice.
        dt_str = str(dt_str).strip()[:19]
        dt = _dt.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return None
