"""Text ingestion pipeline.

Pipeline per file::

    detect → read (utf-8 → latin-1 → skip)
           → empty / oversize gate
           → chunk  (markdown by `#`, plaintext by blank lines)
           → for each chunk:
                embed_document(filename + "\\n\\n" + chunk)
           → return rows ready for `table.add(...)`

The dispatcher in ``ingest/__init__.py`` calls this module's
``ingest_file`` based on the file's extension; file-walking and the
top-level ``ingest_path`` facade live there.

Chunking heuristics:

- Soft target: ~600 tokens per chunk
- Hard ceiling: 2000 tokens (the only time we split *inside* a single
  heading block, by collapsing on blank lines)
- No overlap (jina v5 omni is last-token pooled — each chunk is
  semantically self-contained)
- Fenced code blocks are kept intact even if they exceed the soft target
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from .. import settings
from ..embed import embed_document, embed_documents_batch, load
from ..store import is_doc_indexed, make_row, path_cache

log = logging.getLogger(__name__)

# === Tunables ===============================================================

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".org"})
MAX_FILE_BYTES: int = 10 * 1024 * 1024
SHORT_CHAR_THRESHOLD: int = 1000

# Chunk-size knobs live in ``rubick_backend.settings`` (mutable at runtime
# via ``PATCH /settings``). Read inside ``_greedy_pack`` so each ingest
# job sees fresh values.


# === Public API =============================================================


def ingest_file(path: Path | str) -> list[dict[str, Any]]:
    """Process one text file. Returns a list of rows ready for ``table.add``.

    Returns ``[]`` (with a warn log) when the file is skipped — wrong
    extension, oversize, empty, undecodable. Never raises for per-file
    errors; that's the writer coroutine's responsibility to handle.
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
    if size > MAX_FILE_BYTES:
        log.warning("skip %s — too large (%d bytes)", p, size)
        return []

    mtime = int(st.st_mtime)

    # Fast-path dedup: skip if path+mtime match a cached entry still in LanceDB.
    cached_doc_id = path_cache.lookup(str(p), mtime)
    if cached_doc_id and is_doc_indexed(cached_doc_id):
        log.debug("fast-skip %s — path+mtime cache hit", p)
        return []

    raw = _read_text(p)
    if raw is None:
        return []
    if not raw.strip():
        log.info("skip %s — empty content", p)
        return []

    file_bytes = p.read_bytes()
    sha = hashlib.sha256(file_bytes).hexdigest()
    doc_id = sha[:16]
    filename = p.stem

    # Content dedup: skip if this exact content is already indexed.
    # An edited file gets a new sha → new doc_id → re-ingested fresh
    # (old row remains until GC).
    if is_doc_indexed(doc_id):
        log.info("skip %s — doc_id=%s already indexed", p, doc_id)
        path_cache.record(str(p), mtime, doc_id)
        return []

    is_md = p.suffix.lower() in {".md", ".markdown"}
    chunks = chunk_text(raw, is_markdown=is_md)
    log.info("ingest %s — %d chunk(s)", p, len(chunks))

    rows: list[dict[str, Any]] = []
    # Prepare all chunk inputs
    doc_inputs = []
    for chunk in chunks:
        doc_input = f"{filename}\n\n{chunk}" if filename else chunk
        doc_inputs.append(doc_input)

    # Batch embed all chunks at once (5-10x faster than one-at-a-time)
    try:
        if len(doc_inputs) == 1:
            vecs = [embed_document(doc_inputs[0])]
        else:
            batch_result = embed_documents_batch(doc_inputs)
            vecs = [batch_result[i] for i in range(len(doc_inputs))]
    except Exception as e:
        log.warning("batch embed failure on %s: %s — falling back to per-chunk", p, e)
        vecs = []
        for doc_input in doc_inputs:
            try:
                vecs.append(embed_document(doc_input))
            except Exception as e2:
                log.warning("embed failure on %s: %s", p, e2)
                vecs.append(None)

    for idx, (chunk, vec) in enumerate(zip(chunks, vecs, strict=True)):
        if vec is None:
            continue
        doc_input = doc_inputs[idx]
        try:
            n_tokens = _count_tokens(doc_input)
        except Exception:
            n_tokens = None
        rows.append(
            make_row(
                doc_id=doc_id,
                modality="text",
                chunk_idx=idx,
                embedding=vec.tolist(),
                file_path=str(p),
                sha256=sha,
                mtime=mtime,
                filename=filename,
                raw_text=chunk[:500],
                chunk_n_tokens=n_tokens,
            )
        )
    if rows:
        path_cache.record(str(p), mtime, doc_id)
    return rows


# === Chunkers ===============================================================


def chunk_text(text: str, *, is_markdown: bool) -> list[str]:
    """Split text into chunks (markdown vs plain dispatch)."""
    if len(text) < SHORT_CHAR_THRESHOLD:
        return [text.strip()]
    if is_markdown:
        return _chunk_markdown(text)
    return _chunk_plaintext(text)


def _chunk_markdown(text: str) -> list[str]:
    blocks = _split_on_headings(text)
    return _greedy_pack(blocks)


def _chunk_plaintext(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _greedy_pack(paragraphs)


_HEADING_RE = re.compile(r"^#{1,6}\s")
_FENCE_RE = re.compile(r"^\s*```")


def _split_on_headings(text: str) -> list[str]:
    """Split markdown on ``^# `` boundaries, ignoring ``#`` lines that
    appear inside a fenced code block.
    """
    in_fence = False
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and _HEADING_RE.match(line):
            if current:
                joined = "".join(current).strip()
                if joined:
                    blocks.append(joined)
            current = [line]
        else:
            current.append(line)
    if current:
        joined = "".join(current).strip()
        if joined:
            blocks.append(joined)
    return blocks


def _greedy_pack(blocks: list[str]) -> list[str]:
    """Greedy accumulate blocks until next would exceed
    ``settings.TARGET_TOKENS``; enforce ``settings.HARD_MAX_TOKENS``
    as a backstop.

    The two thresholds are read from ``settings`` at call time (v0.0.2
    M1-5) rather than snapshotted into module-level locals, so a
    ``PATCH /settings`` takes effect on the very next ingest job
    without any process restart. The cost — two attribute lookups
    per call — is negligible against the per-block tokenization cost.
    """
    target_tokens = settings.TARGET_TOKENS
    hard_max_tokens = settings.HARD_MAX_TOKENS
    result: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for b in blocks:
        bt = _count_tokens(b)
        if buf and (buf_tokens + bt) > target_tokens:
            result.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(b)
        buf_tokens += bt
        if buf_tokens >= hard_max_tokens:
            result.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
    if buf:
        result.append("\n\n".join(buf))
    return result


# === I/O helpers ============================================================


def _read_text(p: Path) -> str | None:
    """UTF-8 (BOM-tolerant) → latin-1 → give up."""
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    log.warning("could not decode %s as utf-8 or latin-1; skipping", p)
    return None


def _count_tokens(text: str) -> int:
    """Use jina's tokenizer for accurate token counts.

    Triggers model load on first call — fine because the same model load
    is required for embedding anyway, so we're not paying any new cost.
    """
    return len(load().tokenizer.encode(text).ids)
