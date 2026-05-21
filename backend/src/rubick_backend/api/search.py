"""``GET /search`` (text-only hybrid) and ``POST /search`` (fused image query).

Hybrid retrieval (``GET /search``):

- Vector ANN top-50 (cosine) + BM25 FTS top-50 on ``raw_text`` + ``filename``
- RRF fusion (k=60)
- Doc-level fold to K docs (default 20, max 50)
- Default ``modality != 'rejected'`` filter

Fused multimodal (``POST /search``):

- Multipart form: optional ``q`` + required ``image`` attachment
- ``embed_query_fused_weighted``: ``L2_norm(α·text_vec + (1−α)·image_vec)``
  where α = ``text_weight`` (0–1, default 0.5); empty ``q`` → image forward only
- **Skips BM25** — attached images query by visual/semantic similarity
- Video upload is not wired on the API surface in v1.x

Embedding and LanceDB run on worker threads so MLX / native I/O do not
block the event loop.

JSON response (additive; Swift ``SearchHit`` decodes with fallbacks):

- ``similarity`` — best-chunk cosine (back-compat)
- ``score_rrf``, ``score_vector``, ``score_bm25``, ``hit_count``
- ``took_ms``: ``embed``, ``search``, ``total``
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import asdict
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from ..embed import embed_query, embed_query_fused_weighted
from ..retrieve import hybrid_search

log = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


# === Fused-query upload caps =================================================
#
# v1.x is image-only on the wire. The cap matches what users typically
# drag in from Photos / Finder; the model's processor down-samples
# internally so anything larger than ~10 MB doesn't add quality.
_MAX_IMAGE_BYTES: int = 30 * 1024 * 1024  # 30 MB


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="natural-language query"),
    limit: int = Query(20, ge=1, le=50, description="max docs returned (post-fold)"),
    modality: str | None = Query(
        None,
        description="optional modality filter, e.g. 'text'",
    ),
    include_rejected: bool = Query(
        False,
        description="include rows with modality='rejected' (oversize files). Default: hide.",
    ),
    path_prefix: str | None = Query(
        None,
        description=(
            "keep only docs whose canonical first path starts with this prefix "
            "(matches the sidebar 'limit to folder' facet)."
        ),
    ),
    mtime_after: int | None = Query(
        None,
        ge=0,
        description=(
            "keep only docs with file mtime >= this POSIX epoch (seconds). "
            "Combine with mtime_before for a range."
        ),
    ),
    mtime_before: int | None = Query(
        None,
        ge=0,
        description=(
            "keep only docs with file mtime <= this POSIX epoch (seconds). "
            "Must be >= mtime_after when both are set."
        ),
    ),
) -> dict[str, Any]:
    """Hybrid-search the local index (text-only query).

    Response shape (additive — older clients ignore unknown fields)::

        {
          "query": "<q>",
          "count": <int>,
          "took_ms": { "embed": <float>, "search": <float>, "total": <float> },
          "results": [
            {
              "id": "<doc_id>-<modality>-<chunk_idx>",   # best chunk
              "doc_id": "<16 hex>",
              "modality": "...",
              "chunk_idx": <int>,
              "file_paths": ["<path>", ...],
              "filename": "<name without ext>",
              "raw_text": "<chunk preview>" | null,
              "thumbnail_path": "<path>" | null,
              "similarity": <cosine in [-1, 1]>,          # back-compat
              "score_rrf": <float>,                        # NEW
              "score_vector": <float> | null,              # NEW
              "score_bm25": <float> | null,                # NEW
              "hit_count": <int>                           # NEW
            },
            ...
          ]
        }
    """
    t_total = time.perf_counter()

    try:
        t_embed = time.perf_counter()
        qvec = await asyncio.to_thread(embed_query, q)
        embed_ms = (time.perf_counter() - t_embed) * 1000
    except Exception as e:
        log.exception("embed_query failed for q=%r", q)
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}") from e

    return await _run_search(
        qvec=qvec,
        qtext=q,
        q_repr=q,
        limit=limit,
        modality=modality,
        include_rejected=include_rejected,
        path_prefix=path_prefix,
        mtime_after=mtime_after,
        mtime_before=mtime_before,
        embed_ms=embed_ms,
        t_total=t_total,
    )


# === POST /search — fused multimodal query ===================================


@router.post("/search")
async def search_fused(
    q: str = Form("", description="optional natural-language text part of the query"),
    text_weight: float = Form(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the text component in the fused query vector "
            "(0.0 = pure image, 1.0 = pure text, 0.5 = equal). "
            "Ignored when q is empty."
        ),
    ),
    limit: int = Form(20, ge=1, le=50),
    modality: str | None = Form(None),
    include_rejected: bool = Form(False),
    path_prefix: str | None = Form(None),
    mtime_after: int | None = Form(None),
    mtime_before: int | None = Form(None),
    image: Annotated[
        UploadFile | None,
        File(description="image attachment (PNG / JPEG / HEIC / …); required."),
    ] = None,
) -> dict[str, Any]:
    """Fused multimodal query — image attachment plus optional text.

    Two valid shapes:

    1. ``q`` empty + image: pure image-as-query. ``text_weight`` is
       ignored; only the image forward runs, producing a vector
       identical to the indexed document row (so the image's own entry
       lands at top-1).
    2. ``q`` non-empty + image: weighted T+I query. Runs two independent
       model forwards — text with ``"Query: "`` prefix and image pure —
       then combines: ``L2_norm(text_weight * text_vec +
       (1-text_weight) * image_vec)``. ``text_weight=0.5`` (default)
       gives equal geometric contribution to both modalities.

    Sending *no* image is a 400 (text-only queries belong on
    ``GET /search``). Video attachments are not wired in v1.x.

    **Skips the BM25 leg by design.** When the user attaches an image
    they're asking for visual / semantic similarity, not lexical
    file-name matches. Retrieval here is pure vector ANN.
    """
    t_total = time.perf_counter()

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "POST /search requires an image attachment. "
                "Use GET /search for text-only queries."
            ),
        )

    try:
        qvec, embed_ms = await _embed_fused_image(
            text=q, text_weight=text_weight, upload=image
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("fused embed failed for image=%r", image.filename)
        raise HTTPException(
            status_code=500, detail=f"embedding failed: {e}"
        ) from e

    # Pure vector retrieval — see module docstring on why BM25 is skipped.
    return await _run_search(
        qvec=qvec,
        qtext=None,
        q_repr=f"{q!r} + image:{image.filename!r} (text_weight={text_weight:.2f})",
        limit=limit,
        modality=modality,
        include_rejected=include_rejected,
        path_prefix=path_prefix,
        mtime_after=mtime_after,
        mtime_before=mtime_before,
        embed_ms=embed_ms,
        t_total=t_total,
    )


# === Internal helpers ======================================================


async def _run_search(
    *,
    qvec: np.ndarray,
    qtext: str | None,
    q_repr: str,
    limit: int,
    modality: str | None,
    include_rejected: bool,
    path_prefix: str | None,
    mtime_after: int | None,
    mtime_before: int | None,
    embed_ms: float,
    t_total: float,
) -> dict[str, Any]:
    """Run hybrid_search + build the JSON envelope, shared by GET / POST."""
    try:
        t_search = time.perf_counter()
        results = await asyncio.to_thread(
            hybrid_search,
            qvec=qvec,
            qtext=qtext,
            recall=50,
            doc_limit=limit,
            modality=modality,
            include_rejected=include_rejected,
            path_prefix=path_prefix,
            mtime_after=mtime_after,
            mtime_before=mtime_before,
        )
        search_ms = (time.perf_counter() - t_search) * 1000
    except ValueError as e:
        # ``_build_where`` raises on illegal modality filters,
        # negative mtime, or contradictory mtime range.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("hybrid_search failed for q=%s", q_repr)
        raise HTTPException(status_code=500, detail=f"search failed: {e}") from e

    total_ms = (time.perf_counter() - t_total) * 1000
    return {
        "query": qtext if qtext is not None else "",
        "count": len(results),
        "took_ms": {
            "embed": round(embed_ms, 1),
            "search": round(search_ms, 1),
            "total": round(total_ms, 1),
        },
        "results": [asdict(r) for r in results],
    }


async def _embed_fused_image(
    *, text: str, text_weight: float, upload: UploadFile
) -> tuple[np.ndarray, float]:
    """Read + decode + embed one image attachment.

    Returns ``(qvec, embed_ms)``. Decode runs on a worker thread so
    the event loop stays responsive even on a 30 MB upload.
    """
    t_embed = time.perf_counter()
    payload = await upload.read()
    if not payload:
        raise HTTPException(
            status_code=400, detail="image attachment is empty"
        )
    if len(payload) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"image attachment is {len(payload):,} bytes; "
                f"max {_MAX_IMAGE_BYTES:,}."
            ),
        )
    qvec = await asyncio.to_thread(
        _embed_query_image_from_bytes, text, payload, text_weight
    )
    embed_ms = (time.perf_counter() - t_embed) * 1000
    return qvec, embed_ms


def _embed_query_image_from_bytes(
    text: str, payload: bytes, text_weight: float
) -> np.ndarray:
    """Decode ``payload`` as a PIL RGB image and run weighted fused embedding.

    Mirrors the ingest-side ``image.py`` decode path (HEIC opener
    registered once at module import on the ingest side; safe to
    re-trigger here through the same helper).
    """
    from PIL import Image

    from ..ingest.image import _ensure_heif_opener_registered

    _ensure_heif_opener_registered()
    img = Image.open(io.BytesIO(payload))
    img = img.convert("RGB")
    return embed_query_fused_weighted(text=text, image=img, alpha=text_weight)
