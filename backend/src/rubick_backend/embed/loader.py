"""Single-instance loader for the jina v5 omni nano multimodal embedder.

Design (see ARCHITECTURE.md — Process model):

- Query text uses a ``"Query: "`` prefix; indexed documents use
  ``"Document: "`` — see ``embed_query`` / ``embed_document``.

- One Python process holds **one** MLX model instance — multiple workers
  would each load a 1.8 GB model and instantly blow past 16 GB RAM.
- The loader is module-level and lazy: the first call to ``load()``
  resolves & downloads weights (via ``huggingface_hub.snapshot_download``),
  loads them into MLX, then caches the result for the lifetime of the
  process. Subsequent calls are O(1).
- MLX inference is **synchronous and blocking**; callers in async code
  must wrap calls in ``asyncio.to_thread`` (this is the embedder
  coroutine's job — see ``worker/embedder.py`` once that exists).

The model is shipped with a custom ``model.py`` next to the weights; we
insert the repo dir into ``sys.path`` and discover the
``Config`` / ``EmbeddingModel`` classes by reflection so we stay
compatible with both ``nano`` and ``small`` variants.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import settings
from .executor import Priority, get_executor

if TYPE_CHECKING:
    from PIL.Image import Image

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """In-memory handle to the loaded MLX model + its tokenizer.

    ``processor`` (the historical ``transformers.AutoProcessor``) is
    gone — image / video preprocessing is now hand-rolled in pure
    PIL + numpy via ``embed.preprocessing``. The ``transformers``
    package is no longer a runtime dependency.
    """

    model: Any
    tokenizer: Any
    repo_dir: Path
    config_cls_name: str
    model_cls_name: str


_loaded: LoadedModel | None = None
_load_lock = threading.Lock()

# === Memory bounds for ingest workloads =====================================
#
# Indexing many-thousand-image folders blows memory on Apple Silicon because
# MLX's Metal allocator caches every per-frame buffer until the process
# exits. Capping the cache at a fixed budget + periodically calling
# ``clear_inference_cache()`` between batches keeps RSS bounded for the
# multi-thousand-file ingest scenarios that actually crashed users on
# 16/32 GB Macs. ``set_cache_limit`` is applied once at model load.
#
# 2 GiB is comfortably under the smallest currently-supported Mac (16 GB
# Air) while still letting one ingest forward keep its working set warm.
_MLX_CACHE_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024


def clear_inference_cache() -> None:
    """Drop MLX's Metal cache + run a Python GC pass.

    Safe to call from any thread; safe to call after every N ingest
    files to bound peak memory during long folders. No-op (with a
    debug log) on non-Metal platforms or older MLX builds where the
    Metal cache helpers aren't exposed.
    """
    import gc

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception as e:  # noqa: BLE001 — diagnostic only, never abort
        log.debug("mlx clear_cache unavailable: %s", e)
    gc.collect()


# === Public API =============================================================


def load() -> LoadedModel:
    """Resolve, download, and load the model. Cached after first call."""
    global _loaded
    if _loaded is not None:
        return _loaded
    with _load_lock:
        if _loaded is not None:
            return _loaded
        _loaded = _do_load()
        return _loaded


def is_loaded() -> bool:
    """``True`` iff the main model singleton has been hydrated this process.

    Used by ``GET /healthz/model`` to expose backend-perspective state
    that the file-system-only Settings → Model tab can't observe. Cheap
    (no MLX / HF imports), safe to call from request handlers.
    """
    return _loaded is not None


# === Priority-aware public API =============================================
#
# Every embed_* call goes through the process-wide ``ModelExecutor`` so the
# foreground search path (HIGH) preempts background indexing (LOW) at the
# MLX-model boundary — see ARCHITECTURE.md (Process model).
#
# The public functions still look synchronous (caller blocks on .result());
# the priority decision is local to each wrapper. Splitting "raw" vs
# "exported" lets tests bypass the executor for fast unit testing.


def embed_text(text: str) -> np.ndarray:
    """Embed a raw text string. Returns an L2-normalized 768-dim float32 vector.

    Prefer ``embed_query`` / ``embed_document`` over this — they apply the
    jina v5 omni prefix convention which materially affects retrieval
    quality. Uses HIGH priority; direct ``embed_text`` callers are
    typically tests where waiting behind ingest would be surprising.
    """
    return get_executor().call(Priority.HIGH, _raw_embed_text, text)


def embed_query(query: str) -> np.ndarray:
    """Embed a text query. Applies the mandatory ``"Query: "`` prefix.

    HIGH priority — this is the foreground search path. When ingest is
    actively running, ``embed_query`` slots ahead of pending ingest
    chunks (ARCHITECTURE.md — Process model).
    """
    return get_executor().call(Priority.HIGH, _raw_embed_text, f"Query: {query}")


def embed_document(text: str) -> np.ndarray:
    """Embed a text document chunk. Applies the mandatory ``"Document: "`` prefix.

    Callers typically prepend filename context before calling::

        Document: <filename without ext>

        <chunk content>

    This function only adds the ``"Document: "`` token; the filename header
    is the caller's responsibility (so it can also be omitted for
    non-file text payloads). LOW priority — yields the
    model to ``embed_query`` when both are pending.
    """
    return get_executor().call(Priority.LOW, _raw_embed_text, f"Document: {text}")


def embed_documents_batch(texts: list[str]) -> np.ndarray:
    """Embed multiple document chunks in one batched forward pass.

    Each text gets the ``"Document: "`` prefix prepended. Returns
    shape (N, 768) normalized float32 array. LOW priority.

    Significantly faster than calling ``embed_document`` N times for
    text-heavy folders (5-10x speedup from batching).
    """
    prefixed = [f"Document: {t}" for t in texts]
    return get_executor().call(Priority.LOW, _raw_embed_text_batch, prefixed)


def embed_image(image: Image) -> np.ndarray:
    """Embed a PIL image. Returns an L2-normalized 768-dim float32 vector.

    LOW priority (ingest path)."""
    return get_executor().call(Priority.LOW, _raw_embed_image, image)


def embed_image_preprocessed(preprocessed: dict) -> np.ndarray:
    """Embed an already-preprocessed image dict (from preprocess_image).

    Skips the preprocessing step — used by the prefetch pipeline to
    overlap CPU preprocessing with GPU forward pass.
    LOW priority (ingest path).
    """
    return get_executor().call(Priority.LOW, _raw_embed_image_from_preprocessed, preprocessed)


def embed_video(frames: list) -> np.ndarray:
    """Embed a list of PIL frames (must be ``len % 2 == 0``).

    The jina v5 omni video path uses ``temporal_patch_size=2``; an odd
    frame count would crash inside the model, so we silently drop the
    last frame (consistent with spike-0 behaviour). LOW priority
    (ingest path)."""
    return get_executor().call(Priority.LOW, _raw_embed_video, frames)


# === Fused query path =======================================================
#
# Two fusion strategies are available:
#
# embed_query_fused_weighted (default, recommended)
# -------------------------------------------------
# Runs two independent model forwards — text via encode_text (with the
# mandatory "Query: " retrieval prefix) and image via encode_image (pure,
# no text prefix, matching the document side) — then combines:
#
#     qvec = L2_norm( alpha * text_vec + (1-alpha) * image_vec )
#
# alpha=0.0 → pure image query (identical to what is indexed, so the
# image's own document row lands at top-1). alpha=1.0 → pure text query.
# alpha=0.5 (default) → equal geometric weight.
#
# This approach gives explicit, predictable control over text influence
# and is robust to the token-count imbalance that makes the single-pass
# approach (below) insensitive to text: a typical image produces 256-1300
# image tokens versus 5-15 text tokens, so in a single encode_image
# forward the EOS representation is numerically dominated by visual
# features regardless of text prefix length.
#
# embed_query_fused (kept for backward-compat / tests)
# ----------------------------------------------------
# Single encode_image forward with text prepended as a prefix to the
# image placeholder tokens. Preserved because tests and the e2e suite
# reference it directly and because pure-image degradation (alpha=0 path)
# needs no text forward at all. Not recommended for T+I queries.


def embed_query_fused_weighted(
    *,
    text: str | None = None,
    image: Image | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Embed a fused (text + image) query via weighted vector combination.

    Runs **two** independent model forwards (text and image) then combines:
    ``L2_norm(alpha * text_vec + (1-alpha) * image_vec)``.

    ``alpha`` controls text vs image weight (0.0 = pure image, 1.0 = pure
    text). Text is embedded with the ``"Query: "`` prefix so it lives in
    the same retrieval-query half-space as indexed text documents.

    When ``text`` is absent/empty, skips the text forward entirely and
    returns the pure image embedding (same as the document-side ingest
    result, so the image's own indexed row still lands at top-1).

    Both forwards run inside one executor call so they share the single
    MLX thread — no preemption hazard between the two.

    HIGH priority (foreground search path).
    """
    if image is None:
        raise ValueError("embed_query_fused_weighted requires an image")
    text_stripped = (text or "").strip()
    if not text_stripped:
        return get_executor().call(Priority.HIGH, _raw_embed_image, image)
    return get_executor().call(
        Priority.HIGH, _raw_embed_fused_weighted, text_stripped, image, alpha
    )


def embed_query_fused(
    *,
    text: str | None = None,
    image: Image | None = None,
    video_frames: list | None = None,
) -> np.ndarray:
    """Single-pass fused query (text prefix + image tokens in one forward).

    Kept for backward compatibility and tests. For T+I queries prefer
    ``embed_query_fused_weighted`` — it gives text a predictable and
    visible influence on the result vector.

    Exactly **one** of ``image`` / ``video_frames`` must be supplied.
    ``text`` is optional; empty text degrades to a pure-media embedding.

    HIGH priority (foreground search path).
    """
    attachments = [
        ("image", image),
        ("video", video_frames),
    ]
    set_attachments = [name for name, val in attachments if val is not None]
    if len(set_attachments) == 0:
        raise ValueError(
            "embed_query_fused requires one of image / video_frames"
        )
    if len(set_attachments) > 1:
        raise ValueError(
            "embed_query_fused accepts exactly one attachment "
            f"(got: {set_attachments})"
        )

    text_prefix = (text or "").strip()

    if image is not None:
        return get_executor().call(
            Priority.HIGH, _raw_embed_query_image, text_prefix, image
        )
    # video_frames branch
    return get_executor().call(
        Priority.HIGH, _raw_embed_query_video, text_prefix, video_frames
    )


# === Raw (unprioritized) implementations ====================================
#
# These do the actual MLX work. Don't call them directly from production
# code — always go through the priority-aware wrappers above. Tests that
# only care about correctness (not scheduling) may call them to skip
# the executor thread hop.


def _raw_embed_text(text: str) -> np.ndarray:
    import mlx.core as mx

    state = load()
    enc = state.tokenizer.encode(text)
    input_ids = mx.array([enc.ids])
    attn = mx.array([enc.attention_mask])
    emb = state.model.encode_text(input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


def _raw_embed_text_batch(texts: list[str]) -> np.ndarray:
    """Embed multiple texts in one batched forward pass (padded + stacked).

    Returns shape (N, 768) normalized float32.
    """
    import mlx.core as mx

    state = load()
    encs = [state.tokenizer.encode(t) for t in texts]
    max_len = max(len(e.ids) for e in encs)
    pad_id = state.tokenizer.token_to_id("<|endoftext|>") or 0
    input_ids = mx.array(
        [e.ids + [pad_id] * (max_len - len(e.ids)) for e in encs]
    )
    attn = mx.array(
        [e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in encs]
    )
    emb = state.model.encode_text(input_ids, attn)  # (N, 768)
    mx.eval(emb)
    return _to_normalized_numpy_batch(emb)


def _raw_embed_image(image: Image) -> np.ndarray:
    import mlx.core as mx

    from .preprocessing import preprocess_image

    state = load()
    inputs = preprocess_image(image, text_prefix="", tokenizer=state.tokenizer)
    pixel_values = mx.array(inputs["pixel_values"])
    grid_thw = mx.array(inputs["image_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    attn = mx.array(inputs["attention_mask"])
    emb = state.model.encode_image(pixel_values, grid_thw, input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


def _raw_embed_image_from_preprocessed(inputs: dict) -> np.ndarray:
    """GPU forward pass only — preprocessing already done on another thread."""
    import mlx.core as mx

    state = load()
    pixel_values = mx.array(inputs["pixel_values"])
    grid_thw = mx.array(inputs["image_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    attn = mx.array(inputs["attention_mask"])
    emb = state.model.encode_image(pixel_values, grid_thw, input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


def _raw_embed_video(frames: list) -> np.ndarray:
    import mlx.core as mx

    from .preprocessing import preprocess_video

    state = load()
    inputs = preprocess_video(frames, text_prefix="", tokenizer=state.tokenizer)
    pixel_values = mx.array(inputs["pixel_values_videos"])
    grid_thw = mx.array(inputs["video_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    attn = mx.array(inputs["attention_mask"])
    emb = state.model.encode_video(pixel_values, grid_thw, input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


# === Fused-query implementations =============================================
#
# These mirror the ingest-side ``_raw_embed_image`` / ``_raw_embed_video``
# exactly except for the ``text`` argument we hand to the processor:
# an optional user-text prefix gets stitched in front of the fixed
# placeholder tokens. ``""`` for ``text_prefix`` reproduces the
# document-side numerical result bit-for-bit — that's the invariant the
# "image-as-query" UI path relies on (an empty TextField + dragged image
# should hit identical results to the same image being indexed).


def _raw_embed_query_image(text_prefix: str, image: Image) -> np.ndarray:
    import mlx.core as mx

    from .preprocessing import preprocess_image

    state = load()
    inputs = preprocess_image(
        image, text_prefix=text_prefix, tokenizer=state.tokenizer
    )
    pixel_values = mx.array(inputs["pixel_values"])
    grid_thw = mx.array(inputs["image_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    attn = mx.array(inputs["attention_mask"])
    emb = state.model.encode_image(pixel_values, grid_thw, input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


def _raw_embed_fused_weighted(text: str, image: Image, alpha: float) -> np.ndarray:
    """Two sequential model forwards combined as a weighted L2-normalized sum.

    ``text`` must be pre-stripped; the ``"Query: "`` prefix is added here.
    Both forwards run on the same MLX thread (single executor call), so
    there is no concurrency between them.
    """
    text_vec = _raw_embed_text(f"Query: {text}")
    image_vec = _raw_embed_image(image)
    combined = alpha * text_vec + (1.0 - alpha) * image_vec
    norm = float(np.linalg.norm(combined))
    if norm < 1e-12:
        return combined.astype(np.float32, copy=False)
    return (combined / norm).astype(np.float32, copy=False)


def _raw_embed_query_video(text_prefix: str, frames: list) -> np.ndarray:
    import mlx.core as mx

    from .preprocessing import preprocess_video

    state = load()
    inputs = preprocess_video(
        frames, text_prefix=text_prefix, tokenizer=state.tokenizer
    )
    pixel_values = mx.array(inputs["pixel_values_videos"])
    grid_thw = mx.array(inputs["video_grid_thw"])
    input_ids = mx.array(inputs["input_ids"])
    attn = mx.array(inputs["attention_mask"])
    emb = state.model.encode_video(pixel_values, grid_thw, input_ids, attn)
    mx.eval(emb)
    return _to_normalized_numpy(emb)


# === Internal helpers =======================================================

# --- Download state (shared with API for progress reporting) ---
_download_state: dict = {
    "status": "idle",  # idle | downloading | complete | error
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "error": None,
}


def get_download_state() -> dict:
    """Return current download progress (read by /model/download-progress)."""
    return dict(_download_state)


def _download_with_fallback(snapshot_download_fn) -> Path:
    """Try official HF first; if that fails (network error), retry with mirror."""
    import os

    endpoints = []

    # User-configured endpoint takes priority
    configured = settings.HF_ENDPOINT
    if configured:
        endpoints.append(configured)
    else:
        # Try official first, then mirror
        endpoints.append("")  # empty = default huggingface.co
        endpoints.append("https://hf-mirror.com")

    last_error = None
    for endpoint in endpoints:
        try:
            env_backup = os.environ.get("HF_ENDPOINT")
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint
                log.info("trying download with endpoint: %s", endpoint)
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
                log.info("trying download with official HuggingFace")

            # Set short connection timeout so blocked endpoints fail fast
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")

            _download_state["status"] = "downloading"
            _download_state["error"] = None

            result = Path(snapshot_download_fn(settings.MAIN_MODEL_REPO))

            _download_state["status"] = "complete"
            return result
        except Exception as e:
            last_error = e
            log.warning("download failed with endpoint %r: %s", endpoint, e)
            # Restore env
            if env_backup is not None:
                os.environ["HF_ENDPOINT"] = env_backup
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
            continue

    _download_state["status"] = "error"
    _download_state["error"] = str(last_error)
    raise last_error  # type: ignore[misc]


def _do_load() -> LoadedModel:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    log.info("resolving model weights: %s", settings.MAIN_MODEL_REPO)
    repo_dir = _download_with_fallback(snapshot_download)
    log.info("model repo dir: %s", repo_dir)

    # The HF repo ships its own model.py / config; load by reflection so
    # we don't hardcode class names (spike-0 pattern).
    sys.path.insert(0, str(repo_dir))
    model_module = importlib.import_module("model")
    config_cls = _find_config_class(model_module)
    model_cls = _find_model_class(model_module)
    log.info("classes: %s / %s", config_cls.__name__, model_cls.__name__)

    cfg = config_cls.from_dict(json.loads((repo_dir / "config.json").read_text()))
    model = model_cls(cfg)
    model.load_weights(str(repo_dir / "model.safetensors"))
    mx.eval(model.parameters())

    # Bound the Metal allocator's cache so multi-thousand-image ingest
    # can't pile up unreleased GPU buffers and OOM the machine.
    # Best-effort: older MLX builds and non-Metal platforms simply
    # skip. MLX > 0.20 moved ``set_cache_limit`` from ``mx.metal``
    # to top-level ``mx``; we prefer the new location and silently
    # fall back to the deprecated namespace on older wheels.
    try:
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(_MLX_CACHE_LIMIT_BYTES)
            log.info("mlx cache limit: %d bytes", _MLX_CACHE_LIMIT_BYTES)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(_MLX_CACHE_LIMIT_BYTES)
            log.info("mlx (metal) cache limit: %d bytes", _MLX_CACHE_LIMIT_BYTES)
    except Exception as e:  # noqa: BLE001 — never fail load over a tuning knob
        log.debug("mlx set_cache_limit unavailable: %s", e)

    tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
    # The bundled tokenizer.json declares ``truncation.max_length=512``,
    # but the model itself is happy at 8192 (TextConfig) and a single
    # large image's placeholder expansion can exceed 512 tokens (e.g.
    # 70x70 grid → 1225 image tokens). The reference AutoProcessor
    # builds on transformers' fast tokenizer, which defaults to
    # ``truncation=False`` unless the caller asks for it — we mirror
    # that here so the placeholder expansion never gets cut off.
    tokenizer.no_truncation()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        repo_dir=repo_dir,
        config_cls_name=config_cls.__name__,
        model_cls_name=model_cls.__name__,
    )


def _find_config_class(model_module) -> type:
    """The top-level Config has a ``from_dict`` classmethod; the sub-configs
    (TextConfig / VisionConfig) do not — this is how we
    distinguish them.
    """
    for attr_name in dir(model_module):
        if attr_name.startswith("_") or not attr_name.endswith("Config"):
            continue
        obj = getattr(model_module, attr_name)
        if isinstance(obj, type) and hasattr(obj, "from_dict") and callable(obj.from_dict):
            return obj
    raise RuntimeError("no top-level *Config class with from_dict found in model.py")


def _find_model_class(model_module) -> type:
    """Prefer the JinaOmni*EmbeddingModel naming (covers both nano + small)."""
    for attr_name in dir(model_module):
        if attr_name.startswith("_"):
            continue
        if not (attr_name.startswith("JinaOmni") and attr_name.endswith("EmbeddingModel")):
            continue
        obj = getattr(model_module, attr_name)
        if isinstance(obj, type):
            return obj
    raise RuntimeError("no JinaOmni*EmbeddingModel class found in model.py")


def _to_normalized_numpy(emb_mx) -> np.ndarray:
    """Convert (1, D) MLX array → (D,) L2-normalized float32 numpy array.

    Normalizing here lets LanceDB cosine-distance reduce to a dot product.

    jina v5 omni's MLX weights are bfloat16, which NumPy can't grok via
    the buffer protocol; we cast to float32 inside MLX first.
    """
    import mlx.core as mx

    arr = np.array(emb_mx[0].astype(mx.float32))
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return arr.astype(np.float32, copy=False)
    return (arr / norm).astype(np.float32, copy=False)


def _to_normalized_numpy_batch(emb_mx) -> np.ndarray:
    """Convert (N, D) MLX array → (N, D) L2-normalized float32 numpy array."""
    import mlx.core as mx

    arr = np.array(emb_mx.astype(mx.float32))  # (N, D)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32, copy=False)
