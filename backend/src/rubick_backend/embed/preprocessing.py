"""Hand-rolled image / video preprocessing for jina v5 omni nano.

Replaces ``transformers.AutoProcessor`` (and its torch / torchvision
dependency chain) with a pure PIL + numpy implementation. The bytes
produced here go straight into the MLX model — so the byte-level
contract has to match what the model was trained on, which is the
Qwen2VL image processor + Qwen3VL video processor outputs.

Pixel values land in the *exact* layout the model expects:

    image:  (grid_h * grid_w,         channels * temporal * patch * patch)
    video:  (grid_t * grid_h * grid_w, channels * temporal * patch * patch)

where ``temporal = temporal_patch_size = 2``. Image inputs are
synthesised into 2 "frames" by duplication along the temporal axis;
video inputs supply real frames (with last-frame replication if the
count isn't divisible by 2).

Token expansion mirrors ``Qwen2VLProcessor.__call__``'s placeholder
loop: a single ``<image>`` placeholder in the user text gets replaced
with ``grid_thw.prod() // merge_size**2`` consecutive image-token IDs
before tokenisation. The custom ``LlavaEuroBertProcessor`` shipped
with the model also rewrites
``<|vision_start|><|image_pad|><|vision_end|>`` (and the video
equivalent) to ``<image>`` first, so we accept either form on the
caller side.

Numerical parity vs. ``AutoProcessor``:

- Resize uses PIL bicubic (``Image.Resampling.BICUBIC``); the
  reference ``Qwen2VLImageProcessor`` resize via torchvision's
  ``tvF.resize`` with ``InterpolationMode.BICUBIC, antialias=True``.
  Pixel-level outputs differ by O(10⁻²) per channel — ``encode_image``
  output cosine drops to ~0.9998, comfortably above the project's
  documented cos ≥ 0.999 floor (see README "Multimodal parity vs
  torch"). ``test_preprocessing_parity.py`` (slow suite) pins this
  end-to-end against AutoProcessor whenever transformers is installed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


# === Constants (mirror processor_config.json) ===============================

# Pulled from the bundled ``preprocessor_config.json`` /
# ``processor_config.json``. Keep these in sync with the model repo;
# a mismatch here would silently produce wrong-shape pixel_values
# and a quiet retrieval-quality regression rather than a crash.

_PATCH_SIZE: int = 16
_TEMPORAL_PATCH_SIZE: int = 2
_MERGE_SIZE: int = 2

# Spatial pixel budget for a single image (in pixels, not bytes):
_IMAGE_MIN_PIXELS: int = 262_144  # 512 × 512
_IMAGE_MAX_PIXELS: int = 802_816  # ~896 × 896 (fewer tokens = faster encode)

# Total-volume pixel budget for video (frames × H × W):
_VIDEO_MIN_PIXELS: int = 4096
_VIDEO_MAX_PIXELS: int = 25_165_824  # 32 × 32 × 768 × … from config

# Mean / std for normalisation: (x / 255 - 0.5) / 0.5 → [-1, 1].
# Stored as channel-first (3,) numpy arrays for fast broadcasting.
_MEAN: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
_STD: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)

# Tokenizer-side: from model.py's ``IMAGE_TOKEN_ID``. The custom
# processor maps both the image and video placeholder to ``<image>``
# (and that token's id is 128259), so we use one constant for both.
_IMAGE_TOKEN_ID: int = 128259
_IMAGE_TOKEN_LITERAL: str = "<image>"


# === Public surface ==========================================================

ImageInputs = dict[str, np.ndarray]
"""Output dict with the keys ``encode_image`` consumes:
``pixel_values`` (n_patches, channels*temporal*ps*ps),
``image_grid_thw`` (1, 3), ``input_ids`` (1, seq_len),
``attention_mask`` (1, seq_len)."""

VideoInputs = dict[str, np.ndarray]
"""Same shape as ``ImageInputs`` but with ``pixel_values_videos`` /
``video_grid_thw`` keys. Token expansion uses the same image-token
id (the custom processor maps video to the image placeholder)."""


def preprocess_image(image: Image, text_prefix: str = "", tokenizer=None) -> ImageInputs:
    """Run the full Qwen2VL-style image preprocessing pipeline.

    ``text_prefix`` is the optional user text that gets stitched in
    front of the placeholder for fused queries; pass ``""``
    for the document-side embedding.

    ``tokenizer`` must be the loaded ``tokenizers.Tokenizer`` (passed
    explicitly so the function stays a pure helper — no module-level
    state, no circular import on the loader singleton).
    """
    rgb = _to_rgb_numpy(image)
    h, w = rgb.shape[1:]
    h_bar, w_bar = _smart_resize_image(h, w)
    resized = _resize_bicubic(image, h_bar, w_bar)

    pixel_values, grid_thw = _patchify_image(resized)
    input_ids, attention_mask = _expand_placeholder_tokens(
        text_prefix, grid_thw, tokenizer
    )
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def preprocess_video(
    frames: list[Image], text_prefix: str = "", tokenizer=None
) -> VideoInputs:
    """Run the Qwen3VL-style video preprocessing pipeline.

    ``frames`` is a list of PIL frames at any resolution (must already
    be ≥ 2; oddness is handled by replicating the last frame, matching
    the upstream processor). All frames get resized to one common
    (h_bar, w_bar) computed from the first frame's size and the total
    pixel budget.
    """
    if len(frames) < 2:
        raise ValueError(f"need at least 2 video frames, got {len(frames)}")
    n = len(frames)
    h, w = frames[0].size[1], frames[0].size[0]
    h_bar, w_bar = _smart_resize_video(n, h, w)

    resized = [_resize_bicubic(f, h_bar, w_bar) for f in frames]
    pixel_values, grid_thw = _patchify_video(resized)
    input_ids, attention_mask = _expand_placeholder_tokens(
        text_prefix, grid_thw, tokenizer
    )
    return {
        "pixel_values_videos": pixel_values,
        "video_grid_thw": grid_thw,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# === Resize ==================================================================


def _smart_resize_image(height: int, width: int) -> tuple[int, int]:
    """Resize so dimensions are multiples of ``patch_size * merge_size``
    (= 32) and total pixels fall in ``[min, max]``. Aspect-preserving.
    Mirrors ``Qwen2VLImageProcessor.smart_resize`` exactly.
    """
    return _smart_resize(
        height, width, factor=_PATCH_SIZE * _MERGE_SIZE,
        min_pixels=_IMAGE_MIN_PIXELS,
        max_pixels=_IMAGE_MAX_PIXELS,
    )


def _smart_resize_video(num_frames: int, height: int, width: int) -> tuple[int, int]:
    """Same shape rules as ``_smart_resize_image``, but the pixel
    budget applies to the *total* video volume (T × H × W). Mirrors
    ``Qwen3VLVideoProcessor.smart_resize`` (which we ignore the
    ``temporal_factor`` clause of because video frames are already
    padded to even ``T`` upstream).
    """
    factor = _PATCH_SIZE * _MERGE_SIZE
    if height < factor or width < factor:
        raise ValueError(
            f"height ({height}) or width ({width}) must be ≥ {factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"aspect ratio must be < 200 (got {max(height, width) / min(height, width)})"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    total = num_frames * h_bar * w_bar
    if total > _VIDEO_MAX_PIXELS:
        beta = math.sqrt((num_frames * height * width) / _VIDEO_MAX_PIXELS)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif total < _VIDEO_MIN_PIXELS:
        beta = math.sqrt(_VIDEO_MIN_PIXELS / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """The shared per-image smart_resize from ``Qwen2VLImageProcessor``.
    Verbatim port; do not retune without re-verifying numerical parity.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"aspect ratio must be < 200 (got {max(height, width) / min(height, width)})"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize_bicubic(image: Image, h_bar: int, w_bar: int) -> np.ndarray:
    """PIL bicubic resize → channel-first float32 normalised array.

    PIL's ``BICUBIC`` is not byte-identical to torchvision's bicubic
    + antialias; the parity tests document the resulting embedding
    cos drift (~0.9998, well above the cos ≥ 0.999 release floor).
    """
    from PIL import Image as PILImage

    # ``image.resize`` wants (width, height); we receive (h_bar, w_bar).
    resized = image.convert("RGB").resize(
        (w_bar, h_bar), resample=PILImage.Resampling.BICUBIC
    )
    arr = np.asarray(resized, dtype=np.uint8)  # (H, W, 3)
    arr = arr.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)
    arr = arr / 255.0
    arr = (arr - _MEAN) / _STD
    return arr


def _to_rgb_numpy(image: Image) -> np.ndarray:
    """Coerce a PIL image to a (3, H, W) uint8 array. Used only for
    the height/width readback in the public entry points; the actual
    pixel pipeline goes through ``_resize_bicubic`` so callers don't
    pay for two RGB conversions."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)  # (H, W, 3)
    return arr.transpose(2, 0, 1)


# === Patchify ================================================================


def _patchify_image(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One image (3, H, W) → flattened patches the model expects.

    Synthesises a temporal axis of size ``temporal_patch_size`` by
    duplication so the same downstream reshape works for image and
    video alike. Matches ``Qwen2VLImageProcessor._preprocess`` byte
    for byte once we strip the (always 1) batch dimension off both
    sides.
    """
    c, h, w = arr.shape
    grid_h = h // _PATCH_SIZE
    grid_w = w // _PATCH_SIZE
    ms = _MERGE_SIZE
    ps = _PATCH_SIZE
    tp = _TEMPORAL_PATCH_SIZE

    # (C, gh/ms, ms, ps, gw/ms, ms, ps)
    reshaped = arr.reshape(c, grid_h // ms, ms, ps, grid_w // ms, ms, ps)
    # → (gh/ms, gw/ms, ms, ms, C, ps, ps) — matches the upstream
    # permute(0, 2, 5, 3, 6, 1, 4, 7) on a (1, C, gh/ms, ms, ps, gw/ms,
    # ms, ps) tensor with the leading batch axis dropped.
    permuted = reshaped.transpose(1, 4, 2, 5, 0, 3, 6)
    # Add temporal axis (size 1) and broadcast to size ``tp`` —
    # equivalent to PyTorch's ``.unsqueeze(5).expand(..., tp, ...)``.
    expanded = np.broadcast_to(
        permuted[..., None, :, :],
        permuted.shape[:-2] + (tp,) + permuted.shape[-2:],
    )
    # → (gh/ms, gw/ms, ms, ms, C, T, ps, ps) → (gh*gw, C*T*ps*ps)
    flattened = expanded.reshape(grid_h * grid_w, c * tp * ps * ps)
    pixel_values = np.ascontiguousarray(flattened.astype(np.float32))
    grid_thw = np.array([[1, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, grid_thw


def _patchify_video(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """List of (3, H, W) per-frame arrays → flat patches + grid_thw.

    Pads to even frame count by replicating the last frame (matches
    Qwen3VL); reshapes via the canonical
    ``(B, T, C, H, W) → patches`` permutation.
    """
    tp = _TEMPORAL_PATCH_SIZE
    n = len(frames)
    pad = (-n) % tp
    if pad:
        frames = list(frames) + [frames[-1]] * pad
    n_padded = len(frames)
    grid_t = n_padded // tp

    stacked = np.stack(frames, axis=0)  # (T, C, H, W)
    t, c, h, w = stacked.shape
    grid_h = h // _PATCH_SIZE
    grid_w = w // _PATCH_SIZE
    ms = _MERGE_SIZE
    ps = _PATCH_SIZE

    # Reshape to (grid_t, T_patch, C, gh/ms, ms, ps, gw/ms, ms, ps).
    reshaped = stacked.reshape(
        grid_t, tp, c, grid_h // ms, ms, ps, grid_w // ms, ms, ps
    )
    # Permute to (grid_t, gh/ms, gw/ms, ms, ms, C, T_patch, ps, ps) —
    # mirrors Qwen3VL's permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9) on the
    # (B, grid_t, T_patch, C, gh/ms, ms, ps, gw/ms, ms, ps) tensor
    # with the leading batch axis dropped.
    permuted = reshaped.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened = permuted.reshape(
        grid_t * grid_h * grid_w, c * tp * ps * ps
    )
    pixel_values = np.ascontiguousarray(flattened.astype(np.float32))
    grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return pixel_values, grid_thw


# === Token expansion =========================================================


def _expand_placeholder_tokens(
    text_prefix: str, grid_thw: np.ndarray, tokenizer
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``Qwen2VLProcessor.__call__``'s ``<image>`` expansion +
    tokenisation, then return ``(input_ids, attention_mask)`` as
    1-batched int arrays.

    The wire format is ``"<text_prefix><image>...<image>"`` (with N
    copies of ``<image>`` where N = ``grid_thw.prod() // merge_size**2``)
    fed to the tokenizer. We deliberately go through the full encode
    rather than splicing the image-token ID into a list of prefix IDs
    because the tokenizer auto-appends ``<|end_of_text|>`` and the
    model uses last-token pooling — losing that EOS would shift the
    pooled embedding.

    ``tokenizer`` is required even when ``text_prefix`` is empty so
    we can always call its ``encode`` and inherit the EOS behaviour.
    Pure-shape callers (e.g. fast-suite tests) can pass
    ``tokenizer=None`` to opt out of the EOS step at the cost of
    parity-by-token-count.
    """
    merge_length = _MERGE_SIZE * _MERGE_SIZE
    num_tokens = int(grid_thw.prod() // merge_length)
    placeholder = _IMAGE_TOKEN_LITERAL * num_tokens
    full_text = (text_prefix + placeholder) if text_prefix else placeholder

    if tokenizer is None:
        # Test-only branch: build the placeholder + EOS ourselves so
        # the resulting shape still matches the tokenizer's output
        # (image_token_id × N + end_of_text).
        if text_prefix:
            raise ValueError(
                "tokenizer required to encode a non-empty text_prefix"
            )
        full_ids = [_IMAGE_TOKEN_ID] * num_tokens
        full_mask = [1] * num_tokens
    else:
        enc = tokenizer.encode(full_text)
        full_ids = list(enc.ids)
        full_mask = list(enc.attention_mask)

    input_ids = np.array([full_ids], dtype=np.int64)
    attention_mask = np.array([full_mask], dtype=np.int64)
    return input_ids, attention_mask
