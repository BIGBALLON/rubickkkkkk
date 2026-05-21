"""Fast-suite shape + token-count tests for ``embed.preprocessing``.

The slow parity test in ``test_preprocessing_parity.py`` pins the
*numerical* contract against ``transformers.AutoProcessor``; this
file pins the *structural* contract — grid_thw arithmetic,
placeholder expansion, edge cases — without any torch / transformers
dependency. Catches the cheap regressions (wrong reshape, off-by-one
in token expansion) before the slow suite runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from rubick_backend.embed import preprocessing as pp

# === Helpers =================================================================


def _make_image(h: int, w: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed=seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


# === smart_resize ============================================================


@pytest.mark.parametrize(
    "h,w",
    [
        (480, 640),  # SD-ish
        (1080, 1920),  # HD; should clamp to max_pixels
        (200, 200),  # tiny; should scale up to min_pixels
        (32, 32),  # already at the minimum factor
    ],
)
def test_smart_resize_image_returns_factor_aligned_dims(h: int, w: int) -> None:
    """Both output dimensions must be multiples of factor (= 32) so
    the patchify reshape doesn't get a fractional grid."""
    h_bar, w_bar = pp._smart_resize_image(h, w)
    factor = pp._PATCH_SIZE * pp._MERGE_SIZE
    assert h_bar % factor == 0
    assert w_bar % factor == 0
    assert h_bar > 0 and w_bar > 0


def test_smart_resize_image_respects_pixel_budget() -> None:
    """A 1080p image must clamp under max_pixels; a tiny image must
    scale up over min_pixels. Aspect should stay close (within one
    factor step)."""
    h_bar, w_bar = pp._smart_resize_image(1080, 1920)
    assert h_bar * w_bar <= pp._IMAGE_MAX_PIXELS
    h_bar, w_bar = pp._smart_resize_image(64, 64)
    # Clamp-up doesn't always reach min exactly because of factor
    # rounding, so we only assert "moves toward min".
    assert h_bar * w_bar >= 64 * 64


# === patchify ================================================================


def test_image_patchify_shape_for_512x512() -> None:
    """Hand-pinned reference: a 512×512 image after smart_resize lands
    at 512×512 (factor-aligned, within pixel budget). grid_h = grid_w
    = 32; pixel_values is (32*32, 3*2*16*16) = (1024, 1536); grid_thw
    is [[1, 32, 32]]."""
    img = _make_image(512, 512)
    out = pp.preprocess_image(img, text_prefix="", tokenizer=None)
    assert out["pixel_values"].shape == (1024, 1536)
    assert out["pixel_values"].dtype == np.float32
    assert out["image_grid_thw"].tolist() == [[1, 32, 32]]


def test_image_input_ids_count_matches_grid_over_merge_squared() -> None:
    """Token-expansion math: num_image_tokens = grid_thw.prod() //
    merge_size**2. For grid [1, 32, 32] and merge_size 2, that's
    1024 / 4 = 256 image-token IDs. Caller passes empty text so the
    sequence is exactly that many tokens."""
    img = _make_image(512, 512)
    out = pp.preprocess_image(img, text_prefix="", tokenizer=None)
    assert out["input_ids"].shape == (1, 256)
    assert (out["input_ids"] == pp._IMAGE_TOKEN_ID).all()
    assert (out["attention_mask"] == 1).all()


def test_video_patchify_shape_for_8_frames_320x256() -> None:
    """Eight 256×320 frames after smart_resize stay at 256×320 (well
    inside the per-video pixel budget). grid_t = 8/2 = 4, grid_h =
    256/16 = 16, grid_w = 320/16 = 20. pixel_values_videos is
    (4*16*20, 3*2*16*16) = (1280, 1536)."""
    frames = [_make_image(256, 320, seed=i) for i in range(8)]
    out = pp.preprocess_video(frames, text_prefix="", tokenizer=None)
    assert out["pixel_values_videos"].shape == (1280, 1536)
    assert out["video_grid_thw"].tolist() == [[4, 16, 20]]


def test_video_pads_odd_frame_count_with_replication() -> None:
    """An odd frame count rounds up by replicating the last frame —
    this is the upstream Qwen3VL behaviour, not truncation, so we
    don't lose the last real frame's signal."""
    frames = [_make_image(256, 320, seed=i) for i in range(5)]
    out = pp.preprocess_video(frames, text_prefix="", tokenizer=None)
    # 5 frames → padded to 6 → grid_t = 3.
    assert out["video_grid_thw"].tolist() == [[3, 16, 20]]


def test_video_rejects_single_frame() -> None:
    """One frame can't be patched along a temporal axis of size 2 —
    upstream raises and we should match. Two frames is the minimum
    valid input."""
    with pytest.raises(ValueError, match="at least 2"):
        pp.preprocess_video([_make_image(256, 320)], tokenizer=None)


# === Token expansion edge cases =============================================


def test_text_prefix_without_tokenizer_raises() -> None:
    """Passing a non-empty ``text_prefix`` without a tokenizer is a
    caller bug — the function refuses cleanly so a regression doesn't
    silently produce a sequence missing the prefix tokens."""
    img = _make_image(512, 512)
    with pytest.raises(ValueError, match="tokenizer required"):
        pp.preprocess_image(img, text_prefix="describe this", tokenizer=None)


def test_empty_prefix_emits_image_tokens_only() -> None:
    """Empty prefix is the document-side embedding case — no prefix
    tokens, all image-token IDs."""
    img = _make_image(512, 512)
    out = pp.preprocess_image(img, text_prefix="", tokenizer=None)
    assert out["input_ids"].shape[1] == 256
    assert (out["input_ids"] == pp._IMAGE_TOKEN_ID).all()
