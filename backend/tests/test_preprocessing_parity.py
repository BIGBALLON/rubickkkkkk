"""Parity tests for the hand-rolled image / video preprocessing.

Compares ``embed.preprocessing.preprocess_image`` /
``preprocess_video`` against the canonical
``transformers.AutoProcessor`` output that the model was trained
against. Slow because every test loads ``transformers`` + the
custom processor's ``trust_remote_code=True`` chain.

Two parity bars:

- **Shape parity** — exact match. Different shapes mean a wrong
  reshape order, which would silently corrupt every embedding.
  Asserted strictly.
- **Embedding cosine** — ``encode_image`` / ``encode_video`` outputs
  must agree with cos ≥ 0.9990. Pixel-level outputs are NOT bit
  identical because PIL bicubic and torchvision bicubic-with-
  antialias diverge by O(10⁻²) per channel; the model is robust
  enough to absorb that into a near-1.0 cosine. The 0.9990 floor
  matches the model's documented "Multimodal parity vs torch"
  (README) bar of cos ≥ 0.999.

Run with::

    RUBICK_RUN_SLOW=1 pytest tests/test_preprocessing_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


# === Fixtures ===============================================================


@pytest.fixture(scope="module")
def autoprocessor():
    """Load the canonical ``AutoProcessor`` for the production model.

    Skipped if transformers isn't installed (the production deps tree
    no longer pulls it; only the dev extras do).
    """
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor
    except ImportError as e:
        pytest.skip(f"transformers not installed: {e}")

    from rubick_backend import settings as _settings

    repo_dir = Path(snapshot_download(_settings.MAIN_MODEL_REPO))
    return AutoProcessor.from_pretrained(str(repo_dir), trust_remote_code=True)


@pytest.fixture(scope="module")
def tokenizer():
    """The same Rust tokenizer the loader caches in production —
    needed so the parity test exercises the placeholder-expansion +
    text-encoding path under the same bytes the runtime uses.
    """
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    from rubick_backend import settings as _settings

    repo_dir = Path(snapshot_download(_settings.MAIN_MODEL_REPO))
    return Tokenizer.from_file(str(repo_dir / "tokenizer.json"))


@pytest.fixture()
def synthetic_image():
    """A non-trivial RGB gradient PNG that exercises the resize path
    (640 × 480 isn't aligned to the 32-pixel grid factor, so
    ``smart_resize`` must round it).
    """
    from PIL import Image

    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(
        arr[:, :, 0].astype(int) + np.linspace(0, 255, 640).astype(int),
        0, 255,
    ).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


@pytest.fixture()
def synthetic_frames():
    """Eight 320 × 256 frames — a typical 32-frame sample is overkill
    for parity (the patchify reshape only depends on the per-frame
    spatial layout once the temporal dim is fixed); 8 keeps the test
    fast while still exercising grid_t ≥ 4."""
    from PIL import Image

    frames = []
    rng = np.random.default_rng(seed=7)
    for i in range(8):
        arr = rng.integers(0, 256, size=(256, 320, 3), dtype=np.uint8)
        arr[:, :, 1] = (i * 32) % 256  # green ramp across frames
        frames.append(Image.fromarray(arr).convert("RGB"))
    return frames


# === Helpers ================================================================


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine between two 1-D vectors. Returns 0 on a degenerate
    zero vector (won't happen for embeddings but defensive)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# === Image parity ===========================================================


def test_image_shape_parity(autoprocessor, tokenizer, synthetic_image) -> None:
    """``pixel_values`` and ``image_grid_thw`` must have identical
    shapes between our numpy path and the AutoProcessor; ``input_ids``
    must have the same length (placeholder expansion math agrees)."""
    from rubick_backend.embed.preprocessing import preprocess_image

    ours = preprocess_image(synthetic_image, text_prefix="", tokenizer=tokenizer)
    ref = autoprocessor(
        images=[synthetic_image],
        text="<|vision_start|><|image_pad|><|vision_end|>",
        return_tensors="pt",
    )

    ref_pv = ref["pixel_values"].numpy()
    ref_grid = ref["image_grid_thw"].numpy()
    ref_ids = ref["input_ids"].numpy()

    assert ours["pixel_values"].shape == ref_pv.shape, (
        ours["pixel_values"].shape, ref_pv.shape,
    )
    assert ours["image_grid_thw"].shape == ref_grid.shape
    assert (ours["image_grid_thw"] == ref_grid).all()
    assert ours["input_ids"].shape[1] == ref_ids.shape[1]


def test_image_pixel_values_close(autoprocessor, tokenizer, synthetic_image) -> None:
    """Pixel-level outputs aren't bit-identical (PIL bicubic vs
    torchvision bicubic+antialias) but must stay close enough that
    the per-channel max-abs deviation is bounded — a guardrail
    against a regression in the patchify permute order, which would
    show up as a much larger error than the resize drift."""
    from rubick_backend.embed.preprocessing import preprocess_image

    ours = preprocess_image(synthetic_image, text_prefix="", tokenizer=tokenizer)
    ref = autoprocessor(
        images=[synthetic_image],
        text="<|vision_start|><|image_pad|><|vision_end|>",
        return_tensors="pt",
    )
    diff = np.abs(ours["pixel_values"] - ref["pixel_values"].numpy())
    # Empirically ~0.04 max-abs across PIL vs tvF bicubic; a regression
    # in reshape permutation would push this over 1.0.
    assert diff.max() < 0.2, (diff.max(), diff.mean())
    assert diff.mean() < 0.05


def test_image_encode_cosine_above_floor(
    autoprocessor, tokenizer, synthetic_image
) -> None:
    """End-to-end: feeding our pixel_values into the same MLX model
    as the AutoProcessor output must produce embeddings with
    cos ≥ 0.999 (matches the model's documented torch-vs-MLX bar)."""
    import mlx.core as mx

    from rubick_backend.embed.loader import _to_normalized_numpy, load
    from rubick_backend.embed.preprocessing import preprocess_image

    state = load()

    ours = preprocess_image(synthetic_image, text_prefix="", tokenizer=tokenizer)
    ours_emb = state.model.encode_image(
        mx.array(ours["pixel_values"]),
        mx.array(ours["image_grid_thw"]),
        mx.array(ours["input_ids"]),
        mx.array(ours["attention_mask"]),
    )
    mx.eval(ours_emb)
    ours_v = _to_normalized_numpy(ours_emb)

    ref = autoprocessor(
        images=[synthetic_image],
        text="<|vision_start|><|image_pad|><|vision_end|>",
        return_tensors="pt",
    )
    ref_emb = state.model.encode_image(
        mx.array(ref["pixel_values"].numpy()),
        mx.array(ref["image_grid_thw"].numpy()),
        mx.array(ref["input_ids"].numpy()),
        mx.array(ref["attention_mask"].numpy()),
    )
    mx.eval(ref_emb)
    ref_v = _to_normalized_numpy(ref_emb)

    cos = _cos(ours_v, ref_v)
    assert cos >= 0.999, f"image embedding cos drifted: {cos:.6f}"


# === Video parity ===========================================================


def test_video_shape_parity(autoprocessor, tokenizer, synthetic_frames) -> None:
    """Same shape contract for video.

    Note on ``do_sample_frames``: the legacy AutoProcessor flow
    silently re-sampled the caller's frame list down to ~min_frames=4
    via the default ``fps=2`` + assumed ``metadata.fps=24`` heuristic
    — which meant production was embedding 4 frames per video, not
    the 32 the ingest pipeline extracted (a quiet quality bug). The
    new numpy path always embeds every frame the caller passes; the
    parity test disables sampling on the reference side so we
    compare like-with-like."""
    from rubick_backend.embed.preprocessing import preprocess_video

    ours = preprocess_video(synthetic_frames, text_prefix="", tokenizer=tokenizer)
    ref = autoprocessor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=synthetic_frames,
        do_sample_frames=False,
        return_tensors="pt",
    )

    ref_pv = ref["pixel_values_videos"].numpy()
    ref_grid = ref["video_grid_thw"].numpy()

    assert ours["pixel_values_videos"].shape == ref_pv.shape
    assert ours["video_grid_thw"].shape == ref_grid.shape
    assert (ours["video_grid_thw"] == ref_grid).all()


def test_video_encode_cosine_above_floor(
    autoprocessor, tokenizer, synthetic_frames
) -> None:
    """End-to-end ``encode_video`` parity. Identical bar to image.
    See ``test_video_shape_parity`` for why we pass
    ``do_sample_frames=False`` on the reference side."""
    import mlx.core as mx

    from rubick_backend.embed.loader import _to_normalized_numpy, load
    from rubick_backend.embed.preprocessing import preprocess_video

    state = load()

    ours = preprocess_video(synthetic_frames, text_prefix="", tokenizer=tokenizer)
    ours_emb = state.model.encode_video(
        mx.array(ours["pixel_values_videos"]),
        mx.array(ours["video_grid_thw"]),
        mx.array(ours["input_ids"]),
        mx.array(ours["attention_mask"]),
    )
    mx.eval(ours_emb)
    ours_v = _to_normalized_numpy(ours_emb)

    ref = autoprocessor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=synthetic_frames,
        do_sample_frames=False,
        return_tensors="pt",
    )
    ref_emb = state.model.encode_video(
        mx.array(ref["pixel_values_videos"].numpy()),
        mx.array(ref["video_grid_thw"].numpy()),
        mx.array(ref["input_ids"].numpy()),
        mx.array(ref["attention_mask"].numpy()),
    )
    mx.eval(ref_emb)
    ref_v = _to_normalized_numpy(ref_emb)

    cos = _cos(ours_v, ref_v)
    assert cos >= 0.999, f"video embedding cos drifted: {cos:.6f}"


# === Cheap shape-only sanity (fast suite) ===================================
#
# These run unconditionally (no slow marker, no transformers
# requirement). They pin the math that turns image / video dimensions
# into ``grid_thw`` so a refactor of ``smart_resize`` or the patchify
# reshape can't silently shift the contract.


@pytest.mark.skip(reason="moved to fast suite via test_preprocessing_shapes.py")
def test_image_shape_math_pinned() -> None:
    pass  # see test_preprocessing_shapes.py
