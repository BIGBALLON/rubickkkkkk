"""Tests for the small loader helpers added alongside the ingest
memory-cap work.

We don't load the real MLX model here — that's the slow suite's
job. We just exercise:

- ``clear_inference_cache`` is callable, idempotent, and degrades
  cleanly when MLX's Metal helpers aren't exposed.
- The internal cache-limit constant is a sane positive integer (a
  literal-typo regression caught at import time is cheaper than a
  CI run discovering a 200 GB cap on a 16 GB machine).
"""

from __future__ import annotations


def test_clear_inference_cache_is_safely_callable() -> None:
    """Importing + calling ``clear_inference_cache`` must never raise,
    even on builds where ``mlx.core.clear_cache`` / ``mlx.metal.*``
    aren't available. Tests run on machines without GPU all the time
    (CI containers); a noisy ImportError or AttributeError there
    would block the whole suite.
    """
    from rubick_backend.embed.loader import clear_inference_cache

    # Two consecutive calls — idempotency check.
    clear_inference_cache()
    clear_inference_cache()


def test_mlx_cache_limit_constant_is_sane() -> None:
    """The 2 GiB default should not regress to zero or to absurd values
    that would defeat the bound (e.g. accidentally measured in MB).
    """
    from rubick_backend.embed.loader import _MLX_CACHE_LIMIT_BYTES

    # > 256 MB so we don't trip ourselves with a unit confusion;
    # < 16 GiB so we never pretend an Air can spare more than half
    # its RAM for one process's GPU cache.
    assert 256 * 1024 * 1024 < _MLX_CACHE_LIMIT_BYTES < 16 * 1024 * 1024 * 1024
