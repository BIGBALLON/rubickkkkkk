"""Nebula map state tracking — progress, staleness, debounce.

Module-level mutable state (same pattern as settings.py runtime knobs).
Thread-safe via a lock since the compute runs in asyncio.to_thread.
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum

_lock = threading.Lock()


class ComputeState(StrEnum):
    idle = "idle"
    computing = "computing"


_state: ComputeState = ComputeState.idle
_progress: float = 0.0
_last_computed_at: int = 0
_total_points: int = 0


def get_status() -> dict:
    """Snapshot for GET /nebula/status."""
    with _lock:
        return {
            "state": _state.value,
            "progress": _progress,
            "last_computed_at": _last_computed_at,
            "total_points": _total_points,
        }


def set_computing(progress: float = 0.0) -> None:
    global _state, _progress
    with _lock:
        _state = ComputeState.computing
        _progress = progress


def set_progress(progress: float) -> None:
    global _progress
    with _lock:
        _progress = progress


def set_idle(total_points: int) -> None:
    global _state, _progress, _last_computed_at, _total_points
    with _lock:
        _state = ComputeState.idle
        _progress = 1.0
        _last_computed_at = int(time.time())
        _total_points = total_points


def is_computing() -> bool:
    with _lock:
        return _state == ComputeState.computing


def last_computed_at() -> int:
    with _lock:
        return _last_computed_at
