"""Embedding subsystem — jina v5 omni nano singleton + per-modality wrappers.

The wrappers (`embed_query`, `embed_document`, `embed_image`,
`embed_video`) all funnel through a single MLX model behind a
priority-aware executor (``ModelExecutor``). The executor ensures
that foreground search (`embed_query`, HIGH priority) jumps ahead
of background indexing (`embed_document` / image / video, LOW
priority) in the work queue — see ARCHITECTURE.md (Process model) and
``embed/executor.py``.
"""

from .executor import ModelExecutor, Priority, get_executor, reset_executor_for_tests
from .loader import (
    clear_inference_cache,
    embed_document,
    embed_documents_batch,
    embed_image,
    embed_image_preprocessed,
    embed_query,
    embed_query_fused,
    embed_query_fused_weighted,
    embed_text,
    embed_video,
    is_loaded,
    load,
)

__all__ = [
    "ModelExecutor",
    "Priority",
    "clear_inference_cache",
    "embed_document",
    "embed_documents_batch",
    "embed_image",
    "embed_image_preprocessed",
    "embed_query",
    "embed_query_fused",
    "embed_query_fused_weighted",
    "embed_text",
    "embed_video",
    "get_executor",
    "is_loaded",
    "load",
    "reset_executor_for_tests",
]
