"""UMAP 3-D projection of image/video embeddings.

Pipeline:
1. Read all image+video embeddings from LanceDB.
2. If < 10 points, place randomly and persist.
3. Run UMAP fit_transform (cosine metric).
4. Normalize output to [0, 1].
5. Persist atomically to NEBULA_MAP_FILE.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from .. import settings
from ..store.schema import open_table
from . import state

log = logging.getLogger(__name__)


def run_nebula_compute() -> dict:
    """Blocking UMAP computation. Run via asyncio.to_thread.

    Returns the map dict (same shape as the JSON file) so the caller
    can serve it immediately without a disk round-trip.
    """
    state.set_computing(progress=0.1)

    # Step 1: Read embeddings
    table = open_table()
    try:
        df = (
            table.search()
            .where("modality IN ('image', 'video')")
            .select(["id", "doc_id", "embedding", "modality", "thumbnail_path", "filename"])
            .limit(2**31 - 1)
            .to_pandas()
        )
    except Exception as e:
        log.error("nebula compute: failed to read embeddings: %s", e)
        state.set_idle(total_points=0)
        raise

    if df.empty:
        result = _empty_map()
        _persist(result)
        state.set_idle(total_points=0)
        return result

    state.set_progress(0.3)

    # Step 2: Extract embedding matrix
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    n_points = len(embeddings)

    # Step 3: Compute 3-D projection
    if n_points < 10:
        # Too few for meaningful UMAP — random placement
        rng = np.random.default_rng(42)
        coords = rng.random((n_points, 3)).astype(np.float32)
    else:
        state.set_progress(0.5)
        import umap

        reducer = umap.UMAP(
            n_components=3,
            metric="cosine",
            n_neighbors=min(15, n_points - 1),
            min_dist=0.1,
            random_state=42,
        )
        coords = reducer.fit_transform(embeddings)

    state.set_progress(0.8)

    # Step 4: Normalize to [0, 1]
    coords = _normalize(coords)

    # Step 4.5: Cluster assignment (HDBSCAN — auto-discovers cluster count)
    cluster_labels = _compute_clusters(coords, n_points)

    # Step 5: Build result
    points = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        points.append({
            "doc_id": row["doc_id"],
            "chunk_id": row["id"],
            "x": float(coords[idx, 0]),
            "y": float(coords[idx, 1]),
            "z": float(coords[idx, 2]),
            "cluster": int(cluster_labels[idx]),
            "modality": row["modality"],
            "thumbnail_path": row.get("thumbnail_path"),
            "filename": row.get("filename", ""),
        })

    result = {
        "version": 1,
        "computed_at": int(time.time()),
        "total_points": n_points,
        "points": points,
    }

    _persist(result)
    state.set_idle(total_points=n_points)
    log.info("nebula compute complete: %d points", n_points)
    return result


def load_map() -> dict:
    """Read the persisted map from disk, or return empty map."""
    path = settings.NEBULA_MAP_FILE
    if not path.is_file():
        return _empty_map()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and "points" in data:
            return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("nebula map read failed: %s", e)
    return _empty_map()


def is_stale() -> bool:
    """True if index has image/video chunks added since last compute."""
    last = state.last_computed_at()
    if last == 0:
        # Never computed — stale if any image/video exists
        table = open_table()
        try:
            count = table.count_rows(filter="modality IN ('image', 'video')")
            return count > 0
        except Exception:
            return False

    # Check if any chunk was created after last compute
    table = open_table()
    try:
        count = table.count_rows(
            filter=f"modality IN ('image', 'video') AND created_at > {last}"
        )
        return count > 0
    except Exception:
        return False


def _normalize(coords: np.ndarray) -> np.ndarray:
    """Min-max normalize each axis to [0, 1]."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    # Avoid division by zero (all points same position on an axis)
    ranges[ranges == 0] = 1.0
    return (coords - mins) / ranges


def _compute_clusters(coords: np.ndarray, n_points: int) -> np.ndarray:
    """Auto-discover clusters using HDBSCAN (density-based, no fixed K).

    Returns an array of cluster labels (0-based). Noise points get
    assigned to the nearest cluster so every point has a color.
    """
    if n_points < 15:
        return np.zeros(n_points, dtype=int)

    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(10, n_points // 50),
            min_samples=5,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(coords)

        # Assign noise points (-1) to nearest non-noise cluster
        noise_mask = labels == -1
        if noise_mask.any() and not noise_mask.all():
            from scipy.spatial import cKDTree

            valid_mask = ~noise_mask
            tree = cKDTree(coords[valid_mask])
            _, indices = tree.query(coords[noise_mask])
            labels[noise_mask] = labels[valid_mask][indices]

        # Re-number from 0
        unique = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique)}
        labels = np.array([remap[lbl] for lbl in labels])

        log.info("nebula clusters: %d found by HDBSCAN", len(unique))
        return labels

    except ImportError:
        log.warning("hdbscan not installed; falling back to spatial bucketing")
        # Fallback: simple spatial bucketing (same as before)
        k = min(10, max(3, n_points // 100))
        order = np.argsort(coords[:, 0] + coords[:, 1] * 1.5 + coords[:, 2] * 0.8)
        labels = np.zeros(n_points, dtype=int)
        bucket_size = max(1, n_points // k)
        for i, idx in enumerate(order):
            labels[idx] = min(i // bucket_size, k - 1)
        return labels


def _persist(result: dict) -> None:
    """Atomic write to NEBULA_MAP_FILE (tmp + rename)."""
    settings.ensure_data_dirs()
    path = settings.NEBULA_MAP_FILE
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("nebula map write failed: %s", e)


def _empty_map() -> dict:
    return {"version": 1, "computed_at": 0, "total_points": 0, "points": []}
