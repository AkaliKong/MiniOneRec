"""Validation, persistence and catalog-level SID export shared by the builders."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import os
import tempfile

import numpy as np

FORMAT_VERSION = 1


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def vectors(values, expected_dim=None):
    array = np.asarray(values)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("embeddings must have nonempty shape (items, dimensions)")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("embeddings must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("embeddings must be finite and representable as float32")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ValueError(f"expected embedding dimension {expected_dim}, got {array.shape[1]}")
    return array


def normalize_rows(array, eps=1e-10):
    array = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    result = np.zeros_like(array)
    np.divide(array, norms, out=result, where=norms > eps)
    return result.astype(np.float32)


def projection_residual(array, directions, eps=1e-10):
    """Normalize r - proj_c(r); a near-zero c removes no component."""
    r = np.asarray(array, dtype=np.float64)
    c = np.asarray(directions, dtype=np.float64)
    if r.ndim != 2 or c.ndim not in (1, 2) or c.shape[-1] != r.shape[1]:
        raise ValueError("projection directions must match the residual dimension")
    if c.ndim == 2 and c.shape[0] != r.shape[0]:
        raise ValueError("one projection direction is required per residual row")
    dot = np.sum(r * c, axis=1, keepdims=True)
    norm_sq = np.sum(c * c, axis=-1, keepdims=True)
    scale = np.zeros_like(dot)
    np.divide(dot, norm_sq, out=scale, where=norm_sq > eps * eps)
    return normalize_rows(r - scale * c, eps)


def canonical_item_ids(values, count):
    if len(values) != count:
        raise ValueError(f"expected {count} item IDs, got {len(values)}")
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
            raise ValueError("each item ID must be a string or an integer")
        value = str(value)
        if not value or value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("item IDs must be nonempty, trimmed, and contain no control characters")
        ids.append(value)
    if len(set(ids)) != count:
        raise ValueError("item IDs must be unique after conversion to strings")
    return ids


def position_prefix(position):
    return chr(97 + position) if position < 26 else f"l{position}"


def index_statistics(index):
    groups = Counter(tuple(tokens) for tokens in index.values())
    count = len(index)
    lengths = Counter(len(tokens) for tokens in index.values())
    return {
        "items": count,
        "unique_sids": len(groups),
        "excess_collision_count": count - len(groups),
        "excess_collision_rate": (count - len(groups)) / count if count else 0.0,
        "affected_item_count": sum(size for size in groups.values() if size > 1),
        "affected_item_rate": sum(size for size in groups.values() if size > 1) / count if count else 0.0,
        "max_collision_group": max(groups.values(), default=0),
        "length_histogram": {str(k): v for k, v in sorted(lengths.items())},
        "observed_vocabulary_size": len({token for tokens in index.values() for token in tokens}),
    }


def build_indexes(item_ids, codes, capacities, collision_policy="append"):
    """Export a complete catalog. The optional suffix is an identity code, not semantics.

    Suffix ranks are invariant to input row order, but may change when catalog
    membership changes. Keep the exported mapping as part of the dataset version.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or min(codes.shape) == 0 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("codes must be a nonempty integer matrix")
    ids = canonical_item_ids(item_ids, len(codes))
    capacities = [positive_int(k, "code capacity") for k in capacities]
    if len(capacities) != codes.shape[1]:
        raise ValueError("code capacities must match the number of positions")
    for level, capacity in enumerate(capacities):
        if np.any(codes[:, level] < 0) or np.any(codes[:, level] >= capacity):
            raise ValueError(f"codes at position {level} must be in [0, {capacity})")
    if collision_policy not in ("append", "error", "allow"):
        raise ValueError("collision policy must be append, error, or allow")
    raw = {
        item_id: [f"<{position_prefix(p)}_{int(code)}>" for p, code in enumerate(row)]
        for item_id, row in zip(ids, codes)
    }
    raw_stats = index_statistics(raw)
    if collision_policy == "error" and raw_stats["excess_collision_count"]:
        raise ValueError("raw SID collisions remain; use --collision-policy append or an explicit research protocol")
    final = {key: list(tokens) for key, tokens in raw.items()}
    appended = collision_policy == "append" and raw_stats["excess_collision_count"] > 0
    if appended:
        groups = defaultdict(list)
        for item_id, tokens in raw.items():
            groups[tuple(tokens)].append(item_id)
        prefix = position_prefix(codes.shape[1])
        for members in groups.values():
            for rank, item_id in enumerate(sorted(members)):
                final[item_id].append(f"<{prefix}_{rank}>")
    position_stats = []
    for level, capacity in enumerate(capacities):
        counts = np.bincount(codes[:, level], minlength=capacity)
        probabilities = counts[counts > 0] / len(codes)
        position_stats.append({
            "position": level,
            "capacity": capacity,
            "used_codes": int(np.count_nonzero(counts)),
            "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities))),
        })
    stats = {
        "collision_policy": collision_policy,
        "semantic_token_count": int(codes.shape[1]),
        "identity_suffix_appended": appended,
        "identity_suffix_rule": "lexicographic item-ID rank within each raw SID group; complete catalog snapshot",
        "semantic_positions": position_stats,
        "raw": raw_stats,
        "final": index_statistics(final),
    }
    return raw, final, stats


def save_npz(path, metadata, arrays, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"model already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, format_version=FORMAT_VERSION)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_npz(path, algorithm):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format_version") != FORMAT_VERSION or metadata.get("algorithm") != algorithm:
            raise ValueError(f"expected a version {FORMAT_VERSION} {algorithm} model")
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    return metadata, arrays
