"""Utilities for long/short history splitting and deterministic compression."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import List, Sequence

import numpy as np


@dataclass
class HistoryCompressionConfig:
    """Config for history compression."""

    use_history_compression: bool = False
    history_threshold: int = 100
    compression_type: str = "attention"
    compressed_dim: int = 16


def split_history(history_sids: Sequence[str], threshold: int) -> tuple[List[str], List[str]]:
    """Split history into long and short chunks."""
    if threshold <= 0:
        return list(history_sids), []
    if len(history_sids) <= threshold:
        return [], list(history_sids)
    split_idx = len(history_sids) - threshold
    return list(history_sids[:split_idx]), list(history_sids[split_idx:])


def _sid_to_vector(sid: str, dim: int) -> np.ndarray:
    """Map a SID string to a deterministic dense vector."""
    digest = md5(sid.encode("utf-8")).digest()
    vals = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    if dim > vals.shape[0]:
        vals = np.pad(vals, (0, dim - vals.shape[0]), mode="wrap")
    vals = vals[:dim]
    vals = (vals / 255.0) * 2.0 - 1.0
    return vals


def compress_long_history(
    long_history: Sequence[str],
    compression_type: str = "attention",
    compressed_dim: int = 16,
) -> np.ndarray:
    """Compress long history into a fixed-size vector."""
    if len(long_history) == 0:
        return np.zeros(compressed_dim, dtype=np.float32)

    mat = np.stack([_sid_to_vector(sid, compressed_dim) for sid in long_history], axis=0)

    if compression_type == "mean":
        vec = mat.mean(axis=0)
    elif compression_type == "attention":
        scores = np.linalg.norm(mat, axis=1)
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / np.maximum(weights.sum(), 1e-8)
        vec = (mat * weights[:, None]).sum(axis=0)
    elif compression_type == "mlp":
        mean_vec = mat.mean(axis=0)
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((compressed_dim, compressed_dim)).astype(np.float32) / np.sqrt(compressed_dim)
        w2 = rng.standard_normal((compressed_dim, compressed_dim)).astype(np.float32) / np.sqrt(compressed_dim)
        hidden = np.tanh(mean_vec @ w1)
        vec = np.tanh(hidden @ w2)
    else:
        raise ValueError(f"Unsupported compression_type: {compression_type}")

    return vec.astype(np.float32)


def format_compressed_history(
    history_sids: Sequence[str],
    cfg: HistoryCompressionConfig,
) -> tuple[str, dict]:
    """Return a prompt-ready history string and metadata."""
    long_history, short_history = split_history(history_sids, cfg.history_threshold)
    meta = {
        "long_history_len": len(long_history),
        "short_history_len": len(short_history),
        "history_total_len": len(history_sids),
    }

    if not cfg.use_history_compression or len(long_history) == 0:
        return ", ".join(history_sids), meta

    vec = compress_long_history(
        long_history=long_history,
        compression_type=cfg.compression_type,
        compressed_dim=cfg.compressed_dim,
    )
    vec_str = ",".join(f"{x:.4f}" for x in vec)
    compressed_token = f"<compressed_{cfg.compression_type}:{vec_str}>"

    combined = [compressed_token] + short_history
    return ", ".join(combined), meta
