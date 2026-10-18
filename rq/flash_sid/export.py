"""Item alignment, collision accounting and MiniOneRec index export."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def validate_item_ids(item_ids: Sequence[str | int], count: int) -> list[str]:
    if len(item_ids) != count:
        raise ValueError(f"Expected {count} item IDs, got {len(item_ids)}")
    result = []
    for value in item_ids:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Item IDs must be strings or integers, not booleans")
        value = str(value)
        if not value or value.strip() != value or any(c in value for c in "\r\n\t"):
            raise ValueError("Item IDs must be nonempty and contain no surrounding whitespace or tabs")
        result.append(value)
    if len(set(result)) != count:
        raise ValueError("Item IDs must be unique after conversion to strings")
    return result


def load_item_ids(count: int, item_ids_path: Path | None = None,
                  item_meta_path: Path | None = None) -> list[str]:
    """Use an explicit row-aligned list, or validate upstream's 0..N-1 convention."""
    if item_ids_path is not None and item_meta_path is not None:
        raise ValueError("Specify item_ids_path or item_meta_path, not both")
    if item_ids_path is not None:
        values = json.loads(Path(item_ids_path).read_text(encoding="utf-8-sig"))
        if not isinstance(values, list):
            raise ValueError("--item-ids must contain a JSON list in embedding row order")
        return validate_item_ids(values, count)
    expected = [str(i) for i in range(count)]
    if item_meta_path is not None:
        values = json.loads(Path(item_meta_path).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict) or set(values) != set(expected):
            raise ValueError("--item-meta requires exactly IDs 0..N-1. For other IDs use --item-ids with explicit row order")
    return expected


def validate_codes(codes: np.ndarray, codebook_size: int | None = None) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2 or min(codes.shape) < 1 or codes.dtype.kind not in "iu":
        raise ValueError("Codes must be a nonempty 2-D integer array")
    if np.any(codes < 0) or (codebook_size is not None and np.any(codes >= codebook_size)):
        raise ValueError("Codes lie outside the codebook")
    return codes


def collision_metrics(codes: np.ndarray, codebook_size: int) -> dict:
    codes = validate_codes(codes, codebook_size)
    n = len(codes)
    groups = Counter(tuple(int(v) for v in row) for row in codes)
    sizes = list(groups.values())
    positions = []
    for j in range(codes.shape[1]):
        _, counts = np.unique(codes[:, j], return_counts=True)
        probabilities = counts / n
        positions.append({
            "position": j,
            "used_codes": len(counts),
            "utilization": len(counts) / codebook_size,
            "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
            "largest_code_bucket": int(counts.max()),
        })
    return {
        "items": n,
        "unique_raw_sids": len(groups),
        "excess_collision_items": n - len(groups),
        "raw_collision_rate": (n - len(groups)) / n,
        "items_in_collision_groups": sum(s for s in sizes if s > 1),
        "collision_groups": sum(s > 1 for s in sizes),
        "max_collision_group_size": max(sizes),
        "positions": positions,
    }


def build_minionerec_index(codes: np.ndarray, item_ids: Sequence[str | int],
                         suffix_size: int = 256) -> dict[str, list[str]]:
    """Three semantic positions + deterministic catalog-local ordinal suffix.

    The suffix ranks canonical string IDs lexicographically within each bucket.
    Reordering input rows is safe; adding/removing catalog items may change ranks.
    """
    codes = validate_codes(codes)
    if codes.shape[1] != 3:
        raise ValueError("MiniOneRec export requires exactly 3 semantic code positions")
    if isinstance(suffix_size, bool) or not isinstance(suffix_size, int) or suffix_size < 1:
        raise ValueError("suffix_size must be a positive integer")
    ids = validate_item_ids(item_ids, len(codes))
    groups = defaultdict(list)
    for item_id, row in zip(ids, codes):
        groups[tuple(int(c) for c in row)].append(item_id)
    max_group = max(map(len, groups.values()))
    if max_group > suffix_size:
        raise ValueError(f"Largest collision group has {max_group} items, exceeding suffix_size={suffix_size}. Increase --suffix-size or improve embeddings/codebook size")
    suffixes = {}
    for members in groups.values():
        for rank, item_id in enumerate(sorted(members)):
            suffixes[item_id] = rank
    return {
        item_id: [f"<a_{int(row[0])}>", f"<b_{int(row[1])}>",
                  f"<c_{int(row[2])}>", f"<d_{suffixes[item_id]}>"]
        for item_id, row in zip(ids, codes)
    }
