"""Global capacity assignment to fixed centroids for hierarchical K-means."""

import numpy as np


def greedy_labels(values, centers):
    """Return raw centroid labels in (squared distance, row, center) priority.

    Inputs use the tree's float32 working space. The first N % K centers get
    the extra slots before children are sorted for SID numbering.
    """
    if (values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]
            or not 1 <= len(centers) <= len(values) or values.shape[1] == 0):
        raise ValueError("invalid embedding or centroid shape for greedy assignment")
    if values.dtype != np.float32 or centers.dtype != np.float32:
        raise ValueError("greedy assignment requires float32 embeddings and centroids")
    if not np.isfinite(centers).all():
        raise ValueError("greedy centroids must be finite")

    n, dimension = values.shape
    k = len(centers)
    distances = np.empty((n, k), dtype=np.float32)
    # One center and up to 1 MiB of coordinate differences at a time (at
    # least one row). Direct subtraction preserves small distances between
    # nearby vectors that a norm/dot-product expansion could cancel out.
    block_rows = max(1, (1 << 20) // (dimension * values.itemsize))
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, n, block_rows):
            block = values[start:start + block_rows]
            for j, center in enumerate(centers):
                delta = block - center
                np.square(delta, out=delta)
                distances[start:start + len(block), j] = delta.sum(axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("non-finite or overflowing float32 squared distance in greedy assignment")

    # C-order encodes (row, center). A stable sort on just distance therefore
    # implements the tuple order, including ties across items and centers.
    candidates = np.argsort(distances.ravel(order="C"), kind="stable")
    capacity = np.full(k, n // k, dtype=np.intp)
    capacity[:n % k] += 1
    remaining = capacity.copy()
    labels = np.full(n, -1, dtype=np.intp)
    unassigned = n
    for candidate in candidates:
        row, center = divmod(int(candidate), k)
        if labels[row] < 0 and remaining[center]:
            labels[row] = center
            remaining[center] -= 1
            unassigned -= 1
            if unassigned == 0:
                break
    if unassigned or not np.array_equal(np.bincount(labels, minlength=k), capacity):
        raise RuntimeError("greedy assignment did not fill the prescribed centroid capacities")
    return labels
