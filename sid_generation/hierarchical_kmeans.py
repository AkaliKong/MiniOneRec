"""Fixed-catalog hierarchical K-means with per-node capacity constraints."""

import logging
import time

import numpy as np

from .hierarchical_greedy import greedy_labels
from .hierarchical_tree import SPLIT_BACKENDS, HierarchicalTree, canonical_catalog, split_method, validate_config


class HierarchicalKMeans:
    def __init__(self, branch_factor=10, seed=0, n_init=10, max_iter=300, tol=1e-4, n_jobs=1,
                 method="balanced", greedy_threshold=2000, greedy_n_init=5, greedy_max_iter=20):
        self.config = validate_config({"branch_factor": branch_factor, "seed": seed, "n_init": n_init,
                                       "max_iter": max_iter, "tol": tol, "n_jobs": n_jobs,
                                       "method": method, "greedy_threshold": greedy_threshold,
                                       "greedy_n_init": greedy_n_init, "greedy_max_iter": greedy_max_iter})
        self.tree = None
        self.split_diagnostics = []

    def fit(self, embeddings, item_ids):
        """Build a complete deterministic catalog tree in canonical ID order.

        Each nonterminal node splits a subset of the original working vectors.
        Child order is canonicalized only after the chosen assignment is final.
        """
        from k_means_constrained import KMeansConstrained
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits

        start = time.perf_counter()
        values, ids, fingerprint = canonical_catalog(embeddings, item_ids)
        branch_factor = self.config["branch_factor"]
        nodes = []
        diagnostics = []

        def build(rows, parent, depth):
            node_id = len(nodes)
            size = len(rows)
            node = {"id": node_id, "parent": parent, "depth": depth, "size": size,
                    "token": f"<node_{node_id}>" if parent is not None and size > 1 else None,
                    "children": [], "items": []}
            nodes.append(node)
            if size <= branch_factor:
                node["items"] = [ids[row] for row in rows]
                return node_id
            lower = size // branch_factor
            upper = lower + 1
            method = split_method(self.config, size)
            start = time.perf_counter()
            try:
                if method == "constrained":
                    estimator = KMeansConstrained(n_clusters=branch_factor, size_min=lower, size_max=upper,
                                                  random_state=self.config["seed"], n_init=self.config["n_init"],
                                                  max_iter=self.config["max_iter"], tol=self.config["tol"],
                                                  n_jobs=self.config["n_jobs"])
                    labels = np.asarray(estimator.fit_predict(values[rows]))
                else:
                    local = values[rows]
                    estimator = KMeans(n_clusters=branch_factor, init="k-means++", algorithm="lloyd",
                                       random_state=self.config["seed"], tol=self.config["tol"],
                                       n_init=self.config["greedy_n_init"], max_iter=self.config["greedy_max_iter"])
                    estimator.fit(local)
                    if estimator.cluster_centers_.shape != (branch_factor, values.shape[1]):
                        raise ValueError("greedy solver returned the wrong centroid shape")
                    labels = greedy_labels(local, estimator.cluster_centers_)
                    del local
            except Exception as exc:
                raise RuntimeError(f"{method} K-means failed at node {node_id} (N={size}, "
                                   f"K={branch_factor}, sizes={lower}..{upper}); no fallback was used: {exc}") from exc
            elapsed = time.perf_counter() - start
            if labels.shape != (size,) or not np.issubdtype(labels.dtype, np.integer):
                raise RuntimeError(f"{method} solver returned invalid cluster labels")
            if np.any(labels < 0) or np.any(labels >= branch_factor):
                raise RuntimeError(f"{method} solver returned cluster labels outside [0, K)")
            counts = np.bincount(labels, minlength=branch_factor)
            if np.any(counts < lower) or np.any(counts > upper):
                raise RuntimeError(f"{method} solver violated capacities at node {node_id}: {counts.tolist()}")
            if not np.isfinite(estimator.cluster_centers_).all() or not np.isfinite(estimator.inertia_):
                raise RuntimeError(f"{method} solver returned non-finite centers or inertia at node {node_id}")
            groups = [rows[labels == label] for label in range(branch_factor)]
            groups.sort(key=lambda group: ids[group[0]])
            record = {"node_id": node_id, "depth": depth, "items": size, "size_min": lower,
                      "size_max": upper, "child_sizes": [len(group) for group in groups],
                      "split_method": method, "backend": SPLIT_BACKENDS[method],
                      "elapsed_seconds": elapsed}
            diagnostics.append(record)
            logging.info("cluster node=%d depth=%d items=%d sizes=%s seconds=%.3f", node_id, depth,
                         size, record["child_sizes"], elapsed)
            for group in groups:
                node["children"].append(build(group, node_id, depth + 1))
            return node_id

        # Bound BLAS/OpenMP oversubscription; n_jobs remains an explicit solver
        # parameter. Canonical rows and single-thread defaults stabilize ties.
        with threadpool_limits(limits=1):
            build(np.arange(len(ids)), None, 0)
        fitted = HierarchicalTree(nodes=nodes, item_ids=ids, dimension=values.shape[1],
                                  catalog_hash=fingerprint, config=self.config)
        self.tree = fitted
        self.split_diagnostics = diagnostics
        self.fit_seconds = time.perf_counter() - start
        return self

    def encode(self, embeddings, item_ids):
        if self.tree is None:
            raise ValueError("fit or load a tree before encoding")
        return self.tree.encode(embeddings, item_ids)
