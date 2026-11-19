"""Validated, JSON-only catalog trees and variable-length MiniOneRec SIDs.

This is a fixed-catalog mapping, not a nearest-centroid insertion model. Tree
nodes describe clusters; terminal clusters hold item leaves. A singleton cluster
does not need an additional cluster token.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .sid_common import canonical_item_ids, index_statistics, positive_int, vectors


ALGORITHM = "hierarchical_kmeans"
ALGORITHM_VERSION = "2.0.0"
TREE_FORMAT_VERSION = 2
CATALOG_HASH_RULE = "lexicographic string IDs; compact UTF-8 JSON [IDs, shape]; little-endian float64 values"
STRATEGY_DEFAULTS = {"method": "balanced", "greedy_threshold": 2000, "greedy_n_init": 5, "greedy_max_iter": 20}
CONFIG_KEYS = {"branch_factor", "seed", "n_init", "max_iter", "tol", "n_jobs"} | STRATEGY_DEFAULTS.keys()
SPLIT_BACKENDS = {"constrained": "k_means_constrained.KMeansConstrained", "greedy": "sklearn.cluster.KMeans"}


def check_artifact_version(payload):
    if not isinstance(payload, dict):
        raise ValueError("expected a hierarchical artifact object")
    version = payload.get("format_version")
    if (type(version) is not int or version != TREE_FORMAT_VERSION
            or payload.get("algorithm") != ALGORITHM or payload.get("algorithm_version") != ALGORITHM_VERSION):
        raise ValueError("unsupported hierarchical format or algorithm version")


def split_method(config, size):
    """Resolve a nonterminal split independently for this node."""
    if size <= config["branch_factor"]:
        raise ValueError("terminal nodes do not use a split solver")
    if config["method"] == "greedy" or (config["method"] == "hybrid" and size > config["greedy_threshold"]):
        return "greedy"
    return "constrained"


def validate_config(config):
    """Validate the required parameters and normalize their types."""
    if not isinstance(config, dict):
        raise ValueError("tree config must be an object")
    if set(config) != CONFIG_KEYS:
        raise ValueError("tree config has missing or unsupported parameters")
    result = {
        "branch_factor": positive_int(config["branch_factor"], "branch_factor", 2),
        "seed": positive_int(config["seed"], "seed", 0),
        "n_init": positive_int(config["n_init"], "n_init"),
        "max_iter": positive_int(config["max_iter"], "max_iter"),
        "n_jobs": positive_int(config["n_jobs"], "n_jobs"),
    }
    if result["seed"] > 2**32 - 1:
        raise ValueError("seed must be <= 2**32 - 1")
    tol = config["tol"]
    if isinstance(tol, bool) or not isinstance(tol, Real):
        raise ValueError("tol must be a finite nonnegative number")
    if not math.isfinite(float(tol)) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")
    result["tol"] = float(tol)
    method = config["method"]
    if method not in ("balanced", "greedy", "hybrid"):
        raise ValueError("method must be balanced, greedy, or hybrid")
    result.update(method=method,
                  greedy_threshold=positive_int(config["greedy_threshold"], "greedy_threshold", 0),
                  greedy_n_init=positive_int(config["greedy_n_init"], "greedy_n_init"),
                  greedy_max_iter=positive_int(config["greedy_max_iter"], "greedy_max_iter"))
    return result


def canonical_catalog(embeddings, item_ids):
    """Align rows explicitly, retaining original numeric values in the digest.

    The solver uses float32, consistent with the shared SID input validation.
    The digest uses the input's values converted to float64 so a float64 change
    below float32 precision still invalidates a saved catalog. Equivalent numeric
    values in float16/32/64 share a digest. JSON member iteration is never used
    as evidence of a row mapping: callers must supply one ID per input row.
    """
    original = np.asarray(embeddings)
    working = vectors(original)
    # The catalog digest must not silently round distinct original numbers to
    # the same float64 value. The solver's float32 conversion is intentional;
    # loss of identity in the catalog fingerprint is not.
    if np.issubdtype(original.dtype, np.integer):
        upper = np.any(original > 2**53)
        lower = original.dtype.kind != "u" and np.any(original < -(2**53))
        if upper or lower:
            raise ValueError("integer embeddings must lie in [-2**53, 2**53] for exact catalog fingerprinting")
    if (np.issubdtype(original.dtype, np.floating)
            and np.finfo(original.dtype).nmant > np.finfo(np.float64).nmant):
        raise ValueError("embedding precision greater than float64 is unsupported for exact catalog fingerprinting")
    if not isinstance(item_ids, (list, tuple, np.ndarray)):
        raise ValueError("item_ids must be a row-aligned sequence")
    ids = canonical_item_ids(item_ids, len(working))
    order = sorted(range(len(ids)), key=ids.__getitem__)
    ids = [ids[position] for position in order]
    working = np.ascontiguousarray(working[order])
    exact = np.asarray(original[order], dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(json.dumps([ids, list(working.shape)], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(memoryview(exact).cast("B"))
    return working, ids, digest.hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_artifact_json(data):
    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_json_object, parse_constant=invalid_constant)


def read_artifact_json(path):
    return parse_artifact_json(Path(path).read_bytes())


class HierarchicalTree:
    """Complete cluster topology, canonical item IDs and catalog fingerprint."""

    def __init__(self, *, nodes, item_ids, dimension, catalog_hash, config):
        # Own the JSON state: callers cannot later mutate a fit through aliases.
        self.nodes = json.loads(json.dumps(nodes, allow_nan=False))
        self.item_ids = list(item_ids)
        self.dimension = dimension
        self.catalog_hash = catalog_hash
        self.config = config
        self.validate()

    def validate(self):
        """Reject malformed topology, wrong capacities and noncanonical IDs."""
        self.config = validate_config(self.config)
        _integer(self.dimension, "dimension", 1)
        if not isinstance(self.item_ids, list) or not self.item_ids:
            raise ValueError("tree must contain at least one item")
        ids = canonical_item_ids(self.item_ids, len(self.item_ids))
        if ids != self.item_ids or ids != sorted(ids):
            raise ValueError("tree item_ids must be canonical lexicographically sorted strings")
        if not isinstance(self.catalog_hash, str) or not re.fullmatch("[0-9a-f]{64}", self.catalog_hash):
            raise ValueError("catalog_hash must be a SHA256 hexadecimal string")
        if not isinstance(self.nodes, list) or not self.nodes:
            raise ValueError("tree must contain a root node")
        branch_factor = self.config["branch_factor"]
        keys = {"id", "parent", "depth", "size", "token", "children", "items"}
        for position, node in enumerate(self.nodes):
            if not isinstance(node, dict) or set(node) != keys:
                raise ValueError("cluster node has missing or unexpected fields")
            if _integer(node["id"], "node id") != position:
                raise ValueError("node IDs must be consecutive in preorder")
            _integer(node["depth"], "node depth")
            _integer(node["size"], "node size", 1)
            if not isinstance(node["children"], list) or not isinstance(node["items"], list):
                raise ValueError("node children and items must be lists")
            if node["parent"] is not None:
                _integer(node["parent"], "node parent")
            for child in node["children"]:
                _integer(child, "child id")
                if child >= len(self.nodes):
                    raise ValueError("child id is outside the tree")
            if any(not isinstance(item, str) for item in node["items"]):
                raise ValueError("terminal item IDs must be strings")
            expected_token = f"<node_{position}>" if position and node["size"] > 1 else None
            if node["token"] != expected_token:
                raise ValueError("cluster token does not match the canonical node ID")

        # Iterative postorder avoids trusting file-provided recursion depth.
        visiting = [(0, None, 0, False)]
        seen = set()
        descendants = {}
        preorder = []
        while visiting:
            node_id, parent, depth, exiting = visiting.pop()
            node = self.nodes[node_id]
            if not exiting:
                if node_id in seen:
                    raise ValueError("tree contains a cycle or multiply referenced node")
                seen.add(node_id)
                preorder.append(node_id)
                if node["parent"] != parent or node["depth"] != depth:
                    raise ValueError("incorrect node parent or depth")
                if node["children"]:
                    if node["items"] or len(node["children"]) != branch_factor or node["size"] <= branch_factor:
                        raise ValueError("split nodes must have K children, N>K and no direct items")
                    visiting.append((node_id, parent, depth, True))
                    visiting.extend((child, node_id, depth + 1, False)
                                    for child in reversed(node["children"]))
                else:
                    if not 1 <= len(node["items"]) <= branch_factor:
                        raise ValueError("terminal cluster must hold between 1 and K items")
                    if node["items"] != sorted(node["items"]) or len(node["items"]) != node["size"]:
                        raise ValueError("terminal items must be sorted and match node size")
                    descendants[node_id] = node["items"]
            else:
                children = node["children"]
                lower = node["size"] // branch_factor
                sizes = [self.nodes[child]["size"] for child in children]
                if any(size < lower or size > lower + 1 for size in sizes):
                    raise ValueError("child cluster sizes violate balanced capacities")
                if sum(sizes) != node["size"]:
                    raise ValueError("child cluster sizes do not sum to parent size")
                minima = [min(descendants[child]) for child in children]
                if minima != sorted(minima):
                    raise ValueError("children must be sorted by smallest canonical item ID")
                descendants[node_id] = [item for child in children for item in descendants.pop(child)]
        if preorder != list(range(len(self.nodes))):
            raise ValueError("tree has unreachable nodes or noncanonical preorder")
        members = descendants[0]
        if len(members) != len(self.item_ids) or sorted(members) != self.item_ids:
            raise ValueError("every catalog item must occur in exactly one terminal cluster")
        if self.nodes[0]["size"] != len(self.item_ids):
            raise ValueError("root size must equal the catalog size")
        return self

    def to_index(self):
        """Return variable-length token lists without START, END or padding."""
        self.validate()
        leaf_tokens = {item: f"<leaf_{rank}>" for rank, item in enumerate(self.item_ids)}
        index = {}
        pending = [(0, [])]
        while pending:
            node_id, prefix = pending.pop()
            node = self.nodes[node_id]
            path = prefix + ([node["token"]] if node["token"] is not None else [])
            if node["children"]:
                pending.extend((child, path) for child in reversed(node["children"]))
            else:
                for item in node["items"]:
                    index[item] = path + [leaf_tokens[item]]
        return {item: index[item] for item in self.item_ids}

    def encode(self, embeddings, item_ids):
        """Re-export only an exact complete catalog, allowing row permutations."""
        values, ids, fingerprint = canonical_catalog(embeddings, item_ids)
        if ids != self.item_ids:
            raise ValueError("item IDs do not match the fitted catalog; unknown or missing items are unsupported")
        if values.shape[1] != self.dimension or fingerprint != self.catalog_hash:
            raise ValueError("embeddings do not match the fitted catalog")
        return self.to_index()

    def statistics(self):
        index = self.to_index()
        stats = index_statistics(index)
        levels = defaultdict(list)
        split_counts = Counter({method: 0 for method in SPLIT_BACKENDS})
        splits_by_depth = defaultdict(lambda: {method: 0 for method in SPLIT_BACKENDS})
        for node in self.nodes:
            levels[node["depth"]].append(node["size"])
            if node["children"]:
                method = split_method(self.config, node["size"])
                split_counts[method] += 1
                splits_by_depth[node["depth"]][method] += 1
        lengths = [len(tokens) for tokens in index.values()]
        paths = {tuple(tokens) for tokens in index.values()}
        prefix_collisions = sum(tuple(tokens[:length]) in paths
                                for tokens in index.values() for length in range(1, len(tokens)))
        return {
            "items": len(self.item_ids),
            "embedding_dimension": self.dimension,
            "branch_factor": self.config["branch_factor"],
            "method": self.config["method"],
            "split_counts": dict(split_counts),
            "split_counts_by_depth": {str(d): counts for d, counts in sorted(splits_by_depth.items())},
            "cluster_nodes": len(self.nodes),
            "split_nodes": sum(bool(node["children"]) for node in self.nodes),
            "terminal_clusters": sum(not node["children"] for node in self.nodes),
            "internal_node_tokens": sum(node["token"] is not None for node in self.nodes),
            "item_leaf_nodes": len(self.item_ids),
            "vocabulary_size": stats["observed_vocabulary_size"],
            "max_cluster_depth": max(levels),
            "max_depth": max(lengths),
            "mean_depth": sum(lengths) / len(lengths),
            "depth_definition": "number of tokens/edges from omitted root to item leaf after singleton compression",
            "sid_length_distribution": stats["length_histogram"],
            "levels": [{"depth": depth, "cluster_count": len(sizes), "min_size": min(sizes),
                        "max_size": max(sizes), "mean_size": sum(sizes) / len(sizes),
                        "size_histogram": {str(size): count for size, count in sorted(Counter(sizes).items())}}
                       for depth, sizes in sorted(levels.items())],
            "prefix_collisions": prefix_collisions,
            "sid_collisions": stats["excess_collision_count"],
            "sid_collision_rate": stats["excess_collision_rate"],
            "index_statistics": stats,
            "uniqueness_explanation": "unique item leaf tokens guarantee uniqueness; this is not a clustering-quality metric",
        }

    def to_dict(self):
        self.validate()
        return {
            "format_version": TREE_FORMAT_VERSION,
            "algorithm": ALGORITHM,
            "algorithm_version": ALGORITHM_VERSION,
            "catalog_hash_rule": CATALOG_HASH_RULE,
            "catalog_hash": self.catalog_hash,
            "dimension": self.dimension,
            "config": dict(self.config),
            "item_ids": list(self.item_ids),
            "nodes": json.loads(json.dumps(self.nodes)),
        }

    def save(self, path, overwrite=False):
        """Atomically write one model; refuse existing files by default."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"tree already exists: {path}")
        if path.is_symlink():
            raise ValueError("tree destination must not be a symbolic link")
        payload = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".json", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
            if overwrite:
                os.replace(temporary, path)
            else:
                # link is an atomic create-if-absent; unlike replace it cannot
                # overwrite a file created by a concurrent publisher.
                os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        return cls.from_dict(read_artifact_json(path))

    @classmethod
    def from_dict(cls, payload):
        fields = {"format_version", "algorithm", "algorithm_version", "catalog_hash_rule",
                  "catalog_hash", "dimension", "config", "item_ids", "nodes"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("tree file has missing or unexpected fields")
        check_artifact_version(payload)
        if payload["catalog_hash_rule"] != CATALOG_HASH_RULE:
            raise ValueError("unsupported catalog hash rule")
        return cls(nodes=payload["nodes"], item_ids=payload["item_ids"], dimension=payload["dimension"],
                   catalog_hash=payload["catalog_hash"], config=payload["config"])
