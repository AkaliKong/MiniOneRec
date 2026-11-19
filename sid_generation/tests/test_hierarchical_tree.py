"""Mathematical, determinism and persistence checks using the real solver."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from sid_generation.hierarchical_kmeans import HierarchicalKMeans
from sid_generation.hierarchical_tree import HierarchicalTree, canonical_catalog


def fit(values, ids=None, k=3, **kwargs):
    if ids is None:
        ids = [str(position) for position in range(len(values))]
    return HierarchicalKMeans(branch_factor=k, n_init=2, max_iter=30, **kwargs).fit(values, ids)


def assert_partition(tree):
    k = tree.config["branch_factor"]
    encountered = []
    for node in tree.nodes:
        if node["children"]:
            assert len(node["children"]) == k
            sizes = [tree.nodes[child]["size"] for child in node["children"]]
            assert all(node["size"] // k <= size <= node["size"] // k + 1 for size in sizes)
            assert sum(sizes) == node["size"]
        else:
            assert 1 <= len(node["items"]) <= k
            encountered.extend(node["items"])
    assert sorted(encountered) == tree.item_ids
    assert len(encountered) == len(set(encountered))
    index = tree.to_index()
    assert len({tuple(path) for path in index.values()}) == len(index)
    paths = {tuple(path) for path in index.values()}
    assert not any(tuple(path[:i]) in paths for path in index.values() for i in range(1, len(path)))
    assert len({path[-1] for path in index.values()}) == len(index)


@pytest.mark.parametrize("k,count", [(2, 5), (2, 19), (3, 11), (3, 28), (10, 101), (10, 100)])
def test_recursive_balance_nondivisible_and_complete_coverage(k, count):
    values = np.random.default_rng(72).normal(size=(count, 6))
    model = fit(values, k=k)
    assert_partition(model.tree)
    assert model.tree.statistics()["split_nodes"] > 0
    for split in model.split_diagnostics:
        assert split["size_max"] == split["items"] // k + 1


@pytest.mark.parametrize("count,k", [(1, 2), (2, 2), (3, 10)])
def test_small_catalog_omits_root_and_unnecessary_clusters(count, k):
    tree = fit(np.zeros((count, 3)), k=k).tree
    assert len(tree.nodes) == 1
    assert all(len(tokens) == 1 for tokens in tree.to_index().values())
    assert tree.statistics()["vocabulary_size"] == count
    assert_partition(tree)


def test_seed_reproducible_with_duplicates_and_reordered_rows():
    rng = np.random.default_rng(304)
    values = rng.normal(size=(29, 4))
    values[3:7] = values[1]
    ids = [f"item-{position * 7}" for position in range(len(values))]
    model = fit(values, ids, seed=42)
    duplicate = fit(values, ids, seed=42)
    permutation = rng.permutation(len(values))
    shuffled = fit(values[permutation], [ids[i] for i in permutation], seed=42)
    assert model.tree.to_dict() == duplicate.tree.to_dict() == shuffled.tree.to_dict()
    assert model.encode(values[permutation], [ids[i] for i in permutation]) == model.tree.to_index()
    assert_partition(model.tree)


@pytest.mark.parametrize("k", [2, 3, 10])
def test_identical_vectors_are_constrained_without_losing_items(k):
    values = np.ones((31, 8), dtype=np.float32)
    model = fit(values, k=k)
    assert_partition(model.tree)
    assert model.tree.to_index() == fit(values, k=k).tree.to_index()


def test_canonical_leaf_ids_not_embedding_positions_and_zero_is_real():
    ids = ["77", "0", "1001", "A", "2"]
    tree = fit(np.arange(15).reshape(5, 3), ids, k=2).tree
    index = tree.to_index()
    assert list(index) == sorted(ids)
    for rank, item in enumerate(sorted(ids)):
        assert index[item][-1] == f"<leaf_{rank}>"
    assert "0" in index
    assert_partition(tree)


def test_singleton_compression_supports_variable_lengths():
    tree = fit(np.array([[0], [1], [2], [20], [21]], dtype=np.float32), k=2).tree
    assert len({len(path) for path in tree.to_index().values()}) == 2
    assert all(node["token"] is None for node in tree.nodes if node["size"] == 1)
    assert tree.statistics()["prefix_collisions"] == 0
    assert tree.statistics()["sid_collision_rate"] == 0


@pytest.mark.parametrize("values", [np.empty((0, 3)), np.empty((3, 0)), np.ones(3),
                                   [[float("nan")]], [[float("inf")]], [[1e300]], [[1j]], [["word"]]])
def test_invalid_embeddings_fail(values):
    with pytest.raises(ValueError):
        fit(values)


@pytest.mark.parametrize("k", [0, 1, -3, True, 2.5, "3"])
def test_invalid_branch_factor_fails(k):
    with pytest.raises(ValueError):
        HierarchicalKMeans(branch_factor=k)


@pytest.mark.parametrize("kwargs", [{"seed": -1}, {"seed": 2**32}, {"seed": True},
                                     {"n_init": 0}, {"max_iter": 0}, {"n_jobs": 0},
                                     {"tol": -1}, {"tol": float("nan")}, {"tol": True}])
def test_invalid_solver_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        HierarchicalKMeans(**kwargs)


@pytest.mark.parametrize("ids", [["a"], ["a", "a"], [0, "0"], [" a", "b"], [True, 2], {"a": 0, "b": 1}])
def test_invalid_item_mapping_fails(ids):
    with pytest.raises(ValueError):
        fit(np.ones((2, 2)), ids)


def test_solver_failure_is_explicit_and_has_no_fallback():
    with patch("k_means_constrained.KMeansConstrained.fit_predict", side_effect=ValueError("infeasible")):
        with pytest.raises(RuntimeError, match="no fallback"):
            fit(np.ones((5, 2)), k=2)


@pytest.mark.parametrize("labels", [np.zeros(5, dtype=int), np.array([0, 0, 1, 1, 2]),
                                   np.ones(5), np.ones(4, dtype=int)])
def test_invalid_solver_result_is_rejected(labels):
    with patch("k_means_constrained.KMeansConstrained.fit_predict", return_value=labels):
        with pytest.raises(RuntimeError, match="solver"):
            fit(np.ones((5, 2)), k=2)


def test_tree_round_trip_is_exact_and_encode_checks_catalog(tmp_path):
    values = np.random.default_rng(1).normal(size=(17, 4))
    ids = [str(13 * i) for i in range(len(values))]
    tree = fit(values, ids).tree
    path = tmp_path / "tree.json"
    tree.save(path)
    restored = HierarchicalTree.load(path)
    assert restored.to_dict() == tree.to_dict()
    assert restored.statistics() == tree.statistics()
    assert restored.encode(values, ids) == tree.to_index()
    with pytest.raises(FileExistsError):
        tree.save(path)
    changed = values.copy()
    changed[0, 0] += 1e-10  # smaller than float32 precision; still a different catalog
    with pytest.raises(ValueError, match="embeddings"):
        restored.encode(changed, ids)
    with pytest.raises(ValueError, match="item IDs"):
        restored.encode(values, ["unknown"] + ids[1:])
    with pytest.raises(ValueError, match="item IDs"):
        restored.encode(values[:-1], ids[:-1])
    with pytest.raises(ValueError, match="embeddings"):
        restored.encode(values[:, :-1], ids)


def test_encoding_before_fit_fails():
    with pytest.raises(ValueError, match="fit or load"):
        HierarchicalKMeans().encode([[1]], ["0"])


def test_numeric_dtype_does_not_change_catalog_fingerprint():
    values = np.array([[1.5, 0], [2, 3]], dtype=np.float16)
    assert canonical_catalog(values, ["b", "a"])[2] == canonical_catalog(values.astype(np.float64), ["b", "a"])[2]


@pytest.mark.parametrize("values", [np.array([[2**53 + 1]], dtype=np.int64),
                                   np.array([[-(2**53) - 1]], dtype=np.int64),
                                   np.array([[2**53 + 1]], dtype=np.uint64)])
def test_rejects_integer_values_that_cannot_be_fingerprinted_exactly(values):
    with pytest.raises(ValueError, match="exact catalog fingerprinting"):
        canonical_catalog(values, ["0"])


def test_exact_integer_fingerprint_boundary_keeps_adjacent_values_distinct():
    a = np.array([[-(2**53), 2**53]], dtype=np.int64)
    b = np.array([[-(2**53) + 1, 2**53 - 1]], dtype=np.int64)
    assert canonical_catalog(a, ["0"])[2] != canonical_catalog(b, ["0"])[2]
    assert canonical_catalog(a, ["0"])[2] == canonical_catalog(a.astype(np.float64), ["0"])[2]


def test_platform_longdouble_fingerprint_precision_rule():
    values = np.ones((1, 2), dtype=np.longdouble)
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        with pytest.raises(ValueError, match="precision greater than float64"):
            canonical_catalog(values, ["0"])
    else:
        assert canonical_catalog(values, ["0"])[2] == canonical_catalog(values.astype(np.float64), ["0"])[2]


@pytest.fixture
def saved_payload():
    return fit(np.arange(24).reshape(12, 2), k=3).tree.to_dict()


@pytest.mark.parametrize("mutation", [
    lambda payload: payload.update(format_version=999),
    lambda payload: payload.update(format_version=1),
    lambda payload: payload.update(format_version=True),
    lambda payload: payload.update(algorithm="other"),
    lambda payload: payload.update(catalog_hash="bad"),
    lambda payload: payload.update(dimension=0),
    lambda payload: payload["item_ids"].reverse(),
    lambda payload: payload["config"].update(branch_factor=2),
    lambda payload: payload["nodes"][0].update(children=[0, 1, 2]),
    lambda payload: payload["nodes"][0]["children"].append(1),
    lambda payload: payload["nodes"][0].update(children=[999, 1, 2]),
    lambda payload: payload["nodes"][1].update(parent=99),
    lambda payload: payload["nodes"][1].update(depth=99),
    lambda payload: payload["nodes"][1].update(size=99),
    lambda payload: payload["nodes"][1].update(token="<leaf_1>"),
    lambda payload: payload["nodes"][-1].update(items=["unknown"]),
    lambda payload: payload["nodes"].append(copy.deepcopy(payload["nodes"][-1])),
])
def test_corrupt_tree_models_are_rejected(tmp_path, saved_payload, mutation):
    mutation(saved_payload)
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(saved_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        HierarchicalTree.load(path)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"format_version":2,"format_version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        HierarchicalTree.load(path)


def test_statistics_describe_actual_tokens_and_levels():
    tree = fit(np.random.default_rng(7).normal(size=(17, 5)), k=3).tree
    stats = tree.statistics()
    index = tree.to_index()
    assert stats["vocabulary_size"] == stats["internal_node_tokens"] + stats["item_leaf_nodes"]
    assert stats["cluster_nodes"] == stats["split_nodes"] + stats["terminal_clusters"]
    assert sum(level["cluster_count"] for level in stats["levels"]) == len(tree.nodes)
    assert stats["max_depth"] == max(map(len, index.values()))
    assert stats["mean_depth"] == sum(map(len, index.values())) / len(index)
    assert stats["sid_length_distribution"] == stats["index_statistics"]["length_histogram"]
