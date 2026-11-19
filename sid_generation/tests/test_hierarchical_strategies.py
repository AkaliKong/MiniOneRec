"""Capacity assignment, per-node dispatch and determinism checks."""

from unittest.mock import patch

import numpy as np
import pytest
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from sid_generation.hierarchical_greedy import greedy_labels
from sid_generation.hierarchical_kmeans import HierarchicalKMeans


def tuple_oracle(points, centers):
    k = len(centers)
    n = len(points)
    candidates = sorted((sum((float(a) - float(b)) ** 2 for a, b in zip(point, center)), i, j)
                        for i, point in enumerate(points) for j, center in enumerate(centers))
    assignments = {}
    members = [[] for _ in centers]
    for _, i, j in candidates:
        if i not in assignments and len(members[j]) < n // k + (j < n % k):
            members[j].append(i)
            assignments[i] = j
    assert sorted(assignments) == list(range(n))
    return [assignments[i] for i in range(n)]


def small_fit(method, n, threshold=7, ids=None, values=None, **kwargs):
    if values is None:
        values = np.random.default_rng(13).normal(size=(n, 4)).astype(np.float32)
    if ids is None:
        ids = [f"item-{17*i}" for i in range(n)]
    return HierarchicalKMeans(branch_factor=2, method=method, greedy_threshold=threshold,
                              n_init=2, max_iter=12, greedy_n_init=2, greedy_max_iter=9,
                              **kwargs).fit(values, ids)


def test_global_priority_changes_the_first_items_assignment():
    points = np.array([[4], [0]], dtype=np.float32)
    centers = np.array([[0], [10]], dtype=np.float32)
    labels = greedy_labels(points, centers)
    assert labels.tolist() == [1, 0]
    assert ((points - centers[labels]) ** 2).sum() == 36


@pytest.mark.parametrize("field,value", [
    ("method", "speedup"), ("method", None), ("greedy_threshold", -1),
    ("greedy_threshold", True), ("greedy_n_init", 0), ("greedy_max_iter", 1.5),
])
def test_rejects_invalid_strategy_configuration(field, value):
    with pytest.raises(ValueError, match=field):
        HierarchicalKMeans(**{field: value})


@pytest.mark.parametrize("n,k", [(2, 2), (7, 3), (17, 4), (24, 6)])
def test_stable_sort_matches_independent_tuple_oracle(n, k):
    rng = np.random.default_rng(24)
    values = rng.integers(-2, 3, size=(n, 4)).astype(np.float32)
    centers = rng.integers(-1, 2, size=(k, 4)).astype(np.float32)
    centers[-1] = centers[0]
    values[1] = values[0]
    result = greedy_labels(values, centers)
    assert result.tolist() == tuple_oracle(values, centers)
    assert np.bincount(result, minlength=k).tolist() == [n // k + (j < n % k) for j in range(k)]


def test_all_ties_assign_extra_slots_to_original_centroid_order():
    points = np.ones((8, 2), dtype=np.float32)
    centers = np.ones((3, 2), dtype=np.float32)
    assert greedy_labels(points, centers).tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def test_distance_priority_is_global_across_coordinate_blocks():
    # Places enough rows in the second block to fill center 0 ahead of
    # equidistant early rows. Per-block assignment gives a different answer.
    points = np.zeros((700, 1024), dtype=np.float32)
    points[:300, 0] = 4
    centers = np.zeros((2, 1024), dtype=np.float32)
    centers[1, 0] = 10
    result = greedy_labels(points, centers)
    assert result.tolist() == [1] * 300 + [0] * 350 + [1] * 50


@pytest.mark.parametrize("x,c", [(0, np.nan), (np.inf, 0), (np.finfo(np.float32).max, -1)])
def test_invalid_or_overflowing_distances_fail(x, c):
    with pytest.raises(ValueError, match="finite|overflow"):
        greedy_labels(np.array([[x]], dtype=np.float32), np.array([[c]], dtype=np.float32))


@pytest.mark.parametrize("n,expected", [(6, "constrained"), (7, "constrained"), (8, "greedy")])
def test_hybrid_boundary_at_each_node(n, expected):
    model = small_fit("hybrid", n, threshold=7)
    assert model.split_diagnostics[0]["split_method"] == expected
    assert all(s["split_method"] == ("greedy" if s["items"] > 7 else "constrained")
               for s in model.split_diagnostics)


def test_hybrid_siblings_use_their_own_size_and_call_actual_backends():
    # Root N=15 -> sizes 7 and 8. These siblings at the same depth need
    # different backends. Instrument the fitted estimators, not just metadata.
    greedy_fit = KMeans.fit
    constrained_fit = KMeansConstrained.fit_predict
    with (patch.object(KMeans, "fit", autospec=True, side_effect=greedy_fit) as greedy,
          patch.object(KMeansConstrained, "fit_predict", autospec=True, side_effect=constrained_fit) as constrained):
        model = small_fit("hybrid", 15, threshold=7, seed=31, tol=0.002)
    calls = [(s["items"], s["split_method"]) for s in model.split_diagnostics if s["depth"] == 1]
    assert sorted(calls) == [(7, "constrained"), (8, "greedy")]
    assert sorted(len(c.args[1]) for c in greedy.call_args_list) == [8, 15]
    assert len(constrained.call_args_list) > 0 and all(len(c.args[1]) <= 7 for c in constrained.call_args_list)
    for call in greedy.call_args_list:
        params = call.args[0].get_params()
        assert {key: params[key] for key in ["n_init", "max_iter", "random_state", "tol", "algorithm", "init"]} == {
            "n_init": 2, "max_iter": 9, "random_state": 31, "tol": 0.002, "algorithm": "lloyd", "init": "k-means++"}


@pytest.mark.parametrize("method", ["balanced", "greedy", "hybrid"])
def test_terminal_catalog_does_not_invoke_either_fit(method):
    with (patch.object(KMeans, "fit", side_effect=AssertionError("unexpected greedy fit")),
          patch.object(KMeansConstrained, "fit_predict", side_effect=AssertionError("unexpected constrained fit"))):
        model = small_fit(method, 2, threshold=0)
    assert model.split_diagnostics == []
    assert len(model.tree.nodes) == 1


@pytest.mark.parametrize("method", ["balanced", "greedy", "hybrid"])
def test_reordered_fixed_catalog_repeats_and_encodes_identically(method):
    rng = np.random.default_rng(5)
    values = rng.normal(size=(17, 4)).astype(np.float32)
    values[2:5] = values[0]
    ids = [str(i * 19) for i in range(len(values))]
    permutation = rng.permutation(len(ids))
    other_ids = [ids[i] for i in permutation]
    model = small_fit(method, len(ids), ids=ids, values=values)
    repeat = small_fit(method, len(ids), ids=ids, values=values)
    shuffled = small_fit(method, len(ids), ids=other_ids, values=values[permutation])
    assert model.tree.to_dict() == repeat.tree.to_dict() == shuffled.tree.to_dict()
    assert model.encode(values[permutation], other_ids) == model.tree.to_index()
    assert set(model.tree.to_index()) == set(ids)


@pytest.mark.parametrize("threshold,equivalent", [(0, "greedy"), (2, "greedy"), (28, "balanced")])
def test_hybrid_extremes_reproduce_pure_method(threshold, equivalent):
    mixed = small_fit("hybrid", 28, threshold)
    pure = small_fit(equivalent, 28)
    assert mixed.tree.nodes == pure.tree.nodes
    assert mixed.tree.to_index() == pure.tree.to_index()


@pytest.mark.parametrize("method", ["greedy", "hybrid"])
def test_identical_vectors_use_capacity_assignment(method):
    values = np.ones((17, 2), dtype=np.float32)
    with pytest.warns(ConvergenceWarning):
        model = small_fit(method, len(values), values=values)
    assert len(model.tree.to_index()) == len(values)
    assert model.tree.statistics()["sid_collisions"] == 0
    for record in model.split_diagnostics:
        assert max(record["child_sizes"]) - min(record["child_sizes"]) <= 1
