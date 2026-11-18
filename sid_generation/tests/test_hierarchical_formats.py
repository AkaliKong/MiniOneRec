"""Current-format publications through the actual fit/encode/diagnose CLI."""

import hashlib
import json

import numpy as np
import pytest

from sid_generation.hierarchical_cli import main
from sid_generation.hierarchical_tree import HierarchicalTree


IDS = ["42", "0", "c", "7", "15", "x", "2"]
VALUES = np.array([[10, 1], [0, 0], [2, 2], [-3, 1], [11, 0], [-4, 0], [1, 3]], dtype=np.float32)


def publish(tmp, directory, operation, *options, reverse=False):
    np.save(tmp / "vectors.npy", VALUES[::-1] if reverse else VALUES)
    (tmp / "ids.json").write_text(json.dumps(IDS[::-1] if reverse else IDS), encoding="utf-8")
    path = tmp / directory
    main([operation, "--dataset-name", "Catalog", "--data-path", str(tmp / "vectors.npy"),
          "--item-ids", str(tmp / "ids.json"), "--output-dir", str(path), *options])
    return path


def read(directory, suffix):
    return json.loads((directory / ("Catalog." + suffix + ".json")).read_text(encoding="utf-8"))


def diagnose(directory):
    return main(["diagnose", "--model-path", str(directory / "Catalog.tree.json"),
                 "--manifest-path", str(directory / "Catalog.manifest.json")])


@pytest.mark.parametrize("method,constrained,greedy", [
    ("balanced", 3, 0), ("greedy", 0, 3), ("hybrid", 2, 1),
])
def test_v2_cli_roundtrip_reports_only_executed_backends(tmp_path, method, constrained, greedy):
    options = ["--branch-factor", "2", "--method", method, "--greedy-threshold", "4",
               "--greedy-n-init", "2", "--greedy-max-iter", "13"]
    first = publish(tmp_path, "fit", "fit", *options)
    other = publish(tmp_path, "repeat", "fit", *options, reverse=True)
    path = first / "Catalog.tree.json"
    assert path.read_bytes() == (other / path.name).read_bytes()
    report = diagnose(first)
    assert report["split_counts"] == {"constrained": constrained, "greedy": greedy}
    assert report["source_format_version"] == 2 and report["method"] == method
    manifest = read(first, "manifest")
    expected = {backend: count for backend, count in [
        ("k_means_constrained.KMeansConstrained", constrained), ("sklearn.cluster.KMeans", greedy)] if count}
    assert manifest["fit_backends"] == expected
    assert report["artifact_hashes_verified"] == 4
    assert HierarchicalTree.load(path).encode(VALUES, IDS) == read(first, "index")
    runtime = manifest["runtime"]
    assert runtime == read(first, "tree_stats")["runtime"]
    assert set(runtime) == {"wall_seconds", "fit_seconds"}
    assert runtime["wall_seconds"] >= runtime["fit_seconds"] >= 0
    assert manifest["configuration"]["greedy_n_init"] == 2
    assert manifest["configuration"]["greedy_max_iter"] == 13
    assert manifest["configuration"]["greedy_threshold"] == 4
    assert manifest["configuration"]["method"] == method
    encoded = publish(tmp_path, "encode", "encode", "--model-path", str(path), reverse=True)
    assert diagnose(encoded)["split_counts"] == report["split_counts"]
    assert read(encoded, "manifest")["fit_backends"] == {}
    assert read(encoded, "manifest")["runtime"]["fit_seconds"] == 0
    assert read(encoded, "item_ids") == IDS[::-1]
    assert read(encoded, "manifest")["source_model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (encoded / "Catalog.index.json").read_bytes() == (first / "Catalog.index.json").read_bytes()


@pytest.mark.parametrize("corruption", [
    "extra-config-key", "bad-version", "bool-format", "wrong-algorithm", "unsupported-format",
    "wrong-config", "missing-hash", "bad-tree-hash", "duplicate-key",
])
def test_corrupt_manifest_is_rejected_before_reexport(tmp_path, corruption):
    fit = publish(tmp_path, "fit", "fit", "--branch-factor", "2")
    manifest_path = fit / "Catalog.manifest.json"
    m = read(fit, "manifest")
    if corruption == "extra-config-key":
        m["configuration"]["extra"] = 0
    elif corruption == "bad-version":
        m["algorithm_version"] = "9.0.0"
    elif corruption == "bool-format":
        m["format_version"] = True
    elif corruption == "wrong-algorithm":
        m["algorithm"] = "other"
    elif corruption == "unsupported-format":
        m["format_version"] = 1
    elif corruption == "wrong-config":
        m["configuration"]["n_init"] += 1
    elif corruption == "missing-hash":
        del m["artifact_sha256"]["Catalog.tree_stats.json"]
    elif corruption == "bad-tree-hash":
        m["artifact_sha256"]["Catalog.tree.json"] = "0" * 64
    data = json.dumps(m)
    if corruption == "duplicate-key":
        data = '{"algorithm":"forged",' + data[1:]
    manifest_path.write_text(data, encoding="utf-8")
    with pytest.raises(SystemExit):
        diagnose(fit)
    with pytest.raises(SystemExit):
        publish(tmp_path, "rejected", "encode", "--model-path", str(fit / "Catalog.tree.json"))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("target,field", [
    ("tree", "method"), ("manifest", "method"), ("manifest", "format_version"),
    ("manifest", "fit_backends"), ("manifest", "backend_options"),
])
def test_v2_requires_explicit_strategy_and_consistent_provenance(tmp_path, target, field):
    path = publish(tmp_path, "new", "fit", "--branch-factor", "2", "--method", "greedy")
    payload = read(path, target)
    if field == "method":
        del payload["config" if target == "tree" else "configuration"][field]
    elif field == "format_version":
        del payload[field]
    else:
        payload[field] = {}
    file = path / f"Catalog.{target}.json"
    file.write_text(json.dumps(payload), encoding="utf-8")
    if target == "tree":
        with pytest.raises(ValueError):
            HierarchicalTree.load(file)
    with pytest.raises(SystemExit):
        diagnose(path)
