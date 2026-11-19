"""Published-catalog integrity and explicit embedding-row mapping contracts."""

import hashlib
import json

import numpy as np
import pytest

from sid_generation.hierarchical_cli import main
from sid_generation.hierarchical_kmeans import HierarchicalKMeans


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def catalog(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    values = np.array([[8, 8], [-2, -2], [0, 0], [1, 1], [9, 9]], dtype=np.float32)
    # Deliberately unrelated to row number, numeric order, and string order.
    ids = ["40", "0", "21", "3", "7"]
    np.save(inputs / "vectors.npy", values, allow_pickle=False)
    write_json(inputs / "ids.json", ids)
    write_json(inputs / "Demo.item.json", {item: {"title": f"Item {item}"} for item in reversed(ids)})
    for split in ("train", "valid", "test"):
        (inputs / f"Demo.{split}.inter").write_text(
            "user_id:token\titem_id_list:token_seq\titem_id:token\n5\t40 0\t21\n7\t3 7\t0\n",
            encoding="utf-8",
        )
    original = {path.name: path.read_bytes() for path in inputs.iterdir()}
    output = tmp_path / "catalog"
    argv = ["fit", "--data-path", str(inputs / "vectors.npy"), "--item-ids", str(inputs / "ids.json"),
            "--dataset-name", "Demo", "--dataset-dir", str(inputs), "--output-dir", str(output),
            "--branch-factor", "2", "--n-init", "1", "--max-iter", "20"]
    main(argv)
    return {"inputs": inputs, "output": output, "ids": ids, "original": original, "argv": argv}


def diagnose(catalog):
    output = catalog["output"]
    return main(["diagnose", "--model-path", str(output / "Demo.tree.json"),
                 "--manifest-path", str(output / "Demo.manifest.json")])


def test_manifest_hashes_cover_all_exported_files(catalog):
    output = catalog["output"]
    manifest = read_json(output / "Demo.manifest.json")
    actual = {path.name for path in output.iterdir()}
    assert set(manifest["outputs"]) == actual
    assert set(manifest["artifact_sha256"]) == actual - {"Demo.manifest.json"}
    for name, expected in manifest["artifact_sha256"].items():
        assert checksum(output / name) == expected
    assert manifest["input"]["embedding_sha256"] == checksum(catalog["inputs"] / "vectors.npy")
    assert manifest["item_id_mapping"]["source_sha256"] == checksum(catalog["inputs"] / "ids.json")
    assert set(manifest["versions"]) >= {"python", "numpy", "scikit-learn", "k-means-constrained"}
    assert diagnose(catalog)["artifact_hashes_verified"] == len(actual) - 1


def test_row_order_mapping_is_distinct_from_canonical_leaf_assignment(catalog):
    output = catalog["output"]
    index = read_json(output / "Demo.index.json")
    tree = read_json(output / "Demo.tree.json")
    assert read_json(output / "Demo.item_ids.json") == catalog["ids"]
    assert tree["item_ids"] == sorted(catalog["ids"])
    assert index["0"][-1] == "<leaf_0>"
    assert index["40"][-1] == "<leaf_3>"
    assert {item: tokens[-1] for item, tokens in index.items()} == {
        item: f"<leaf_{rank}>" for rank, item in enumerate(sorted(catalog["ids"]))}
    assert all(tokens[-1].startswith("<leaf_") for tokens in index.values())


def test_full_serialized_topology_reconstructs_exact_variable_length_index(catalog):
    output = catalog["output"]
    tree = read_json(output / "Demo.tree.json")
    index = read_json(output / "Demo.index.json")
    nodes = {node["id"]: node for node in tree["nodes"]}
    leaves = {item: f"<leaf_{rank}>" for rank, item in enumerate(tree["item_ids"])}
    rebuilt = {}
    for node in nodes.values():
        if node["children"]:
            continue
        tokens = []
        current = node
        while True:
            if current["token"] is not None:
                tokens.append(current["token"])
            if current["parent"] is None:
                break
            current = nodes[current["parent"]]
        for item in node["items"]:
            assert item not in rebuilt
            rebuilt[item] = list(reversed(tokens)) + [leaves[item]]
    assert rebuilt == index
    assert len({len(tokens) for tokens in index.values()}) > 1
    paths = {tuple(tokens) for tokens in index.values()}
    assert len(paths) == len(catalog["ids"])
    assert not any(tuple(tokens[:position]) in paths
                   for tokens in index.values() for position in range(1, len(tokens)))


def test_original_inputs_and_dataset_companions_remain_byte_identical(catalog):
    for name, original in catalog["original"].items():
        assert (catalog["inputs"] / name).read_bytes() == original
    manifest = read_json(catalog["output"] / "Demo.manifest.json")
    for copied in manifest["copied_dataset_inputs"]:
        name = copied["file"]
        assert (catalog["output"] / name).read_bytes() == catalog["original"][name]
        assert copied["sha256"] == checksum(catalog["inputs"] / name)


def test_existing_output_is_refused_before_any_fit_or_modification(catalog, monkeypatch):
    output = catalog["output"]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("fit must not run when output already exists")
    monkeypatch.setattr(HierarchicalKMeans, "fit", unexpected_fit)
    with pytest.raises(SystemExit) as error:
        main(catalog["argv"])
    assert error.value.code == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("name", ["Demo.index.json", "Demo.item_ids.json", "Demo.tree_stats.json", "Demo.train.inter"])
def test_corrupted_published_artifact_is_rejected(catalog, name):
    path = catalog["output"] / name
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


def test_index_disagreement_rejected_even_with_matching_recomputed_hash(catalog):
    output = catalog["output"]
    path = output / "Demo.index.json"
    index = read_json(path)
    index["0"] = ["<leaf_999>"]
    write_json(path, index)
    manifest_path = output / "Demo.manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifact_sha256"][path.name] = checksum(path)
    write_json(manifest_path, manifest)
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


def test_manifest_cannot_hide_an_unverified_mapping_file(catalog):
    output = catalog["output"]
    manifest_path = output / "Demo.manifest.json"
    manifest = read_json(manifest_path)
    del manifest["artifact_sha256"]["Demo.item_ids.json"]
    write_json(manifest_path, manifest)
    write_json(output / "Demo.item_ids.json", ["wrong"])
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["runtime", "topology", "duplicate-key", "invalid-timing"])
def test_forged_statistics_rejected_even_with_matching_hash(catalog, change):
    output = catalog["output"]
    path = output / "Demo.tree_stats.json"
    manifest_path = output / "Demo.manifest.json"
    stats = read_json(path)
    manifest = read_json(manifest_path)
    if change == "runtime":
        manifest["runtime"]["wall_seconds"] += 1
    elif change == "topology":
        stats["items"] += 1
        manifest["tree_statistics"]["items"] = stats["items"]
    elif change == "invalid-timing":
        stats["runtime"]["fit_seconds"] = -1
        manifest["runtime"] = stats["runtime"]
    content = json.dumps(stats)
    if change == "duplicate-key":
        content = '{"items":0,' + content[1:]
    path.write_text(content, encoding="utf-8")
    manifest["artifact_sha256"][path.name] = checksum(path)
    write_json(manifest_path, manifest)
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


def test_wrong_mapping_rejected_even_with_matching_recomputed_hash(catalog):
    output = catalog["output"]
    mapping_path = output / "Demo.item_ids.json"
    wrong_ids = catalog["ids"].copy()
    wrong_ids[0] = "99"
    write_json(mapping_path, wrong_ids)
    manifest_path = output / "Demo.manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifact_sha256"][mapping_path.name] = checksum(mapping_path)
    write_json(manifest_path, manifest)
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


def test_manifest_rejects_parent_path_artifact(catalog):
    output = catalog["output"]
    manifest_path = output / "Demo.manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifact_sha256"]["../outside.json"] = "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2


@pytest.mark.parametrize("payload", [[], None, "not an object"])
def test_malformed_manifest_has_clean_cli_error(catalog, payload):
    write_json(catalog["output"] / "Demo.manifest.json", payload)
    with pytest.raises(SystemExit) as error:
        diagnose(catalog)
    assert error.value.code == 2
