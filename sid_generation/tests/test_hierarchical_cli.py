"""Public CLI workflows and input failure behavior, using the real solver."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sid_generation.hierarchical_cli import main


@pytest.fixture
def catalog(tmp_path):
    x = np.random.default_rng(4).normal(size=(11, 4)).astype(np.float32)
    ids = [str(100 + 3 * i) for i in range(len(x))]
    np.save(tmp_path / "vectors.npy", x)
    (tmp_path / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return x, ids


def options(tmp_path, output="fit"):
    return ["--data-path", str(tmp_path / "vectors.npy"), "--item-ids", str(tmp_path / "ids.json"),
            "--dataset-name", "Demo", "--output-dir", str(tmp_path / output)]


def test_fit_encode_permuted_and_diagnose(tmp_path, catalog):
    x, ids = catalog
    main(["fit", *options(tmp_path), "--branch-factor", "2", "--n-init", "1"])
    model = tmp_path / "fit" / "Demo.tree.json"
    np.save(tmp_path / "vectors.npy", x[::-1])
    (tmp_path / "ids.json").write_text(json.dumps(ids[::-1]), encoding="utf-8")
    main(["encode", *options(tmp_path, "encoded"), "--model-path", str(model)])
    assert (tmp_path / "fit" / "Demo.index.json").read_bytes() == (tmp_path / "encoded" / "Demo.index.json").read_bytes()
    result = main(["diagnose", "--model-path", str(model), "--manifest-path", str(tmp_path / "fit" / "Demo.manifest.json")])
    assert result["items"] == 11
    assert result["artifact_hashes_verified"] == 4


@pytest.mark.parametrize("change", ["unknown", "values", "wrong_rows", "subset"])
def test_encode_rejects_changed_catalog(tmp_path, catalog, change):
    x, ids = catalog
    main(["fit", *options(tmp_path), "--branch-factor", "3", "--n-init", "1"])
    if change == "unknown":
        ids[0] = "unknown"
    elif change == "values":
        x[0, 0] += 1
    elif change == "wrong_rows":
        x = x[::-1]
    else:
        x, ids = x[:-1], ids[:-1]
    np.save(tmp_path / "vectors.npy", x)
    (tmp_path / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["encode", *options(tmp_path, "bad"), "--model-path", str(tmp_path / "fit" / "Demo.tree.json")])
    assert error.value.code == 2
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("dataset", ["../outside", "bad/name", "bad\\name", ".", "A.json", "C:bad"])
def test_rejects_unsafe_dataset_names(tmp_path, catalog, dataset):
    args = options(tmp_path)
    args[args.index("Demo")] = dataset
    with pytest.raises(SystemExit):
        main(["fit", *args])
    assert not (tmp_path / "fit").exists()


def test_requires_explicit_row_ids(tmp_path, catalog):
    with pytest.raises(SystemExit):
        main(["fit", "--data-path", str(tmp_path / "vectors.npy"), "--dataset-name", "Demo", "--output-dir", str(tmp_path / "fit")])


@pytest.mark.parametrize("mode", [[], ["fit"], ["encode"], ["diagnose"]])
def test_module_help(mode):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "sid_generation.hierarchical_cli", *mode, "--help"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_invalid_dimensions_and_nonfinite_cli(tmp_path, catalog):
    for x in (np.array([]), np.array([[float("nan")]]), np.empty((0, 3))):
        np.save(tmp_path / "vectors.npy", x)
        with pytest.raises(SystemExit):
            main(["fit", *options(tmp_path)])
    assert not (tmp_path / "fit").exists()


def test_no_pickle_input(tmp_path, catalog):
    np.save(tmp_path / "vectors.npy", np.array([[object()]], dtype=object))
    with pytest.raises(SystemExit):
        main(["fit", *options(tmp_path)])
    assert not (tmp_path / "fit").exists()
