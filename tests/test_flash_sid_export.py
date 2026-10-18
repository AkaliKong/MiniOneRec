import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rq.flash_sid.cli import main
from rq.flash_sid.export import build_minionerec_index, collision_metrics, load_item_ids


class ExportTests(unittest.TestCase):
    def test_duplicate_embeddings_have_unique_fixed_length_indices(self):
        codes = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 4]])
        result = build_minionerec_index(codes, ["b", "a", "c"])
        self.assertEqual(result["a"], ["<a_1>", "<b_2>", "<c_3>", "<d_0>"])
        self.assertEqual(result["b"][-1], "<d_1>")
        self.assertEqual(len({tuple(v) for v in result.values()}), 3)
        permuted = build_minionerec_index(codes[[2, 1, 0]], ["c", "a", "b"])
        self.assertEqual(result, permuted)

    def test_metrics_distinguish_excess_items_and_involved_items(self):
        stats = collision_metrics(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]]), 2)
        self.assertEqual(stats["excess_collision_items"], 2)
        self.assertEqual(stats["items_in_collision_groups"], 3)
        self.assertEqual(stats["raw_collision_rate"], 0.5)
        self.assertEqual(stats["max_collision_group_size"], 3)

    def test_collision_capacity_and_invalid_ids_fail(self):
        codes = np.ones((3, 3), dtype=np.int64)
        with self.assertRaises(ValueError):
            build_minionerec_index(codes, [0, 1, 2], suffix_size=2)
        for ids in [[0, "0", 1], [0, 1], [True, 1, 2], [0, " ", 2]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_minionerec_index(codes, ids)

    def test_explicit_id_order_and_metadata_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / "ids.json"
            path.write_text('[42, 7, 99]', encoding="utf-8")
            self.assertEqual(load_item_ids(3, item_ids_path=path), ["42", "7", "99"])
            path.write_text('{"2": {}, "0": {}, "1": {}}', encoding="utf-8")
            self.assertEqual(load_item_ids(3, item_meta_path=path), ["0", "1", "2"])
            path.write_text('{"1": {}, "2": {}, "3": {}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_item_ids(3, item_meta_path=path)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "x.npy"
        x = np.random.default_rng(1).normal(size=(8, 12)).astype(np.float32)
        x[1] = x[0]
        np.save(self.input, x)
        self.base = ["--embeddings", str(self.input), "--dataset", "Test"]

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(self.base + list(args))

    def test_generate_reload_and_no_accidental_overwrite(self):
        first, second = self.root / "first", self.root / "second"
        self.assertEqual(self.invoke("--output-dir", str(first), "--batch-size", "3"), 0)
        index = json.loads((first / "Test.index.json").read_text())
        self.assertEqual(set(index), set(map(str, range(8))))
        self.assertNotEqual(index["0"], index["1"])
        self.assertEqual(index["0"][:3], index["1"][:3])
        self.assertEqual(self.invoke("--output-dir", str(second), "--model-in", str(first / "Test.simhash.npz")), 0)
        np.testing.assert_array_equal(np.load(first / "Test.codes.npy"), np.load(second / "Test.codes.npy"))
        original = (first / "Test.index.json").read_bytes()
        self.assertEqual(self.invoke("--output-dir", str(first)), 2)
        self.assertEqual((first / "Test.index.json").read_bytes(), original)
        self.assertEqual(self.invoke("--output-dir", str(first), "--overwrite"), 0)

    def test_raw_defaults_to_32_codes_without_index(self):
        out = self.root / "raw"
        self.assertEqual(self.invoke("--output-dir", str(out), "--format", "raw"), 0)
        self.assertEqual(np.load(out / "Test.codes.npy").shape, (8, 32))
        self.assertFalse((out / "Test.index.json").exists())
        stats = json.loads((out / "Test.metrics.json").read_text())
        self.assertGreater(stats["raw_collision_rate"], 0)

    def test_failures_before_writing_outputs(self):
        for extra in [["--num-codebooks", "32"], ["--codebook-size", "7"],
                      ["--suffix-size", "1"], ["--batch-size", "0"], ["--dataset", "../escape"]]:
            out = self.root / "fail"
            with self.subTest(extra=extra):
                self.assertEqual(self.invoke("--output-dir", str(out), *extra), 2)
                self.assertFalse(out.exists())

    def test_saved_configuration_conflicts_and_stale_index(self):
        out = self.root / "saved"
        self.assertEqual(self.invoke("--output-dir", str(out)), 0)
        self.assertEqual(self.invoke("--output-dir", str(self.root / "bad"), "--model-in",
                                     str(out / "Test.simhash.npz"), "--seed", "1"), 2)
        self.assertEqual(self.invoke("--output-dir", str(out), "--format", "raw", "--overwrite"), 2)

    def test_input_cannot_be_overwritten_by_output(self):
        same = self.root / "Test.codes.npy"
        same.write_bytes(self.input.read_bytes())
        original = same.read_bytes()
        self.assertEqual(self.invoke("--embeddings", str(same), "--output-dir", str(self.root), "--overwrite"), 2)
        self.assertEqual(same.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
