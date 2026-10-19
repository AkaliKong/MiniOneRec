"""CPU correctness and CLI integration tests; no language-model weights required."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from prq_kmeans import PRQKMeans
from rq_opq import RQOPQ, unpack_pq_codes
from sid_cli import generate
from sid_common import build_indexes, canonical_item_ids, normalize_rows, projection_residual


class ProjectionTests(unittest.TestCase):
    def test_projection_is_orthogonal_and_not_centroid_subtraction(self):
        r = np.array([[3, 4, 0], [-2, 7, 1]], dtype=np.float32)
        c = np.array([[2, 0, 0], [2, 0, 0]], dtype=np.float32)
        projected = projection_residual(r, c)
        np.testing.assert_allclose(np.sum(projected * c, axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1, atol=1e-6)
        self.assertFalse(np.allclose(projected, normalize_rows(r - c)))

    def test_mean_direction_removal_is_not_mean_centering(self):
        x = normalize_rows(np.array([[2., 1.], [4., -1.], [1., 3.]]))
        mean = x.mean(axis=0)
        r = projection_residual(x, mean)
        np.testing.assert_allclose(r @ mean, 0, atol=1e-6)
        self.assertFalse(np.allclose(r, normalize_rows(x - mean)))

    def test_degenerate_directions_are_finite(self):
        data = np.array([[0., 0.], [1., 0.]])
        np.testing.assert_array_equal(projection_residual(data, np.zeros(2)), data)
        np.testing.assert_array_equal(projection_residual(data, np.array([1., 0.])), np.zeros_like(data))


class PRQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.default_rng(17).normal(size=(160, 12)).astype(np.float32)
        cls.options = dict(codebook_sizes=(8, 8, 8), top_k=2, n_iter=5, batch_size=37, seed=3)
        cls.model = PRQKMeans(**cls.options).fit(cls.data)

    def test_range_and_row_order(self):
        codes = self.model.encode(self.data)
        self.assertEqual(codes.shape, (160, 3))
        self.assertTrue(np.all((0 <= codes) & (codes < 8)))
        order = np.random.default_rng(2).permutation(len(codes))
        np.testing.assert_array_equal(self.model.encode(self.data[order]), codes[order])

    def test_chunked_encoding(self):
        expected = self.model.encode(self.data)
        actual = np.concatenate([self.model.encode(self.data[:19]), self.model.encode(self.data[19:71]), self.model.encode(self.data[71:])])
        np.testing.assert_array_equal(expected, actual)

    def test_refit_same_seed(self):
        other = PRQKMeans(**self.options).fit(self.data)
        np.testing.assert_array_equal(self.model.encode(self.data), other.encode(self.data))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prq.npz"
            self.model.save(path)
            loaded = PRQKMeans.load(path)
            np.testing.assert_array_equal(self.model.encode(self.data), loaded.encode(self.data))
            np.testing.assert_array_equal(self.model.global_mean, loaded.global_mean)
            with self.assertRaises(FileExistsError):
                self.model.save(path)

    def test_identical_vectors_and_zero_mean(self):
        for data in (np.ones((8, 4)), np.zeros((8, 4)), np.tile([[1., 0.], [-1., 0.]], (4, 1))):
            model = PRQKMeans((2, 2, 2), n_iter=3).fit(data)
            self.assertTrue(all(np.isfinite(cb).all() for cb in model.codebooks))
            codes = model.encode(data)
            self.assertTrue(np.all((0 <= codes) & (codes < 2)))

    def test_invalid_data_and_configuration(self):
        for data in (np.array([[np.nan, 0]]), np.array([[np.inf, 0]]), np.empty((0, 4)), np.ones(3), np.ones((2, 3), dtype=complex)):
            with self.assertRaises(ValueError):
                PRQKMeans((2,), top_k=1).fit(data)
        for options in (dict(top_k=257), dict(beta=0), dict(n_iter=0), dict(codebook_sizes=()), dict(eps=0), dict(batch_size=0)):
            with self.assertRaises(ValueError):
                PRQKMeans(**options)
        with self.assertRaises(ValueError):
            self.model.encode(np.ones((8, 13)))
        with self.assertRaises(ValueError):
            PRQKMeans().encode(self.data)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["90", "10", "30", "20"]
        self.codes = np.array([[0, 1], [0, 1], [0, 1], [1, 0]])

    def test_suffix_is_unique_order_invariant_and_keeps_semantic_prefix(self):
        raw, final, stats = build_indexes(self.ids, self.codes, [2, 2])
        self.assertEqual(stats["raw"]["excess_collision_count"], 2)
        self.assertEqual(stats["raw"]["affected_item_count"], 3)
        self.assertEqual(stats["final"]["unique_sids"], 4)
        self.assertEqual(final["10"], ["<a_0>", "<b_1>", "<c_0>"])
        for key in self.ids:
            self.assertEqual(final[key][:-1], raw[key])
        order = [3, 1, 0, 2]
        _, shuffled, _ = build_indexes([self.ids[i] for i in order], self.codes[order], [2, 2])
        self.assertEqual(final, shuffled)

    def test_no_extra_token_without_collisions(self):
        raw, final, stats = build_indexes([3, 9], np.array([[0, 1], [1, 0]]), [2, 2])
        self.assertEqual(raw, final)
        self.assertFalse(stats["identity_suffix_appended"])

    def test_error_and_allow_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "collisions"):
            build_indexes(self.ids, self.codes, [2, 2], "error")
        raw, final, stats = build_indexes(self.ids, self.codes, [2, 2], "allow")
        self.assertEqual(raw, final)
        self.assertGreater(stats["final"]["excess_collision_rate"], 0)

    def test_ids_and_code_bounds(self):
        for values in ([1, "1"], ["a", " a"], ["a", "b\n"], [True, 2], [1.0, 2]):
            with self.assertRaises(ValueError):
                canonical_item_ids(values, 2)
        with self.assertRaises(ValueError):
            build_indexes([1, 2], np.array([[0, 2], [1, 0]]), [2, 2])
        with self.assertRaises(ValueError):
            build_indexes([1, 2], np.array([[0., 1.], [1., 0.]]), [2, 2])


try:
    import faiss
except ImportError:
    faiss = None


@unittest.skipIf(faiss is None, "optional FAISS dependency is not installed")
class RQOPQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.default_rng(23).normal(size=(512, 12)).astype(np.float32)
        cls.options = dict(rq_levels=1, codebook_size=8, opq_subspaces=2, nbits=3,
                           rq_niter=4, opq_niter=3, pq_niter=4, seed=8, batch_size=67)
        cls.model = RQOPQ(**cls.options).fit(cls.data)

    def test_residual_is_prefix_reconstruction_error(self):
        codes, residuals = self.model.rq_residuals(self.data)
        reconstructed = sum(cb[codes[:, p]] for p, cb in enumerate(self.model.rq_codebooks))
        np.testing.assert_allclose(residuals, self.data - reconstructed, atol=1e-6)

    def test_codes_keep_row_order_and_chunk_boundaries(self):
        codes = self.model.encode(self.data)
        self.assertEqual(codes.shape, (512, 3))
        self.assertTrue(np.all((0 <= codes) & (codes < 8)))
        order = np.random.default_rng(4).permutation(len(codes))
        np.testing.assert_array_equal(self.model.encode(self.data[order]), codes[order])
        chunked = np.concatenate([self.model.encode(self.data[:17]), self.model.encode(self.data[17:])])
        np.testing.assert_array_equal(codes, chunked)

    def test_checkpoint_round_trip_and_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rq_opq.npz"
            self.model.save(path)
            loaded = RQOPQ.load(path)
            np.testing.assert_array_equal(self.model.encode(self.data), loaded.encode(self.data))
            with self.assertRaises(ValueError):
                PRQKMeans.load(path)
        other = RQOPQ(**self.options).fit(self.data)
        np.testing.assert_array_equal(self.model.encode(self.data), other.encode(self.data))

    def test_five_token_pipeline_and_thread_restoration(self):
        before = faiss.omp_get_max_threads()
        model = RQOPQ(rq_levels=3, codebook_size=4, opq_subspaces=2, nbits=2,
                      rq_niter=3, opq_niter=2, pq_niter=3).fit(self.data)
        self.assertEqual(model.encode(self.data).shape, (512, 5))
        codes, residuals = model.rq_residuals(self.data)
        expected = self.data.copy()
        for p, cb in enumerate(model.rq_codebooks):
            expected -= cb[codes[:, p]]
        np.testing.assert_array_equal(expected, residuals)
        self.assertEqual(faiss.omp_get_max_threads(), before)

    def test_rotation_is_trained_and_orthogonal(self):
        rotation = faiss.downcast_VectorTransform(self.model.opq_index.chain.at(0))
        a = faiss.vector_to_array(rotation.A).reshape(12, 12)
        np.testing.assert_allclose(a @ a.T, np.eye(12), atol=2e-5)
        self.assertTrue(rotation.is_trained)

    def test_invalid_dimensions_samples_and_bits(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            RQOPQ(opq_subspaces=5).fit(self.data)
        with self.assertRaisesRegex(ValueError, "at least"):
            RQOPQ().fit(self.data[:100])
        with self.assertRaises(ValueError):
            RQOPQ(nbits=9)

    def test_zero_residual_catalog_can_be_disambiguated(self):
        data = np.zeros((160, 4), dtype=np.float32)
        model = RQOPQ(rq_levels=1, codebook_size=4, opq_subspaces=2, nbits=2,
                      rq_niter=2, opq_niter=2, pq_niter=2).fit(data)
        codes = model.encode(data)
        self.assertTrue(np.all((0 <= codes) & (codes < 4)))
        _, _, stats = build_indexes(list(range(len(data))), codes, model.capacities)
        self.assertEqual(stats["final"]["unique_sids"], len(data))


class PackingTests(unittest.TestCase):
    def test_cross_byte_pq_unpacking(self):
        # 5-bit fields [17, 3, 31], hand-packed as 17 + (3<<5) + (31<<10).
        codes = unpack_pq_codes(np.array([[0x71, 0x7c]], dtype=np.uint8), 3, 5)
        np.testing.assert_array_equal(codes, [[17, 3, 31]])

    def test_eight_bit_pq_unpacking(self):
        data = np.array([[3, 255], [90, 11]], dtype=np.uint8)
        np.testing.assert_array_equal(unpack_pq_codes(data, 2, 8), data)


class CLITests(unittest.TestCase):
    def test_fit_encode_metadata_and_real_upstream_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input"
            source.mkdir()
            x = np.random.default_rng(7).normal(size=(32, 6)).astype(np.float32)
            x[1] = x[0]  # Forces a collision, exercising actual unique export.
            np.save(source / "emb.npy", x)
            ids = [str(i * 3 + 10) for i in range(len(x))]
            (source / "ids.json").write_text(json.dumps(ids))
            (source / "Demo.item.json").write_text(json.dumps({i: {"title": f"Item {i}"} for i in ids}))
            for split in ("train", "valid", "test"):
                (source / f"Demo.{split}.inter").write_text(f"user\thistory\ttarget\n0\t{ids[0]} {ids[1]}\t{ids[2]}\n")
            command = [sys.executable, str(CODE / "prq_kmeans.py"), "fit", "--data-path", str(source / "emb.npy"),
                       "--dataset-name", "Demo", "--output-dir", str(tmp / "fit"), "--item-ids", str(source / "ids.json"),
                       "--dataset-dir", str(source), "--codebook-sizes", "4", "4", "4", "--n-iter", "3"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            final = json.loads((tmp / "fit/Demo.index.json").read_text())
            self.assertEqual(set(final), set(ids))
            self.assertEqual(len({tuple(v) for v in final.values()}), len(ids))
            model_path = tmp / "fit/Demo.tokenizer.npz"
            encoded = subprocess.run([sys.executable, str(CODE / "prq_kmeans.py"), "encode", "--data-path", str(source / "emb.npy"),
                                      "--dataset-name", "Demo", "--output-dir", str(tmp / "encoded"), "--item-ids", str(source / "ids.json"),
                                      "--model-path", str(model_path)], capture_output=True, text=True)
            self.assertEqual(encoded.returncode, 0, encoded.stderr)
            self.assertEqual(final, json.loads((tmp / "encoded/Demo.index.json").read_text()))
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)

            upstream = CODE.parent
            if not (upstream / "convert_dataset.py").exists() or importlib.util.find_spec("pandas") is None:
                self.skipTest("CLI checks completed; repository conversion additionally needs pandas")
            # Execute the actual TokenExtender class without importing torch or
            # transformers from sft.py; this checks its real filename behavior.
            tree = ast.parse((upstream / "sft.py").read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TokenExtender")
            namespace = {"os": __import__("os"), "json": json}
            exec(compile(ast.Module(body=[cls], type_ignores=[]), "sft.py:TokenExtender", "exec"), namespace)
            extender = namespace["TokenExtender"](str(tmp / "fit"), "Demo")
            self.assertEqual(set(extender.get_new_tokens()), {token for tokens in final.values() for token in tokens})
            converted = subprocess.run([sys.executable, str(upstream / "convert_dataset.py"), "--data_dir", str(tmp / "fit"),
                                        "--dataset_name", "Demo", "--output_dir", str(tmp / "converted")], capture_output=True, text=True)
            self.assertEqual(converted.returncode, 0, converted.stderr)
            import pandas as pd
            frame = pd.read_csv(tmp / "converted/train/Demo_5_2016-10-2018-11.csv")
            self.assertEqual(frame.loc[0, "item_sid"], "".join(final[ids[2]]))
            self.assertEqual(len(frame), 1)

    def test_strict_collision_failure_writes_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_path = tmp / "x.npy"
            np.save(data_path, np.ones((8, 4), dtype=np.float32))
            args = argparse.Namespace(operation="fit", data_path=str(data_path), dataset_name="Demo", output_dir=str(tmp / "out"),
                                      item_ids=None, dataset_dir=None, collision_policy="error", overwrite=False,
                                      codebook_sizes=[2, 2], top_k=2, beta=15., n_iter=2, batch_size=8, seed=42)
            with self.assertRaisesRegex(ValueError, "collisions"):
                generate(args, "prq_kmeans", PRQKMeans)
            self.assertFalse((tmp / "out").exists())

    def test_unknown_interaction_id_fails_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "x.npy", np.ones((4, 2), dtype=np.float32))
            (tmp / "Demo.item.json").write_text(json.dumps({str(i): {} for i in range(4)}))
            (tmp / "Demo.train.inter").write_text("user\thistory\ttarget\n0\t0 1\t99\n")
            args = argparse.Namespace(operation="fit", data_path=str(tmp / "x.npy"), dataset_name="Demo", output_dir=str(tmp / "out"),
                                      item_ids=None, dataset_dir=str(tmp), collision_policy="append", overwrite=False)
            with self.assertRaisesRegex(ValueError, "unknown item"):
                generate(args, "prq_kmeans", PRQKMeans)


if __name__ == "__main__":
    unittest.main()
