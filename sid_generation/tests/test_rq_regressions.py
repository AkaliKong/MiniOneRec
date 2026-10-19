"""Real-FAISS regressions for stable RQ assignment and checkpoint structure."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from rq_opq import RQOPQ, _nearest_l2, _rotate_stably, unpack_pq_codes
from sid_common import load_npz, save_npz

try:
    import faiss
except ImportError:
    faiss = None


class StableDistanceTests(unittest.TestCase):
    def test_double_nearest_is_independent_of_workspace_and_keeps_first_tie(self):
        rng = np.random.default_rng(12)
        data = (rng.normal(size=(41, 5)) + 10000).astype(np.float32)
        centroids = (rng.normal(size=(83, 5)) + 10000).astype(np.float32)
        centroids[1] = centroids[0]
        data[0] = centroids[0]
        expected = np.sum((data.astype(np.float64)[:, None] -
                           centroids.astype(np.float64)[None]) ** 2, axis=2).argmin(axis=1)
        self.assertEqual(expected[0], 0)
        for workspace in (128, 4096, 16 * 1024 * 1024):
            with self.subTest(workspace=workspace):
                np.testing.assert_array_equal(_nearest_l2(data, centroids, workspace), expected)


@unittest.skipIf(faiss is None, "optional FAISS dependency is not installed")
class RQRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.default_rng(100).normal(size=(512, 12)).astype(np.float32)
        cls.options = dict(rq_levels=2, codebook_size=8, opq_subspaces=3, nbits=3,
                           rq_niter=3, opq_niter=2, pq_niter=3, seed=9)
        cls.model = RQOPQ(**cls.options).fit(cls.data)

    def test_large_offset_default_chunks_match_double_nearest(self):
        data = self.data + np.float32(10000)
        model = RQOPQ(**self.options).fit(data)
        before = model.configuration()
        full = model.encode(data)
        split = np.concatenate([model.encode(data[:17]), model.encode(data[17:])])
        np.testing.assert_array_equal(full, split)
        self.assertEqual(model.configuration(), before)
        order = np.random.default_rng(22).permutation(len(data))
        np.testing.assert_array_equal(model.encode(data[order]), full[order])
        residuals = data.copy()
        for level, centroids in enumerate(model.rq_codebooks):
            distances = np.sum((residuals.astype(np.float64)[:, None] -
                                centroids.astype(np.float64)[None]) ** 2, axis=2)
            expected = distances.argmin(axis=1)
            np.testing.assert_array_equal(full[:, level], expected)
            residuals -= centroids[expected]
        # Exercise both sides of FAISS's former 20-query switch and tails.
        for batch in (1, 19, 20, 21, 67):
            model.batch_size = batch
            with self.subTest(batch=batch):
                np.testing.assert_array_equal(model.encode(data), full)

    def test_centered_training_preserves_well_separated_clusters_after_translation(self):
        rng = np.random.default_rng(61)
        centers = rng.normal(size=(8, 12)).astype(np.float32) * 5
        data = np.repeat(centers, 64, axis=0) + rng.normal(scale=.03, size=(512, 12)).astype(np.float32)
        options = dict(self.options, rq_levels=1)
        original = RQOPQ(**options).fit(data)
        shifted_data = data + np.float32(10000)
        shifted = RQOPQ(**options).fit(shifted_data)
        np.testing.assert_array_equal(original.encode(data)[:, 0], shifted.encode(shifted_data)[:, 0])
        np.testing.assert_allclose(original.rq_codebooks[0] + 10000, shifted.rq_codebooks[0], atol=.002)

    def test_extreme_finite_inputs_fail_clearly_without_negative_codes(self):
        for value in (1e19, 1e20, np.finfo(np.float32).max):
            for count in (1, 32):
                with self.subTest(value=value, count=count):
                    with self.assertRaisesRegex(ValueError, "range.*rescale embeddings"):
                        self.model.encode(np.full((count, 12), value, dtype=np.float32))
        maximum = np.finfo(np.float32).max
        with self.assertRaisesRegex(ValueError, "RQ residuals.*range"):
            self.model._prefix(np.full((1, 12), maximum, dtype=np.float32),
                               [np.full((8, 12), -maximum, dtype=np.float32)])
        with self.assertRaisesRegex(ValueError, "OPQ training.*safe FAISS float32 range"):
            RQOPQ(**self.options).fit(self.data * np.float32(1e30))

    def test_actual_faiss_bits_one_through_eight_and_round_trip(self):
        for bits in range(1, 9):
            with self.subTest(bits=bits):
                model = RQOPQ(**dict(self.options, nbits=bits)).fit(self.data)
                _, residuals = model.rq_residuals(self.data)
                rotation = faiss.downcast_VectorTransform(model.opq_index.chain.at(0))
                matrix = faiss.vector_to_array(rotation.A).reshape(12, 12)
                rotated = _rotate_stably(residuals, matrix)
                child = faiss.downcast_index(model.opq_index.index)
                packed = child.sa_encode(rotated)
                self.assertEqual(packed.shape, (512, (3 * bits + 7) // 8))
                unpacked = unpack_pq_codes(packed, 3, bits)
                expected = []
                for row in packed:
                    reader = faiss.BitstringReader(faiss.swig_ptr(row), len(row))
                    expected.append([reader.read(bits) for _ in range(3)])
                np.testing.assert_array_equal(unpacked, expected)
                pq = child.pq
                centroids = faiss.vector_to_array(pq.centroids).reshape(3, 1 << bits, 4)
                nearest = []
                for position in range(3):
                    delta = (rotated[:, None, position * 4:(position + 1) * 4].astype(np.float64) -
                             centroids[position].astype(np.float64)[None])
                    nearest.append(np.sum(delta ** 2, axis=2).argmin(axis=1))
                np.testing.assert_array_equal(unpacked, np.column_stack(nearest))
                np.testing.assert_array_equal(model.encode(self.data)[:, model.rq_levels:], unpacked)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "model.npz"
                    model.save(path)
                    loaded = RQOPQ.load(path)
                    np.testing.assert_array_equal(model.encode(self.data), loaded.encode(self.data))

    def test_pq_boundary_ids_match_for_single_repeated_and_full_batches(self):
        model = self.model
        data = self.data[:64]
        _, residuals = model.rq_residuals(data)
        reconstruction = data - residuals
        rotation = faiss.downcast_VectorTransform(model.opq_index.chain.at(0))
        matrix = faiss.vector_to_array(rotation.A).reshape(12, 12).astype(np.float64)
        rotated = _rotate_stably(residuals, matrix).astype(np.float64)
        child = faiss.downcast_index(model.opq_index.index)
        pq = child.pq
        centers = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype(np.float64)
        candidates = []
        for row in range(len(data)):
            for position in range(pq.M):
                section = slice(position * pq.dsub, (position + 1) * pq.dsub)
                sub = rotated[row, section]
                closest = np.sum((centers[position] - sub) ** 2, axis=1).argsort()[:2]
                left, right = centers[position, closest]
                normal = right - left
                norm = normal @ normal
                if norm < 1e-10:
                    continue
                correction = (normal @ ((left + right) / 2) - normal @ sub) / norm
                for shift in (-2e-7, 0., 2e-7):
                    target = rotated[row].copy()
                    target[section] += (correction + shift) * normal
                    candidates.append((reconstruction[row].astype(np.float64) + target @ matrix).astype(np.float32))
        candidates = np.asarray(candidates)
        full = model.encode(candidates)
        single = np.concatenate([model.encode(row[None]) for row in candidates])
        np.testing.assert_array_equal(full, single)
        for row in candidates[:10]:
            repeated = model.encode(np.repeat(row[None], 32, axis=0))
            np.testing.assert_array_equal(repeated, np.repeat(model.encode(row[None]), 32, axis=0))
        # Native PQ encoding itself must keep row identity for exactly the
        # same rotated inputs; the stable rotation removes the BLAS boundary.
        _, residuals = model.rq_residuals(candidates)
        actual_rotated = _rotate_stably(residuals, matrix)
        packed = child.sa_encode(actual_rotated)
        packed_single = np.concatenate([child.sa_encode(row[None]) for row in actual_rotated])
        np.testing.assert_array_equal(packed, packed_single)
        np.testing.assert_array_equal(full[:, model.rq_levels:], unpack_pq_codes(packed, pq.M, pq.nbits))

    def test_version_one_checkpoint_needs_no_new_preprocessing_state(self):
        # The original version-one contract stores centroids in original
        # coordinates plus the self-contained FAISS index. Conditioning adds
        # no fields and existing artifacts keep that contract.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.npz"
            arrays = {f"rq_codebook_{i}": cb for i, cb in enumerate(self.model.rq_codebooks)}
            arrays["opq_index"] = faiss.serialize_index(self.model.opq_index)
            metadata = {"algorithm": "rq_opq", "dimension": 12,
                        "configuration": self.model.configuration(), "faiss_version": faiss.__version__}
            save_npz(path, metadata, arrays)
            loaded = RQOPQ.load(path)
            np.testing.assert_array_equal(self.model.encode(self.data), loaded.encode(self.data))

    def test_encoding_revision_is_written_and_unknown_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            self.model.save(path)
            metadata, arrays = load_npz(path, "rq_opq")
            self.assertEqual(metadata["format_version"], 1)
            self.assertEqual(metadata["encoding_revision"], 2)
            metadata["encoding_revision"] = 999
            unknown = Path(tmp) / "unknown.npz"
            save_npz(unknown, metadata, arrays)
            with self.assertRaisesRegex(ValueError, "unsupported RQ-OPQ encoding revision"):
                RQOPQ.load(unknown)

    def test_reject_missing_wrong_or_nonfinite_opq_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            self.model.save(path)
            metadata, arrays = load_npz(path, "rq_opq")
            child = faiss.downcast_index(self.model.opq_index.index)
            no_transform = faiss.IndexPreTransform(faiss.clone_index(child))
            linear = faiss.LinearTransform(12, 12)
            faiss.copy_array_to_vector((2 * np.eye(12, dtype=np.float32)).ravel(), linear.A)
            linear.is_trained = True
            wrong_transform = faiss.IndexPreTransform(linear, faiss.clone_index(child))
            nonfinite = faiss.clone_index(self.model.opq_index)
            rotation = faiss.downcast_VectorTransform(nonfinite.chain.at(0))
            matrix = faiss.vector_to_array(rotation.A)
            matrix[0] = np.nan
            faiss.copy_array_to_vector(matrix, rotation.A)
            invalid_pq = faiss.clone_index(self.model.opq_index)
            pq = faiss.downcast_index(invalid_pq.index).pq
            centroids = faiss.vector_to_array(pq.centroids)
            centroids[0] = np.inf
            faiss.copy_array_to_vector(centroids, pq.centroids)
            for name, index in (("no_transform", no_transform), ("wrong_transform", wrong_transform),
                                ("nonfinite_rotation", nonfinite), ("nonfinite_pq", invalid_pq),
                                ("bare_pq", faiss.clone_index(child))):
                with self.subTest(name=name):
                    arrays["opq_index"] = faiss.serialize_index(index)
                    altered = Path(tmp) / f"{name}.npz"
                    save_npz(altered, metadata, arrays)
                    with self.assertRaises(ValueError):
                        RQOPQ.load(altered)


if __name__ == "__main__":
    unittest.main()
