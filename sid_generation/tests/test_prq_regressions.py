"""PRQ regressions for stable encoding at cosine decision boundaries."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prq_kmeans import PRQKMeans
from sid_common import normalize_rows


class PRQBatchRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This fixed training set produced a three-token discrepancy between
        # one-row GEMV and many-row GEMM with NumPy 2.3.5 / Accelerate.
        rng = np.random.default_rng(2303)
        rng.normal(size=(32, 4))
        cls.training = rng.normal(size=(32, 4)).astype(np.float32)
        cls.fitted = PRQKMeans((4, 4, 4), top_k=2, n_iter=5,
                              batch_size=2048, seed=42).fit(cls.training)
        directions = normalize_rows(cls.fitted.codebooks[0])
        cls.boundary = normalize_rows(directions[:1].astype(np.float64) + directions[3:4])

    def setUp(self):
        self.model = copy.deepcopy(self.fitted)

    def test_boundary_query_is_independent_of_batch_membership(self):
        expected = self.model.encode(self.boundary)
        repeated = np.repeat(self.boundary, 20, axis=0)
        np.testing.assert_array_equal(self.model.encode(repeated), np.repeat(expected, 20, axis=0))

        # A query's position and its unrelated neighbors must not choose its SID.
        surrounding = np.concatenate([self.training[:9], self.boundary, self.training[9:]])
        np.testing.assert_array_equal(self.model.encode(surrounding)[9:10], expected)

    def test_boundary_catalog_slices_batch_sizes_and_reload(self):
        catalog = np.concatenate([self.boundary, self.training[:13],
                                  np.repeat(self.boundary, 9, axis=0), self.training[13:]])
        expected = np.concatenate([self.model.encode(row[None, :]) for row in catalog])
        slices = (catalog[:1], catalog[1:14], catalog[14:18], catalog[18:])
        for batch_size in (1, 2, 3, 7, 19, 2048):
            with self.subTest(batch_size=batch_size):
                self.model.batch_size = batch_size
                np.testing.assert_array_equal(self.model.encode(catalog), expected)
                np.testing.assert_array_equal(
                    np.concatenate([self.model.encode(block) for block in slices]), expected)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boundary.npz"
            self.model.save(path)
            loaded = PRQKMeans.load(path)
            for batch_size in (1, 5, 2048):
                with self.subTest(reloaded_batch_size=batch_size):
                    loaded.batch_size = batch_size
                    np.testing.assert_array_equal(loaded.encode(catalog), expected)

    def test_random_catalog_reordering_and_batch_sizes(self):
        rng = np.random.default_rng(19)
        training = rng.normal(size=(160, 12)).astype(np.float32)
        model = PRQKMeans((8, 8, 8), n_iter=5, batch_size=37, seed=3).fit(training)
        catalog = np.concatenate([rng.normal(size=(41, 12)).astype(np.float32),
                                  training[:10], np.zeros((1, 12), dtype=np.float32)])
        expected = np.concatenate([model.encode(row[None, :]) for row in catalog])
        order = rng.permutation(len(catalog))
        for batch_size in (1, 3, 16, 2048):
            with self.subTest(batch_size=batch_size):
                model.batch_size = batch_size
                np.testing.assert_array_equal(model.encode(catalog), expected)
                np.testing.assert_array_equal(model.encode(catalog[order]), expected[order])

    def test_exact_ties_keep_lowest_centroid_index(self):
        training = np.ones((8, 4), dtype=np.float32)
        model = PRQKMeans((2, 2, 2), n_iter=3).fit(training)
        for batch_size in (1, 3, 2048):
            with self.subTest(batch_size=batch_size):
                model.batch_size = batch_size
                np.testing.assert_array_equal(model.encode(training), np.zeros((8, 3), dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
