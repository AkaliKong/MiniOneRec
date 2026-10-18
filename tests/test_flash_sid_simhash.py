"""Contract tests for FLASH Stage I's random-hyperplane semantic codes.

The reference below deliberately uses scalar dot products and integer bit
assembly, independently of the implementation's batched matrix operations.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rq.flash_sid.simhash import SimHashTokenizer


def scalar_codes(embeddings, planes):
    """Equation oracle: positive projection b contributes the integer 2**b."""
    rows = []
    for embedding in embeddings:
        codes = []
        for codebook in planes:
            code = 0
            for bit, plane in enumerate(codebook):
                projection = sum(
                    float(coordinate) * float(weight)
                    for coordinate, weight in zip(embedding, plane)
                )
                if projection > 0:
                    code += 1 << bit
            codes.append(code)
        rows.append(codes)
    return np.asarray(rows, dtype=np.int64)


class SimHashFormulaTests(unittest.TestCase):
    def test_matches_independent_scalar_equation_and_rng_definition(self):
        seed = 2024
        expected_planes = np.random.default_rng(seed).standard_normal((3, 8, 11)).astype(np.float32)
        embeddings = np.random.default_rng(91).standard_normal((19, 11)).astype(np.float32)
        tokenizer = SimHashTokenizer(input_dim=11, seed=seed)

        self.assertEqual(tokenizer.planes.shape, (3, 8, 11))
        self.assertEqual(tokenizer.planes.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(tokenizer.planes, expected_planes)
        actual = tokenizer.encode(embeddings, batch_size=4)
        np.testing.assert_array_equal(actual, scalar_codes(embeddings, expected_planes))
        self.assertEqual(actual.shape, (19, 3))
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        self.assertTrue(np.all(actual >= 0))
        self.assertTrue(np.all(actual < 256))

    def test_known_directions_zero_projection_and_little_endian_bits(self):
        tokenizer = SimHashTokenizer(input_dim=3, num_codebooks=2, codebook_size=4)
        tokenizer.planes = np.array(
            [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]], dtype=np.float32
        )
        embeddings = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1],
             [1, 1, 1], [1, -1, 1], [-1, 1, -1]],
            dtype=np.float32,
        )
        expected = np.array([[1, 0], [2, 1], [0, 2], [0, 0], [3, 3], [1, 2], [2, 1]])
        np.testing.assert_array_equal(tokenizer.encode(embeddings), expected)

    def test_highest_and_lowest_bit_positions(self):
        tokenizer = SimHashTokenizer(input_dim=8, num_codebooks=1, codebook_size=256)
        tokenizer.planes = np.eye(8, dtype=np.float32)[None, :, :]
        embeddings = np.array(
            [[1, -1, -1, -1, -1, -1, -1, 1],
             [-1, -1, -1, -1, -1, -1, -1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32
        )
        np.testing.assert_array_equal(tokenizer.encode(embeddings), [[129], [128], [255]])

    def test_two_entry_codebook_has_one_bit(self):
        tokenizer = SimHashTokenizer(input_dim=2, num_codebooks=1, codebook_size=2)
        tokenizer.planes = np.array([[[1, 0]]], dtype=np.float32)
        np.testing.assert_array_equal(
            tokenizer.encode(np.array([[1, 0], [-1, 0], [0, 1]], dtype=np.float32)),
            [[1], [0], [0]],
        )

    def test_same_seed_and_all_batch_sizes_produce_same_codes(self):
        embeddings = np.random.default_rng(193).normal(size=(37, 17)).astype(np.float32)
        first = SimHashTokenizer(input_dim=17, seed=42)
        second = SimHashTokenizer(input_dim=17, seed=42)
        np.testing.assert_array_equal(first.planes, second.planes)
        expected = first.encode(embeddings)
        for batch_size in (1, 2, 7, 37, 100):
            with self.subTest(batch_size=batch_size):
                np.testing.assert_array_equal(second.encode(embeddings, batch_size), expected)

    def test_different_seeds_change_projection(self):
        first = SimHashTokenizer(input_dim=5, seed=1)
        second = SimHashTokenizer(input_dim=5, seed=2)
        self.assertFalse(np.array_equal(first.planes, second.planes))

    def test_positive_scaling_preserves_sign_codes(self):
        embeddings = np.random.default_rng(319).normal(size=(13, 9)).astype(np.float32)
        tokenizer = SimHashTokenizer(input_dim=9)
        np.testing.assert_array_equal(tokenizer.encode(embeddings), tokenizer.encode(embeddings * 8))

    def test_original_32_codebook_configuration(self):
        tokenizer = SimHashTokenizer(input_dim=7, num_codebooks=32, codebook_size=256)
        embeddings = np.random.default_rng(3).normal(size=(5, 7)).astype(np.float32)
        expected_planes = np.random.default_rng(2024).standard_normal((32, 8, 7)).astype(np.float32)
        actual = tokenizer.encode(embeddings, batch_size=2)
        self.assertEqual(actual.shape, (5, 32))
        self.assertEqual(actual.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(actual, scalar_codes(embeddings, expected_planes))

    def test_save_load_retains_exact_projection_and_codes(self):
        tokenizer = SimHashTokenizer(input_dim=3, num_codebooks=2, codebook_size=4, seed=71)
        # Non-random planes ensure this catches a loader that only recreates seed.
        tokenizer.planes = np.array(
            [[[1, 2, 0], [0, 1, -1]], [[-1, 0, 1], [1, 1, 1]]], dtype=np.float32
        )
        embeddings = np.array([[1, 2, 3], [-1, 4, 1], [3, 0, -2]], dtype=np.float32)
        expected = tokenizer.encode(embeddings)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "tokenizer.npz"
            tokenizer.save(path)
            self.assertTrue(path.is_file())
            restored = SimHashTokenizer.load(path)
        np.testing.assert_array_equal(restored.planes, tokenizer.planes)
        self.assertEqual(restored.planes.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(restored.encode(embeddings, batch_size=1), expected)
        self.assertEqual(restored.input_dim, 3)
        self.assertEqual(restored.num_codebooks, 2)
        self.assertEqual(restored.codebook_size, 4)
        self.assertEqual(restored.seed, 71)

    def test_memory_mapped_input_matches_in_memory(self):
        embeddings = np.random.default_rng(52).normal(size=(23, 6)).astype(np.float32)
        tokenizer = SimHashTokenizer(input_dim=6)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "embeddings.npy"
            np.save(path, embeddings)
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                np.testing.assert_array_equal(
                    tokenizer.encode(mapped, batch_size=5), tokenizer.encode(embeddings)
                )
            finally:
                # Release the file on Windows before TemporaryDirectory cleanup.
                mapped._mmap.close()


class SimHashValidationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimHashTokenizer(input_dim=3)
        self.valid = np.array([[1, 2, 3], [3, -2, 1]], dtype=np.float32)

    def test_invalid_codebook_sizes(self):
        for size in (0, 1, -2, 3, 6, 255, 257, 2.5, "256", True):
            with self.subTest(codebook_size=size):
                with self.assertRaises((TypeError, ValueError)):
                    SimHashTokenizer(input_dim=3, codebook_size=size)

    def test_invalid_codebook_counts(self):
        for count in (0, -1, 1.5, "3", True):
            with self.subTest(num_codebooks=count):
                with self.assertRaises((TypeError, ValueError)):
                    SimHashTokenizer(input_dim=3, num_codebooks=count)

    def test_invalid_input_dimensions(self):
        for dimension in (0, -1, 1.5, "3", True):
            with self.subTest(input_dim=dimension):
                with self.assertRaises((TypeError, ValueError)):
                    SimHashTokenizer(input_dim=dimension)

    def test_invalid_rank_wrong_dimension_and_empty_input(self):
        arrays = [
            np.array(1, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.ones((2, 1, 3), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        ]
        for embeddings in arrays:
            with self.subTest(shape=embeddings.shape):
                with self.assertRaises((TypeError, ValueError)):
                    self.tokenizer.encode(embeddings)

    def test_nonfinite_values_are_rejected_in_later_batches(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                embeddings = self.valid.copy()
                embeddings[1, 2] = value
                with self.assertRaises((TypeError, ValueError)):
                    self.tokenizer.encode(embeddings, batch_size=1)

    def test_zero_rows_are_rejected_in_later_batches(self):
        embeddings = self.valid.copy()
        embeddings[1] = 0
        with self.assertRaises((TypeError, ValueError)):
            self.tokenizer.encode(embeddings, batch_size=1)
        with self.assertRaises((TypeError, ValueError)):
            self.tokenizer.encode(np.zeros((1, 3), dtype=np.float32))

    def test_invalid_batch_sizes(self):
        for batch_size in (0, -1, -20, 1.5, "4", True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises((TypeError, ValueError)):
                    self.tokenizer.encode(self.valid, batch_size=batch_size)

    def test_nonreal_or_boolean_input_is_rejected(self):
        for dtype in (np.complex64, np.bool_, object, str):
            with self.subTest(dtype=dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.tokenizer.encode(self.valid.astype(dtype))

    def test_float32_conversion_overflow_and_underflow_to_zero_are_rejected(self):
        for embeddings in (
            np.array([[1e100, 1, 2]], dtype=np.float64),
            np.array([[1e-100, 0, 0]], dtype=np.float64),
        ):
            with self.subTest(embeddings=embeddings.tolist()):
                with self.assertRaises((TypeError, ValueError)):
                    self.tokenizer.encode(embeddings)


if __name__ == "__main__":
    unittest.main()
