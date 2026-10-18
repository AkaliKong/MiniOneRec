"""Serial RQ prefix followed by independent OPQ/PQ residual subcodes.

This is an item-only MiniOneRec adaptation of the OneSearch quantization
structure. It uses FAISS CPU and direct, row-preserving standalone encoding.
It does not implement a parallel recommendation decoder.
"""

from contextlib import contextmanager
import logging

import numpy as np

try:
    from .sid_common import load_npz, positive_int, save_npz, vectors
except ImportError:
    from sid_common import load_npz, positive_int, save_npz, vectors

LOGGER = logging.getLogger(__name__)
_DISTANCE_WORKSPACE_BYTES = 16 * 1024 * 1024


def require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("RQ-OPQ requires FAISS; install the optional requirements-faiss.txt in a separate environment") from exc
    return faiss


@contextmanager
def faiss_threads(faiss, threads):
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(threads)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


def unpack_pq_codes(packed, subspaces, nbits):
    """Decode FAISS little-endian PQ bits, including fields crossing bytes."""
    packed = np.asarray(packed)
    expected = (subspaces * nbits + 7) // 8
    if packed.ndim != 2 or packed.shape[1] != expected or packed.dtype != np.uint8:
        raise ValueError(f"expected uint8 PQ codes with {expected} bytes per row")
    result = np.empty((len(packed), subspaces), dtype=np.int32)
    for position in range(subspaces):
        byte, shift = divmod(position * nbits, 8)
        value = packed[:, byte].astype(np.uint16)
        if shift + nbits > 8:
            value |= packed[:, byte + 1].astype(np.uint16) << 8
        result[:, position] = (value >> shift) & ((1 << nbits) - 1)
    return result


def _nearest_l2(queries, centroids, workspace_bytes=_DISTANCE_WORKSPACE_BYTES):
    """Direct double distances, with a bounded query/centroid workspace.

    FlatL2 switches to a float32 BLAS identity at 20 queries. That identity
    loses small differences between vectors with a large common offset and
    makes IDs depend on the caller's chunk sizes. Each distance here uses the
    same direct subtraction/reduction regardless of either block size.
    """
    dimension = queries.shape[1]
    pairs = max(1, workspace_bytes // (8 * dimension))
    centroid_block = min(len(centroids), 64, pairs)
    query_block = max(1, pairs // centroid_block)
    centroids64 = np.asarray(centroids, dtype=np.float64)
    chosen = np.full(len(queries), -1, dtype=np.int32)
    for start in range(0, len(queries), query_block):
        query = np.asarray(queries[start:start + query_block], dtype=np.float64)
        best = np.full(len(query), np.inf)
        selected = chosen[start:start + len(query)]
        for offset in range(0, len(centroids64), centroid_block):
            delta = query[:, None, :] - centroids64[None, offset:offset + centroid_block, :]
            distances = np.einsum("nkd,nkd->nk", delta, delta, optimize=False)
            local = distances.argmin(axis=1)
            candidate = distances[np.arange(len(query)), local]
            # Iterate centroids in order and keep the first equal minimum.
            better = candidate < best
            best[better] = candidate[better]
            selected[better] = offset + local[better]
        if not np.isfinite(best).all() or np.any(selected < 0):
            raise ValueError("RQ nearest-centroid distances must be finite")
    return chosen


def _check_faiss_magnitude(values, dimension, operation, sample_count=1):
    """Reject finite inputs whose float32 FAISS intermediates can overflow."""
    # Account for squared L2 distances, orthogonal transforms and, during
    # OPQ training, the cross-products accumulated over the training rows.
    limit = np.sqrt(np.finfo(np.float32).max / (16.0 * dimension * sample_count))
    if not np.isfinite(values).all() or np.max(np.abs(values)) > limit:
        raise ValueError(f"{operation} values exceed the safe FAISS float32 range; rescale embeddings")


def _rotate_stably(values, matrix):
    """Use a fixed double reduction, independent of BLAS batch kernels."""
    dimension = values.shape[1]
    rows = max(1, _DISTANCE_WORKSPACE_BYTES // (8 * dimension))
    matrix64 = np.asarray(matrix, dtype=np.float64)
    result = np.empty_like(values)
    for start in range(0, len(values), rows):
        block = np.asarray(values[start:start + rows], dtype=np.float64)
        rotated = np.einsum("nd,kd->nk", block, matrix64, optimize=False)
        with np.errstate(over="ignore", invalid="ignore"):
            result[start:start + len(block)] = rotated
    return result


def _validate_opq_index(faiss, index, dimension, subspaces, nbits):
    if (not isinstance(index, faiss.IndexPreTransform) or index.d != dimension
            or not index.is_trained or index.chain.size() != 1):
        raise ValueError("invalid RQ-OPQ transform structure, dimension or training state")
    rotation = faiss.downcast_VectorTransform(index.chain.at(0))
    # FAISS serializes a trained OPQMatrix as its LinearTransform base; its
    # training-only M/pq fields and concrete OPQ type are not retained.
    if (not isinstance(rotation, faiss.LinearTransform) or rotation.d_in != dimension
            or rotation.d_out != dimension
            or not rotation.is_trained or rotation.have_bias):
        raise ValueError("checkpoint requires one trained square OPQ rotation")
    matrix = faiss.vector_to_array(rotation.A)
    if matrix.size != dimension * dimension or not np.isfinite(matrix).all():
        raise ValueError("invalid OPQ rotation coefficients")
    # The serialized is_orthonormal flag is not proof about the coefficients.
    # Check the actual matrix in blocks rather than materializing a full Gram
    # matrix alongside a potentially high-dimensional checkpoint.
    matrix = matrix.reshape(dimension, dimension)
    for start in range(0, dimension, 64):
        with np.errstate(over="ignore", invalid="ignore"):
            gram = matrix[start:start + 64] @ matrix.T
        identity_rows = np.zeros_like(gram)
        identity_rows[np.arange(len(gram)), start + np.arange(len(gram))] = 1
        if not np.allclose(gram, identity_rows, atol=2e-4, rtol=2e-4):
            raise ValueError("checkpoint OPQ matrix must be orthogonal")
    child = faiss.downcast_index(index.index)
    if (not isinstance(child, faiss.IndexPQ) or child.d != dimension or not child.is_trained
            or child.metric_type != faiss.METRIC_L2 or child.pq.M != subspaces
            or child.pq.nbits != nbits):
        raise ValueError("FAISS PQ structure does not match checkpoint metadata")
    centroids = faiss.vector_to_array(child.pq.centroids)
    if centroids.size != dimension * (1 << nbits):
        raise ValueError("invalid FAISS PQ centroid count")
    _check_faiss_magnitude(centroids, dimension, "PQ checkpoint")
    return rotation, child


class RQOPQ:
    encoding_revision = 2

    def __init__(self, rq_levels=3, codebook_size=256, opq_subspaces=2, nbits=8,
                 rq_niter=25, opq_niter=20, pq_niter=25, batch_size=4096,
                 seed=42, threads=1):
        self.rq_levels = positive_int(rq_levels, "rq_levels")
        self.codebook_size = positive_int(codebook_size, "codebook_size", minimum=2)
        self.opq_subspaces = positive_int(opq_subspaces, "opq_subspaces")
        self.nbits = positive_int(nbits, "nbits")
        if self.nbits > 8:
            raise ValueError("this implementation supports 1 through 8 bits per OPQ subcode")
        self.rq_niter = positive_int(rq_niter, "rq_niter")
        self.opq_niter = positive_int(opq_niter, "opq_niter")
        self.pq_niter = positive_int(pq_niter, "pq_niter")
        self.batch_size = positive_int(batch_size, "batch_size")
        self.seed = positive_int(seed, "seed", minimum=0)
        if self.seed > 2 ** 31 - 1:
            raise ValueError("FAISS seed must fit in a signed 32-bit integer")
        self.threads = positive_int(threads, "threads")
        self.dimension = None
        self.rq_codebooks = []
        self.opq_index = None

    @property
    def capacities(self):
        return [self.codebook_size] * self.rq_levels + [1 << self.nbits] * self.opq_subspaces

    def configuration(self):
        return {key: getattr(self, key) for key in (
            "rq_levels", "codebook_size", "opq_subspaces", "nbits", "rq_niter",
            "opq_niter", "pq_niter", "batch_size", "seed", "threads")}

    def _prefix(self, embeddings, codebooks):
        residuals = embeddings.copy()
        codes = np.empty((len(residuals), len(codebooks)), dtype=np.int32)
        for level, centroids in enumerate(codebooks):
            for start in range(0, len(residuals), self.batch_size):
                block = residuals[start:start + self.batch_size]
                chosen = _nearest_l2(block, centroids)
                codes[start:start + len(block), level] = chosen
                with np.errstate(over="ignore", invalid="ignore"):
                    block -= centroids[chosen]
                if not np.isfinite(block).all():
                    raise ValueError("RQ residuals exceed the float32 range; rescale embeddings")
        return codes, residuals

    def fit(self, embeddings):
        faiss = require_faiss()
        data = vectors(embeddings)
        if data.shape[1] % self.opq_subspaces:
            raise ValueError("embedding dimension must be divisible by opq_subspaces; no implicit padding is applied")
        required = max(self.codebook_size, 1 << self.nbits)
        if len(data) < required:
            raise ValueError(f"RQ-OPQ fitting requires at least {required} items for these code sizes")
        residuals = data.copy()
        codebooks = []
        with faiss_threads(faiss, self.threads):
            for level in range(self.rq_levels):
                LOGGER.info("RQ level %d/%d: %d centroids", level + 1, self.rq_levels, self.codebook_size)
                kmeans = faiss.Kmeans(data.shape[1], self.codebook_size, niter=self.rq_niter,
                                     nredo=1, seed=(self.seed + level) % (2 ** 31), gpu=False,
                                     max_points_per_centroid=max(256, len(data)))
                # A fixed translation and scalar preserve the L2 objective.
                # Center in double before float32 FAISS training so a shared
                # large offset cannot destroy the training assignments either.
                centered = residuals.astype(np.float64)
                center = centered.mean(axis=0)
                centered -= center
                scale = max(float(np.max(np.abs(centered))), np.finfo(np.float32).tiny)
                centered /= scale
                training = np.ascontiguousarray(centered, dtype=np.float32)
                kmeans.train(training)
                centroids = vectors(kmeans.centroids.astype(np.float64) * scale + center, data.shape[1])
                codebooks.append(centroids)
                _, residuals = self._prefix(residuals, [centroids])

            LOGGER.info("OPQ: %d subspaces, %d bits, %d outer iterations",
                        self.opq_subspaces, self.nbits, self.opq_niter)
            _check_faiss_magnitude(residuals, data.shape[1], "OPQ training", len(data))
            # OPQMatrix otherwise defaults to an 8-bit internal PQ. Set its
            # training PQ explicitly so smaller validation configurations are real.
            training_pq = faiss.ProductQuantizer(data.shape[1], self.opq_subspaces, self.nbits)
            training_pq.cp.seed = self.seed
            training_pq.cp.niter = self.pq_niter
            training_pq.cp.max_points_per_centroid = max(256, len(data))
            rotation = faiss.OPQMatrix(data.shape[1], self.opq_subspaces)
            rotation.pq = training_pq
            rotation.niter = self.opq_niter
            rotation.niter_pq = self.pq_niter
            rotation.niter_pq_0 = self.pq_niter
            rotation.max_train_points = len(data)
            index_pq = faiss.IndexPQ(data.shape[1], self.opq_subspaces, self.nbits)
            index_pq.pq.cp.seed = self.seed
            index_pq.pq.cp.niter = self.pq_niter
            index_pq.pq.cp.max_points_per_centroid = max(256, len(data))
            pipeline = faiss.IndexPreTransform(rotation, index_pq)
            pipeline.train(residuals)
            # Serialize and reload now so the stored model owns all transforms
            # independently of temporary SWIG training references.
            pipeline = faiss.deserialize_index(faiss.serialize_index(pipeline))
            _validate_opq_index(faiss, pipeline, data.shape[1], self.opq_subspaces, self.nbits)
        self.dimension, self.rq_codebooks, self.opq_index = data.shape[1], codebooks, pipeline
        return self

    def rq_residuals(self, embeddings):
        if self.opq_index is None:
            raise ValueError("fit or load an RQ-OPQ model first")
        data = vectors(embeddings, self.dimension)
        faiss = require_faiss()
        with faiss_threads(faiss, self.threads):
            return self._prefix(data, self.rq_codebooks)

    def encode(self, embeddings):
        if self.opq_index is None:
            raise ValueError("fit or load an RQ-OPQ model before encoding")
        data = vectors(embeddings, self.dimension)
        faiss = require_faiss()
        codes = np.empty((len(data), self.rq_levels + self.opq_subspaces), dtype=np.int32)
        rotation = faiss.downcast_VectorTransform(self.opq_index.chain.at(0))
        matrix = faiss.vector_to_array(rotation.A).reshape(self.dimension, self.dimension)
        index_pq = faiss.downcast_index(self.opq_index.index)
        with faiss_threads(faiss, self.threads):
            for start in range(0, len(data), self.batch_size):
                block = data[start:start + self.batch_size]
                prefix, residuals = self._prefix(block, self.rq_codebooks)
                _check_faiss_magnitude(residuals, self.dimension, "OPQ encoding")
                # Native float32 rotation kernels also change their rounding
                # with query count, enough to cross a PQ decision boundary.
                rotated = _rotate_stably(residuals, matrix)
                _check_faiss_magnitude(rotated, self.dimension, "OPQ rotated encoding")
                packed = index_pq.sa_encode(rotated)
                suffix = unpack_pq_codes(packed, self.opq_subspaces, self.nbits)
                codes[start:start + len(block)] = np.column_stack((prefix, suffix))
        return codes

    def save(self, path, overwrite=False):
        if self.opq_index is None:
            raise ValueError("cannot save an unfitted RQ-OPQ model")
        faiss = require_faiss()
        metadata = {"algorithm": "rq_opq", "dimension": self.dimension,
                    "encoding_revision": self.encoding_revision,
                    "configuration": self.configuration(), "faiss_version": faiss.__version__,
                    "reference": "https://arxiv.org/abs/2509.03236",
                    "distance": "squared L2; item-only RQ + OPQ/PQ adaptation"}
        arrays = {f"rq_codebook_{i}": cb for i, cb in enumerate(self.rq_codebooks)}
        arrays["opq_index"] = faiss.serialize_index(self.opq_index)
        save_npz(path, metadata, arrays, overwrite)

    @classmethod
    def load(cls, path):
        faiss = require_faiss()
        metadata, arrays = load_npz(path, "rq_opq")
        revision = positive_int(metadata.get("encoding_revision", 1), "encoding revision")
        if revision not in (1, cls.encoding_revision):
            raise ValueError(f"unsupported RQ-OPQ encoding revision {revision}")
        model = cls(**metadata["configuration"])
        model.dimension = positive_int(metadata["dimension"], "model dimension")
        if model.dimension % model.opq_subspaces:
            raise ValueError("inconsistent RQ-OPQ checkpoint dimensions")
        for level in range(model.rq_levels):
            cb = arrays.get(f"rq_codebook_{level}")
            if cb is None or cb.shape != (model.codebook_size, model.dimension):
                raise ValueError(f"invalid RQ codebook {level}")
            model.rq_codebooks.append(vectors(cb, model.dimension))
        blob = arrays.get("opq_index")
        if blob is None or blob.ndim != 1 or blob.dtype != np.uint8:
            raise ValueError("invalid FAISS index in RQ-OPQ checkpoint")
        try:
            model.opq_index = faiss.deserialize_index(blob)
        except RuntimeError as exc:
            raise ValueError("invalid FAISS index in RQ-OPQ checkpoint") from exc
        _validate_opq_index(faiss, model.opq_index, model.dimension, model.opq_subspaces, model.nbits)
        return model


def main():
    try:
        from .sid_cli import run_cli
    except ImportError:
        from sid_cli import run_cli
    run_cli("rq_opq", RQOPQ)


if __name__ == "__main__":
    main()
