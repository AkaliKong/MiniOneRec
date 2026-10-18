"""PRQ-KMeans, independently implemented from arXiv:2608.24207v2 Algorithm 1.

Levels are fitted and encoded sequentially. This is a NumPy CPU reference
implementation; the learned state is independent of catalog identity suffixes.
"""

import logging

import numpy as np

try:
    from .sid_common import load_npz, normalize_rows, positive_int, projection_residual, save_npz, vectors
except ImportError:
    from sid_common import load_npz, normalize_rows, positive_int, projection_residual, save_npz, vectors

LOGGER = logging.getLogger(__name__)


class PRQKMeans:
    encoding_revision = 2

    def __init__(self, codebook_sizes=(256, 256, 256), top_k=2, beta=15.0,
                 n_iter=30, batch_size=2048, seed=42, eps=1e-10):
        self.codebook_sizes = tuple(positive_int(k, "codebook size") for k in codebook_sizes)
        if not self.codebook_sizes:
            raise ValueError("at least one codebook is required")
        self.top_k = positive_int(top_k, "top_k")
        if self.top_k > min(self.codebook_sizes):
            raise ValueError("top_k must not exceed any codebook size")
        self.beta = float(beta)
        self.eps = float(eps)
        if not np.isfinite(self.beta) or self.beta <= 0 or self.beta > 1e6:
            raise ValueError("beta must be finite and in (0, 1000000]")
        if not np.isfinite(self.eps) or not 0 < self.eps < 1:
            raise ValueError("eps must be finite and in (0, 1)")
        self.n_iter = positive_int(n_iter, "n_iter")
        self.batch_size = positive_int(batch_size, "batch_size")
        self.seed = positive_int(seed, "seed", minimum=0)
        self.dimension = None
        self.global_mean = None
        self.codebooks = []

    @property
    def capacities(self):
        return list(self.codebook_sizes)

    def configuration(self):
        return {"codebook_sizes": list(self.codebook_sizes), "top_k": self.top_k,
                "beta": self.beta, "n_iter": self.n_iter, "batch_size": self.batch_size,
                "seed": self.seed, "eps": self.eps}

    @staticmethod
    def _cosine_scores(residuals, unit_centroids):
        """Reduce each row/centroid pair independently of the input batch shape.

        BLAS can round a vector product differently from a matrix product,
        changing boundary assignments. An unoptimized contraction keeps the
        dimension reduction fixed; float64 also preserves near-tie differences.
        Both operands have contiguous rows so input slicing cannot change the
        reduction layout. The temporary arrays are limited to the current block.
        """
        return np.einsum("nd,kd->nk",
                         np.asarray(residuals, dtype=np.float64, order="C"),
                         np.asarray(unit_centroids, dtype=np.float64, order="C"),
                         optimize=False)

    def _assign(self, residuals, centroids):
        unit_centroids = normalize_rows(centroids, self.eps)
        assignments = np.empty(len(residuals), dtype=np.int32)
        for start in range(0, len(residuals), self.batch_size):
            block = residuals[start:start + self.batch_size]
            assignments[start:start + len(block)] = np.argmax(self._cosine_scores(block, unit_centroids), axis=1)
        return assignments

    def fit(self, embeddings):
        data = vectors(embeddings)
        if len(data) < max(self.codebook_sizes):
            raise ValueError("fitting requires at least as many items as the largest codebook")
        residuals = normalize_rows(data, self.eps)
        mean = residuals.mean(axis=0, dtype=np.float64)
        residuals = projection_residual(residuals, mean, self.eps)
        rng = np.random.default_rng(self.seed)
        codebooks = []
        for level, capacity in enumerate(self.codebook_sizes):
            LOGGER.info("PRQ level %d/%d: %d centroids, %d iterations", level + 1,
                        len(self.codebook_sizes), capacity, self.n_iter)
            centroids = residuals[rng.choice(len(data), capacity, replace=False)].copy()
            for _ in range(self.n_iter):
                unit_centroids = normalize_rows(centroids, self.eps)
                sums = np.zeros(centroids.shape, dtype=np.float64)
                mass = np.zeros(capacity, dtype=np.float64)
                for start in range(0, len(data), self.batch_size):
                    block = residuals[start:start + self.batch_size]
                    scores = self._cosine_scores(block, unit_centroids)
                    # Stable sorting makes ties prefer the lowest centroid index.
                    choices = np.argsort(-scores, axis=1, kind="stable")[:, :self.top_k]
                    selected = np.take_along_axis(scores, choices, axis=1).astype(np.float64)
                    weights = np.exp(self.beta * (selected - selected.max(axis=1, keepdims=True)))
                    weights /= weights.sum(axis=1, keepdims=True)
                    for neighbor in range(self.top_k):
                        np.add.at(mass, choices[:, neighbor], weights[:, neighbor])
                        np.add.at(sums, choices[:, neighbor], weights[:, neighbor, None] * block)
                active = mass > 0
                # An empty centroid retains its previous value. Zero directions
                # and zero residuals are deterministic, not randomly reinitialized.
                centroids[active] = (sums[active] / mass[active, None]).astype(np.float32)
            codebooks.append(centroids)
            if level + 1 < len(self.codebook_sizes):
                assigned = self._assign(residuals, centroids)
                residuals = projection_residual(residuals, centroids[assigned], self.eps)
        self.dimension, self.global_mean, self.codebooks = data.shape[1], mean, codebooks
        return self

    def encode(self, embeddings):
        if not self.codebooks:
            raise ValueError("fit or load a PRQ model before encoding")
        data = vectors(embeddings, self.dimension)
        codes = np.empty((len(data), len(self.codebooks)), dtype=np.int32)
        for start in range(0, len(data), self.batch_size):
            block = data[start:start + self.batch_size]
            residuals = projection_residual(normalize_rows(block, self.eps), self.global_mean, self.eps)
            for level, centroids in enumerate(self.codebooks):
                chosen = self._assign(residuals, centroids)
                codes[start:start + len(block), level] = chosen
                if level + 1 < len(self.codebooks):
                    residuals = projection_residual(residuals, centroids[chosen], self.eps)
        return codes

    def save(self, path, overwrite=False):
        if not self.codebooks:
            raise ValueError("cannot save an unfitted PRQ model")
        metadata = {"algorithm": "prq_kmeans", "dimension": self.dimension,
                    "encoding_revision": self.encoding_revision,
                    "configuration": self.configuration(),
                    "reference": "https://arxiv.org/html/2608.24207v2"}
        arrays = {"global_mean": self.global_mean}
        arrays.update({f"codebook_{i}": cb for i, cb in enumerate(self.codebooks)})
        save_npz(path, metadata, arrays, overwrite)

    @classmethod
    def load(cls, path):
        metadata, arrays = load_npz(path, "prq_kmeans")
        if metadata.get("encoding_revision", 1) not in (1, cls.encoding_revision):
            raise ValueError("unsupported PRQ encoding revision")
        model = cls(**metadata["configuration"])
        model.dimension = positive_int(metadata["dimension"], "model dimension")
        mean = arrays.get("global_mean")
        if mean is None or mean.shape != (model.dimension,) or not np.isfinite(mean).all():
            raise ValueError("invalid global mean in PRQ checkpoint")
        model.global_mean = mean.astype(np.float64)
        for level, capacity in enumerate(model.codebook_sizes):
            cb = arrays.get(f"codebook_{level}")
            if cb is None or cb.shape != (capacity, model.dimension):
                raise ValueError(f"invalid PRQ codebook {level}")
            model.codebooks.append(vectors(cb, model.dimension))
        return model


def main():
    try:
        from .sid_cli import run_cli
    except ImportError:
        from sid_cli import run_cli
    run_cli("prq_kmeans", PRQKMeans)


if __name__ == "__main__":
    main()
