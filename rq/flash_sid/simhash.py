"""Training-free semantic codes from fixed Gaussian hyperplanes.

This is an independent implementation of the Stage I construction in FLASH,
arXiv:2610.07402, Eq. (1). Raw codes use zero-based indexing. There is no
normalization or centering, and no collision-resolution suffix in this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


_FORMAT_VERSION = 1
_CONFIG_KEYS = {"format_version", "input_dim", "num_codebooks", "codebook_size", "seed"}


def _integer(name: str, value: Any, minimum: int = 1) -> int:
    """Accept integer scalars, while rejecting booleans and lossy coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


class SimHashTokenizer:
    """Map real item embeddings to independent, zero-based semantic codes.

    Gaussian samples are generated in float64, then stored in float32. This
    preserves the random-number sequence used by the FLASH reference tokenizer;
    drawing float32 samples directly would produce different hyperplanes.

    Parameters are immutable by convention. ``planes`` has shape
    ``(num_codebooks, log2(codebook_size), input_dim)`` and is read-only.
    """

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int = 3,
        codebook_size: int = 256,
        seed: int = 2024,
    ) -> None:
        self._configure(input_dim, num_codebooks, codebook_size, seed)
        generator = np.random.default_rng(self.seed)
        samples = generator.standard_normal(self._plane_shape)
        self.planes = samples.astype(np.float32)
        self.planes.setflags(write=False)

    def _configure(
        self, input_dim: int, num_codebooks: int, codebook_size: int, seed: int
    ) -> None:
        self.input_dim = _integer("input_dim", input_dim)
        self.num_codebooks = _integer("num_codebooks", num_codebooks)
        self.codebook_size = _integer("codebook_size", codebook_size, minimum=2)
        if self.codebook_size > 2**30 or self.codebook_size & (self.codebook_size - 1):
            raise ValueError("codebook_size must be a power of two between 2 and 2**30")
        self.seed = _integer("seed", seed, minimum=0)
        self._bits_per_code = self.codebook_size.bit_length() - 1

    @property
    def _plane_shape(self) -> tuple[int, int, int]:
        return (self.num_codebooks, self._bits_per_code, self.input_dim)

    def _validate_planes(self) -> None:
        if not isinstance(self.planes, np.ndarray):
            raise ValueError("planes must be a NumPy array")
        if self.planes.shape != self._plane_shape:
            raise ValueError(f"planes must have shape {self._plane_shape}")
        if self.planes.dtype != np.dtype(np.float32):
            raise ValueError("planes must have dtype float32")
        if not np.isfinite(self.planes).all():
            raise ValueError("planes must contain only finite values")

    def encode(self, embeddings: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return an ``(N, num_codebooks)`` int64 array.

        Input validation and conversion run one batch at a time, so a memory-
        mapped input does not need to be copied into memory in its entirety.
        Input values must remain finite and each row nonzero after float32
        conversion. Zero vectors have no angular semantics and are rejected.
        """
        batch_size = _integer("batch_size", batch_size)
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a NumPy array (a memmap is supported)")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("embeddings must be a nonempty two-dimensional array")
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"embedding dimension is {embeddings.shape[1]}, expected {self.input_dim}"
            )
        if embeddings.dtype.kind not in "fiu":
            raise TypeError("embeddings must have a real numeric dtype, excluding booleans")
        self._validate_planes()

        output = np.empty((embeddings.shape[0], self.num_codebooks), dtype=np.int64)
        projection_matrix = self.planes.reshape(-1, self.input_dim).T
        bit_weights = np.left_shift(np.int64(1), np.arange(self._bits_per_code, dtype=np.int64))

        for start in range(0, embeddings.shape[0], batch_size):
            stop = min(start + batch_size, embeddings.shape[0])
            source = embeddings[start:stop]
            if not np.isfinite(source).all():
                raise ValueError(f"embeddings contain a non-finite value in rows [{start}, {stop})")
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.asarray(source, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"float32 conversion overflow in rows [{start}, {stop})")
            nonzero_rows = np.any(values != 0, axis=1)
            if not nonzero_rows.all():
                row = start + int(np.flatnonzero(~nonzero_rows)[0])
                raise ValueError(f"embedding row {row} is zero after float32 conversion")

            with np.errstate(over="ignore", invalid="ignore"):
                scores = np.matmul(values, projection_matrix)
            if not np.isfinite(scores).all():
                raise ValueError(f"float32 projection overflow in rows [{start}, {stop})")
            scores = scores.reshape(stop - start, self.num_codebooks, self._bits_per_code)
            # Position b contributes 2**b: least-significant-bit first.
            output[start:stop] = np.sum(
                np.greater(scores, 0) * bit_weights, axis=2, dtype=np.int64
            )
        return output

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save the exact projection and versioned metadata to NPZ.

        The supplied filename is used exactly, without appending an extension.
        The archive contains ``metadata`` (a JSON scalar) and ``planes``.
        """
        self._validate_planes()
        metadata = {
            "format_version": _FORMAT_VERSION,
            "input_dim": self.input_dim,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "seed": self.seed,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata=np.array(json.dumps(metadata, sort_keys=True)),
                    planes=self.planes,
                )
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SimHashTokenizer:
        """Load and validate an NPZ state without regenerating random planes."""
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "planes"}:
                raise ValueError("invalid tokenizer archive: expected metadata and planes")
            stored_metadata = archive["metadata"]
            if stored_metadata.ndim != 0 or stored_metadata.dtype.kind != "U":
                raise ValueError("metadata must be a Unicode JSON scalar")
            try:
                metadata = json.loads(stored_metadata.item())
            except (TypeError, ValueError) as error:
                raise ValueError("invalid tokenizer metadata JSON") from error
            if not isinstance(metadata, dict) or set(metadata) != _CONFIG_KEYS:
                raise ValueError("invalid tokenizer metadata fields")
            version = _integer("format_version", metadata["format_version"])
            if version != _FORMAT_VERSION:
                raise ValueError(f"unsupported tokenizer format_version: {version}")
            tokenizer = cls.__new__(cls)
            tokenizer._configure(
                metadata["input_dim"],
                metadata["num_codebooks"],
                metadata["codebook_size"],
                metadata["seed"],
            )
            tokenizer.planes = archive["planes"]
            tokenizer._validate_planes()
            tokenizer.planes.setflags(write=False)
        return tokenizer
