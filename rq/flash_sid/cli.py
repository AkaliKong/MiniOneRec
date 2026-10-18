"""Generate offline semantic IDs from existing item embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from .export import build_minionerec_index, collision_metrics, load_item_ids
from .simhash import SimHashTokenizer


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FLASH Stage I SimHash SID generator; CPU/NumPy only")
    p.add_argument("--embeddings", "--data-path", "--data_path", required=True, type=Path,
                   help="Numeric .npy matrix [items, embedding_dim], without a padding row")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--dataset", default="items", help="Output filename stem (letters, digits, _ or -)")
    p.add_argument("--format", choices=["minionerec", "raw"], default="minionerec")
    p.add_argument("--num-codebooks", type=int, default=None,
                   help="Default 3 for MiniOneRec; 32 for raw FLASH-style codes")
    p.add_argument("--codebook-size", type=int, default=None, help="Power of two; default 256")
    p.add_argument("--seed", type=int, default=None, help="Default 2024")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--suffix-size", type=int, default=256,
                   help="Maximum items sharing a raw SID; MiniOneRec only")
    alignment = p.add_mutually_exclusive_group()
    alignment.add_argument("--item-ids", type=Path, help="JSON list matching embedding row order")
    alignment.add_argument("--item-meta", type=Path,
                           help="MiniOneRec .item.json; validates contiguous 0-based IDs")
    p.add_argument("--model-in", type=Path, help="Reuse saved .simhash.npz projection")
    p.add_argument("--overwrite", action="store_true", help="Replace this dataset's existing generated files")
    return p


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.dataset):
        raise ValueError("--dataset must use only letters, digits, _ or -")
    if args.embeddings.suffix.lower() != ".npy":
        raise ValueError("--embeddings must be a numeric .npy file")
    x = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("Embeddings must have shape [N, D] with N and D positive")
    ids = load_item_ids(len(x), args.item_ids, args.item_meta)
    if args.model_in is not None:
        model = SimHashTokenizer.load(args.model_in)
        for option, name in [(args.num_codebooks, "num_codebooks"),
                             (args.codebook_size, "codebook_size"), (args.seed, "seed")]:
            if option is not None and option != getattr(model, name):
                raise ValueError(f"Explicit --{name.replace('_', '-')} conflicts with --model-in")
    else:
        model = SimHashTokenizer(
            input_dim=x.shape[1],
            num_codebooks=args.num_codebooks if args.num_codebooks is not None else (3 if args.format == "minionerec" else 32),
            codebook_size=args.codebook_size if args.codebook_size is not None else 256,
            seed=args.seed if args.seed is not None else 2024,
        )
    if args.format == "minionerec" and model.num_codebooks != 3:
        raise ValueError("MiniOneRec format uses 3 semantic codes plus 1 suffix. Use --format raw for other lengths")

    endings = ["codes.npy", "item_ids.json", "simhash.npz", "metrics.json", "manifest.json"]
    if args.format == "minionerec":
        endings.append("index.json")
    paths = {ending: args.output_dir / f"{args.dataset}.{ending}" for ending in endings}
    # Reject output/input aliasing even with --overwrite.
    inputs = {p.resolve() for p in [args.embeddings, args.item_ids, args.item_meta, args.model_in] if p is not None}
    if any(p.resolve() in inputs for p in paths.values()):
        raise ValueError("An output path overlaps an input; use a different output directory or dataset name")
    existing = [str(p) for p in paths.values() if p.exists()]
    # A raw run must not silently leave a stale MiniOneRec index next to new codes.
    stale_index = args.output_dir / f"{args.dataset}.index.json"
    if args.format == "raw" and stale_index.exists():
        raise ValueError("A MiniOneRec index already exists for this output prefix. Use a fresh directory or dataset name for raw mode")
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; use a new directory or --overwrite: " + ", ".join(existing))

    start = time.perf_counter()
    codes = model.encode(x, batch_size=args.batch_size)
    encoding_seconds = time.perf_counter() - start
    metrics = collision_metrics(codes, model.codebook_size)
    index = None
    if args.format == "minionerec":
        index = build_minionerec_index(codes, ids, args.suffix_size)
        metrics.update({"exported_sid_length": 4, "unique_exported_sids": len(index),
                        "exported_collision_rate": 0.0,
                        "used_suffix_codes": metrics["max_collision_group_size"]})
    metrics["encoding_seconds"] = encoding_seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths["codes.npy"], codes, allow_pickle=False)
    write_json(paths["item_ids.json"], ids)
    model.save(paths["simhash.npz"])
    write_json(paths["metrics.json"], metrics)
    if index is not None:
        write_json(paths["index.json"], index)
    manifest = {
        "schema_version": 1,
        "method": "FLASH Stage I / SimHash",
        "paper": "https://arxiv.org/abs/2610.07402",
        "scope": "SID tokenization only; no parallel recommender or semantic alignment training",
        "format": args.format,
        "input_embeddings": str(args.embeddings.resolve()),
        "shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "num_codebooks": model.num_codebooks,
        "codebook_size": model.codebook_size,
        "seed": model.seed,
        "code_base": 0,
        "normalization": "none",
        "projection_sha256": hashlib.sha256(model.planes.tobytes(order="C")).hexdigest(),
        "item_ids_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "item_id_alignment": "explicit row list" if args.item_ids else "row i maps to string ID i (0-based)",
        "collision_resolution": "lexicographic item-ID rank within each raw SID" if index is not None else "none",
        "catalog_stability": "Suffix ranks are stable for the same catalog, but may change when items are added or removed",
        "numpy_version": np.__version__,
        "files": {k: v.name for k, v in paths.items()},
    }
    write_json(paths["manifest.json"], manifest)
    print(f"Generated {len(codes)} items, {model.num_codebooks} semantic codes, K={model.codebook_size}")
    print(f"Raw collision rate: {metrics['raw_collision_rate']:.6%}; largest bucket: {metrics['max_collision_group_size']}")
    print(f"Output: {args.output_dir.resolve()}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
