"""Command-line catalog generation. All outputs live in the selected directory."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import tempfile
import time

import numpy as np

try:
    from .sid_common import build_indexes, canonical_item_ids, vectors
    from .sid_publish import generated_names, publish_catalog, validate_publication
except ImportError:
    from sid_common import build_indexes, canonical_item_ids, vectors
    from sid_publish import generated_names, publish_catalog, validate_publication


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def catalog_inputs(args, count):
    if args.item_ids:
        ids = json.loads(Path(args.item_ids).read_text(encoding="utf-8"))
        if not isinstance(ids, list):
            raise ValueError("--item-ids must be a JSON list in embedding-row order")
    else:
        ids = list(range(count))
    ids = canonical_item_ids(ids, count)
    copies = []
    if args.dataset_dir:
        source = Path(args.dataset_dir)
        metadata_path = source / f"{args.dataset_name}.item.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or set(metadata) != set(ids):
            raise ValueError("item metadata keys and embedding-row IDs must match exactly; supply --item-ids when needed")
        if any(not key.isascii() or not key.isdecimal() or str(int(key)) != key for key in ids):
            raise ValueError("the upstream converter requires canonical non-negative integer item IDs")
        copies.append(metadata_path)
        known = set(ids)
        for split in ("train", "valid", "test"):
            split_path = source / f"{args.dataset_name}.{split}.inter"
            if not split_path.exists():
                continue
            with split_path.open(encoding="utf-8") as handle:
                next(handle, None)
                for line_number, line in enumerate(handle, start=2):
                    if not line.strip():
                        continue
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) != 3:
                        raise ValueError(f"{split_path.name}:{line_number}: expected three tab-separated fields")
                    references = fields[1].split() + [fields[2].strip()]
                    if any(item not in known for item in references):
                        raise ValueError(f"{split_path.name}:{line_number}: interaction references an unknown item ID")
            copies.append(split_path)
    return ids, copies


def make_parser(algorithm):
    description = ("Sequential projection-residual PRQ-KMeans SID generation" if algorithm == "prq_kmeans"
                   else "Serial RQ + independent OPQ subcodes, exported as ordered MiniOneRec SIDs")
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("fit", "encode"):
        command = subparsers.add_parser(name)
        command.add_argument("--data-path", required=True, help="real-valued (N,D) .npy embeddings")
        command.add_argument("--dataset-name", "--dataset", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--item-ids", help="JSON list mapping embedding rows to item IDs; default: 0..N-1")
        command.add_argument("--dataset-dir", help="validate and copy Dataset.item.json and available .inter splits")
        command.add_argument("--collision-policy", choices=("append", "error", "allow"), default="append")
        command.add_argument("--overwrite", action="store_true", help="publish a complete replacement; remove obsolete owned dataset files")
        if name == "encode":
            command.add_argument("--model-path", required=True)
        else:
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--batch-size", type=int, default=2048 if algorithm == "prq_kmeans" else 4096)
            if algorithm == "prq_kmeans":
                command.add_argument("--codebook-sizes", type=int, nargs="+", default=[256, 256, 256])
                command.add_argument("--top-k", type=int, default=2)
                command.add_argument("--beta", type=float, default=15.0)
                command.add_argument("--n-iter", type=int, default=30)
            else:
                command.add_argument("--rq-levels", type=int, default=3)
                command.add_argument("--codebook-size", type=int, default=256)
                command.add_argument("--opq-subspaces", type=int, default=2)
                command.add_argument("--nbits", type=int, default=8)
                command.add_argument("--rq-niter", type=int, default=25)
                command.add_argument("--opq-niter", type=int, default=20)
                command.add_argument("--pq-niter", type=int, default=25)
                command.add_argument("--threads", type=int, default=1)
    return parser


def generate(args, algorithm, model_class):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset_name):
        raise ValueError("dataset name must use letters, digits, underscores or hyphens, without dots or separators")
    data_path = Path(args.data_path)
    data = vectors(np.load(data_path, mmap_mode="r", allow_pickle=False))
    ids, copies = catalog_inputs(args, len(data))
    output = Path(args.output_dir)
    if args.dataset_dir and output.resolve() == Path(args.dataset_dir).resolve():
        raise ValueError("use a separate output directory to preserve the original dataset")
    dataset = args.dataset_name
    names = generated_names(dataset) + [source.name for source in copies]
    protected = {data_path.resolve()}
    if args.item_ids:
        protected.add(Path(args.item_ids).resolve())
    if args.operation == "encode":
        protected.add(Path(args.model_path).resolve())
    protected.update(source.resolve() for source in copies)
    validate_publication(output, dataset, names, args.overwrite, protected)
    start = time.perf_counter()
    if args.operation == "encode":
        model = model_class.load(args.model_path)
        fit_seconds = 0.0
    else:
        options = vars(args)
        keys = ("codebook_sizes", "top_k", "beta", "n_iter", "batch_size", "seed") if algorithm == "prq_kmeans" else (
            "rq_levels", "codebook_size", "opq_subspaces", "nbits", "rq_niter", "opq_niter", "pq_niter", "batch_size", "seed", "threads")
        model = model_class(**{key: options[key] for key in keys}).fit(data)
        fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    codes = model.encode(data)
    encode_seconds = time.perf_counter() - start
    raw, final, statistics = build_indexes(ids, codes, model.capacities, args.collision_policy)
    statistics.update({"algorithm": algorithm, "fit_seconds": fit_seconds, "encode_seconds": encode_seconds})
    manifest = {
        "algorithm": algorithm,
        "encoding_revision": getattr(model, "encoding_revision", 1),
        "operation": args.operation,
        "configuration": model.configuration(),
        "input": {"path": str(data_path.resolve()), "sha256": sha256_file(data_path),
                  "items": len(data), "dimension": data.shape[1], "computation_dtype": "float32"},
        "item_id_source": str(Path(args.item_ids).resolve()) if args.item_ids else "zero-based embedding row indices",
        "item_id_source_sha256": sha256_file(args.item_ids) if args.item_ids else None,
        "copied_dataset_inputs": [{"file": p.name, "sha256": sha256_file(p)} for p in copies],
        "source_model_sha256": sha256_file(args.model_path) if args.operation == "encode" else None,
        "fit_scope": "supplied embedding catalog only; no interaction supervision" if args.operation == "fit" else "frozen loaded model; catalog fitting not performed",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "collision_policy": args.collision_policy,
        "outputs": names,
        "publication": "complete directory; atomic create or exchange; obsolete owned companions removed",
    }
    if algorithm == "rq_opq":
        import faiss
        manifest["faiss_version"] = faiss.__version__
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage only after validation and collision checks, so an invalid export
    # does not leave behind a plausible but incomplete index.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{dataset}-sid-") as temporary:
        stage = Path(temporary)
        json_write(stage / f"{dataset}.index.json", final)
        json_write(stage / f"{dataset}.raw.index.json", raw)
        json_write(stage / f"{dataset}.item_ids.json", ids)
        json_write(stage / f"{dataset}.sid_stats.json", statistics)
        np.save(stage / f"{dataset}.codes.npy", codes, allow_pickle=False)
        model.save(stage / f"{dataset}.tokenizer.npz")
        for source in copies:
            shutil.copyfile(source, stage / source.name)
        manifest["artifact_sha256"] = {name: sha256_file(stage / name) for name in names if not name.endswith(".manifest.json")}
        json_write(stage / f"{dataset}.manifest.json", manifest)
        publish_catalog(stage, output, dataset, names, args.overwrite, protected)
    summary = {"algorithm": algorithm, "index": str((output / f"{dataset}.index.json").resolve()),
               "items": len(ids), "semantic_tokens": codes.shape[1],
               "identity_suffix_appended": statistics["identity_suffix_appended"],
               "raw_collision_rate": statistics["raw"]["excess_collision_rate"],
               "final_collision_rate": statistics["final"]["excess_collision_rate"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_cli(algorithm, model_class):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = make_parser(algorithm)
    args = parser.parse_args()
    try:
        generate(args, algorithm, model_class)
    except (ValueError, RuntimeError, ImportError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
