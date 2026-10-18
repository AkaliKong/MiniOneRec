"""Generate synthetic catalog examples and exercise real MiniOneRec conversion.

Run from the repository root with --output-dir pointing to a new directory.
Synthetic data and logs stay there; no language-model weights are loaded.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True,
                    help="new directory for generated synthetic data and logs")
OUTPUT = parser.parse_args().output_dir
if OUTPUT.exists():
    parser.error("--output-dir must not already exist; choose a fresh directory")
DEMO = OUTPUT / "demo"
INPUT = DEMO / "input"
LOGS = OUTPUT / "verification"
INPUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261018)
embeddings = rng.normal(size=(10240, 16)).astype(np.float32)
embeddings[1:4] = embeddings[0]
ids = [str(10000 + 3 * row) for row in rng.permutation(len(embeddings))]
np.save(INPUT / "Demo.emb.npy", embeddings, allow_pickle=False)
(INPUT / "Demo.item_ids.json").write_text(json.dumps(ids))
(INPUT / "Demo.item.json").write_text(json.dumps({item: {"title": "Synthetic item " + item} for item in ids}))
for split in ("train", "valid", "test"):
    lines = ["user_id\titem_id_list\titem_id"]
    for user in range(12):
        a, b, c = rng.choice(ids, 3, replace=False)
        lines.append(f"{user}\t{a} {b}\t{c}")
    (INPUT / f"Demo.{split}.inter").write_text("\n".join(lines) + "\n")

configs = [
    ("prq_3x256", "prq_kmeans.py", ["--codebook-sizes", "256", "256", "256", "--n-iter", "5"]),
    ("rq_opq_3tokens", "rq_opq.py", ["--rq-levels", "1", "--rq-niter", "5", "--opq-niter", "3", "--pq-niter", "5"]),
    ("rq_opq_5tokens", "rq_opq.py", ["--rq-levels", "3", "--rq-niter", "5", "--opq-niter", "3", "--pq-niter", "5"]),
]
results = []
upstream = ROOT
tree = ast.parse((upstream / "sft.py").read_text())
extender_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenExtender")
namespace = {"os": __import__("os"), "json": json}
exec(compile(ast.Module(body=[extender_class], type_ignores=[]), "sft.py:TokenExtender", "exec"), namespace)


def run(command, log_name):
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    (LOGS / log_name).write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"command failed; see {LOGS / log_name}")
    return time.perf_counter() - start


for name, entry, flags in configs:
    directory = DEMO / name
    command = [sys.executable, str(ROOT / "sid_generation" / entry), "fit", "--data-path", str(INPUT / "Demo.emb.npy"),
               "--dataset-name", "Demo", "--output-dir", str(directory), "--item-ids", str(INPUT / "Demo.item_ids.json"),
               "--dataset-dir", str(INPUT)] + flags
    elapsed = run(command, name + ".log")
    encoded_directory = DEMO / (name + "_reloaded")
    run([sys.executable, str(ROOT / "sid_generation" / entry), "encode", "--data-path", str(INPUT / "Demo.emb.npy"),
         "--dataset-name", "Demo", "--output-dir", str(encoded_directory), "--item-ids", str(INPUT / "Demo.item_ids.json"),
         "--model-path", str(directory / "Demo.tokenizer.npz")], name + "_reload.log")
    index = json.loads((directory / "Demo.index.json").read_text())
    reloaded = json.loads((encoded_directory / "Demo.index.json").read_text())
    assert index == reloaded
    assert set(index) == set(ids)
    assert len({tuple(sid) for sid in index.values()}) == len(ids)
    assert np.array_equal(np.load(directory / "Demo.codes.npy"), np.load(encoded_directory / "Demo.codes.npy"))
    extender = namespace["TokenExtender"](str(directory), "Demo")
    assert set(extender.get_new_tokens()) == {token for sid in index.values() for token in sid}
    converted = DEMO / (name + "_converted")
    run([sys.executable, str(upstream / "convert_dataset.py"), "--data_dir", str(directory), "--dataset_name", "Demo",
         "--output_dir", str(converted)], name + "_conversion.log")
    import pandas as pd
    for split in ("train", "valid", "test"):
        frame = pd.read_csv(converted / split / "Demo_5_2016-10-2018-11.csv")
        assert len(frame) == 12
        assert all(row.item_sid == "".join(index[str(row.item_id)]) for row in frame.itertuples())
    stats = json.loads((directory / "Demo.sid_stats.json").read_text())
    results.append({"name": name, "items": len(ids), "dimension": 16, "synthetic": True,
                    "semantic_tokens": stats["semantic_token_count"], "identity_suffix_appended": stats["identity_suffix_appended"],
                    "raw_collision_rate": stats["raw"]["excess_collision_rate"], "final_collision_rate": stats["final"]["excess_collision_rate"],
                    "checkpoint_and_reordered_id_mapping_verified": True, "real_upstream_conversion_verified": True,
                    "upstream_vocabulary_loader_verified": True, "wall_seconds": elapsed})
    print(json.dumps(results[-1]), flush=True)

result = {"scope": "synthetic CPU functional validation; no recommendation-quality benchmark", "runs": results,
          "qwen_tokenizer_tested": False, "constrained_model_decoding_tested": False, "sft_or_rl_run": False}
(LOGS / "smoke_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

print("Passed all three synthetic catalog workflows; see verification/smoke_results.json.")
