# Hierarchical K-Means semantic IDs

This CPU builder turns a catalog of item embeddings into MiniOneRec's
`Dataset.index.json`. It fits a tree without model weights or neural-network
training. Choose a strategy for splitting each node with more than K items.

| `--method` | How the node is split |
| --- | --- |
| `balanced` (default) | SEATER-style constrained K-Means with cluster sizes between `floor(N/K)` and `floor(N/K)+1`. |
| `greedy` | Fit ordinary K-Means centroids, then sort all item–centroid pairs by squared distance, item position and centroid index. Accept a pair if its item is unassigned and the centroid has space. |
| `hybrid` | Use Greedy if this node has more than `--greedy-threshold` items, and constrained K-Means otherwise. At the default threshold, a node of exactly 2,000 items uses constrained K-Means. |

Each child is clustered on its original vectors in float32, without residual
subtraction or normalization. With `N = qK + r`, exactly r children receive q+1
items and the rest receive q. Greedy gives the extra slots to the first r fitted
centroids and never refits them after assignment. Nodes with at most K items
are terminal. Greedy stores and sorts N×K distances for each split, which can
limit the size of the largest node that fits in memory.

## Installation

All shell examples below use Bash syntax for Linux or macOS. On Windows, use
WSL or Git Bash. PowerShell cannot run these examples directly. With native
Windows Python in Git Bash, the activation script is under `Scripts/`.

From the repository root, create a Python 3.12 environment and install the builder.

```bash
python -m venv .venv-hierarchical
source .venv-hierarchical/bin/activate
python -m pip install -r sid_generation/requirements-hierarchical.txt
```

## Quick start

This example creates a five-item catalog with noncontiguous IDs and a sample
interaction for each split. Run the blocks in order from the repository root.
Use fresh input and export directories for a new run.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

src = Path("outputs/hierarchical/example-input")
src.mkdir(parents=True)
ids = [42, 0, 7, 100, 3]  # Same order as the embedding rows below
np.save(src / "embeddings.npy", np.array([[8, 9], [0, 1], [2, 2], [9, 8], [4, 5]], dtype="float32"))
(src / "item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
items = {str(i): {"title": f"Item {i}"} for i in ids}
(src / "MyCatalog.item.json").write_text(json.dumps(items), encoding="utf-8")
header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
for split, row in [("train", "1\t42 0\t7"), ("valid", "1\t42 0 7\t100"), ("test", "1\t0 7 100\t3")]:
    (src / f"MyCatalog.{split}.inter").write_text(header + row + "\n", encoding="utf-8")
PY
```

```bash
src=outputs/hierarchical/example-input
dataset=MyCatalog
out=outputs/hierarchical/$dataset

python -m sid_generation.hierarchical_cli fit --branch-factor 2 \
  --data-path "$src/embeddings.npy" --item-ids "$src/item_ids.json" \
  --dataset-dir "$src" --dataset-name "$dataset" --output-dir "$out"

python -m sid_generation.hierarchical_cli diagnose \
  --model-path "$out/$dataset.tree.json" --manifest-path "$out/$dataset.manifest.json"

python -m sid_generation.hierarchical_cli encode \
  --data-path "$src/embeddings.npy" --item-ids "$src/item_ids.json" \
  --model-path "$out/$dataset.tree.json" --dataset-name "$dataset" \
  --output-dir "$out-reencoded"

python convert_dataset.py --data_dir "$out" --dataset_name "$dataset" \
  --output_dir "$out-csv"
```

`fit` and `encode` publish only to new directories. `diagnose` checks the tree.
Passing the manifest also verifies the exported files' SHA256 hashes and
checks the index and row mapping against the tree. `encode` requires the same
complete catalog and vectors. Reordering rows and their IDs together is fine.
Adding, removing or changing an item requires a new fit.

## Using your own catalog

- **Embeddings.** Supply a finite, real-valued NumPy `.npy` matrix of shape
  `(items, dimensions)`. The CLI loads it with pickle disabled.
- **Item IDs.** `--item-ids` is a required JSON array of unique strings or
  integers. Entry `i` identifies embedding row `i`. Save the IDs in row order
  when producing the vectors, including item `0` if present. Metadata key
  iteration cannot establish the row order. Only for a producer verified to
  write IDs `0` through `N-1` consecutively, generate the mapping in the input
  directory with the command below. Substitute your embedding filename for `embeddings.npy`.
  `python -c "import json, numpy as np; from pathlib import Path; n = np.load('embeddings.npy', mmap_mode='r', allow_pickle=False).shape[0]; Path('item_ids.json').write_text(json.dumps(list(range(n))), encoding='utf-8')"`.
- **Metadata and interactions.** For conversion, pass `--dataset-dir` with
  `Dataset.item.json` keyed by the same IDs and containing item titles. Supply
  `Dataset.train.inter`, `Dataset.valid.inter` and optionally `Dataset.test.inter`
  in the tab-separated format above. This path requires canonical nonnegative
  integer IDs. The builder checks every referenced item and copies the
  companions into its export. For standalone SID construction omit `--dataset-dir`.

## MiniOneRec integration

For the official `Industrial_and_Scientific` catalog, pass the embedding file
directly to both `fit` and `encode` with
`--data-path data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy`.
You can use the file under its original name. In the quick start commands,
set `src` to the directory containing your row-aligned `item_ids.json`,
`Industrial_and_Scientific.item.json` and the interaction splits. Set
`dataset=Industrial_and_Scientific` before defining `out`. Omit
`--branch-factor 2` for the default K=10. Convert interactions after fitting
and before SFT.

Set these arguments in the `torchrun` call inside [`sft.sh`](../sft.sh), which
does not forward arguments passed to `bash sft.sh`. Configure the base model
and training output there as well.

| [`sft.py`](../sft.py) argument | Value for the Industrial catalog |
| --- | --- |
| `--category` | `Industrial_and_Scientific` |
| `--train_file` | `outputs/hierarchical/Industrial_and_Scientific-csv/train/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| `--eval_file` | `outputs/hierarchical/Industrial_and_Scientific-csv/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv` |
| `--sid_index_path` | `outputs/hierarchical/Industrial_and_Scientific/Industrial_and_Scientific.index.json` |
| `--item_meta_path` | `outputs/hierarchical/Industrial_and_Scientific/Industrial_and_Scientific.item.json` |

`MyCatalog` is a generic builder example. SFT accepts only
`Industrial_and_Scientific`, `Office_Products`, `Toys_and_Games`, `Sports` and
`Books`. Other categories require a separate change to the upstream SFT
mapping. The converter's `--category` sets the output filename prefix and
defaults to `--dataset_name`. It does not register a category with SFT. This
contribution did not run SFT or RL. See the [MiniOneRec SFT guide](../README.md#4-sft).

For RL, use the same index and metadata, the converted train, valid and info
files, and a checkpoint trained on these SIDs. Evaluation uses the converted
test and info files.

## Output and vocabulary

| File | Contents |
| --- | --- |
| `Dataset.index.json` | Item ID to the complete ordered list of SID tokens |
| `Dataset.tree.json` | Fitted topology, configuration and catalog fingerprint |
| `Dataset.item_ids.json` | Item IDs in original input-row order |
| `Dataset.tree_stats.json` | Tree depth, SID lengths, cluster sizes, vocabulary and timing |
| `Dataset.manifest.json` | Algorithm, configuration, backend, dependency versions and artifact hashes |

Only format 2 trees and manifests (`hierarchical_kmeans`, version `2.0.0`) are supported.
Rows are sorted by canonical string ID before fitting. Children are ordered
by their smallest ID after splitting. Each non-root cluster with multiple
items gets a unique `<node_ID>` token. Each item gets a unique `<leaf_RANK>`
token. The path ends at that leaf, so complete SIDs are unique and prefix-free
even when their lengths differ.

On macOS ARM64, each strategy rebuilt the 3,686-item `Industrial_and_Scientific`
catalog with 1,110 node tokens and 3,686 leaf tokens (4,796 vocabulary entries).
For hidden width d this is 4,796 × d embedding parameters with tied input and
output weights, or twice that with separate matrices, before accounting for
spare vocabulary rows.
Uniqueness does not measure recommendation quality. For catalogs with mixed
SID lengths, allow room for the longest path, newline and EOS during generation.
MiniOneRec's default beam-search `length_penalty=0.0` can favor shorter paths.

## Fit options

| Option | Default | Applies to |
| --- | --- | --- |
| `--branch-factor`, `--seed` | 10, 0 | Children per split (K ≥ 2) and random seed for both solvers |
| `--n-init`, `--max-iter`, `--n-jobs` | 10, 300, 1 | Constrained K-Means restarts, iteration limit and parallel jobs |
| `--greedy-n-init`, `--greedy-max-iter` | 5, 20 | Ordinary K-Means restarts and iteration limit for Greedy splits |
| `--greedy-threshold` | 2000 | Hybrid node-size threshold |
| `--tol` | 1e-4 | Convergence tolerance for both solvers |

BLAS/OpenMP threads are limited to one during fitting. Reproducibility assumes
the same numerical environment. This implementation uses scikit-learn's CPU
Lloyd solver for every Greedy split. Re-SEATER switches to FAISS on larger
nodes, so its results may differ. See the [validation report](HIERARCHICAL_VALIDATION.md)
for the macOS ARM64 test commands and catalog results.

## References

- Si et al., *Generative Retrieval with Semantic Tree-Structured Identifiers
  and Contrastive Learning*, SIGIR-AP 2024. The Balanced strategy follows
  Algorithm 1. [Paper](https://doi.org/10.1145/3673791.3698408) and
  [SEATER source](https://github.com/Ethan00Si/SEATER_Generative_Retrieval/tree/b12cd29585720f891b1a8f41dacec5e0554fac6f).
- Valeau et al., *Efficient Optimization of Hierarchical Identifiers for
  Generative Recommendation*. Greedy's single assignment pass and Hybrid's
  node-local threshold follow the authors' implementation.
  [Paper](https://arxiv.org/html/2512.18434v2) and
  [Re-SEATER builder](https://github.com/joshrosie/re-seater/blob/32235415e5abc1fa97d0516199422b5d9a34471b/SEATER_Generative_Retrieval/utils/build_tree_speedup.py).

All three strategies were implemented independently. At
the referenced commits SEATER has no explicit software license and Re-SEATER
uses [Apache-2.0](https://github.com/joshrosie/re-seater/blob/32235415e5abc1fa97d0516199422b5d9a34471b/LICENSE).
No source code was copied or ported from either project. MiniOneRec retains
its [Apache-2.0 license](../LICENSE).
