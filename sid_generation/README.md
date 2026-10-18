# PRQ-KMeans and RQ-OPQ SID generators

This package adds two optional CPU builders for MiniOneRec semantic IDs (SIDs).
Both consume item embeddings and export the existing `Dataset.index.json`
format. They also save a reusable quantizer, row-to-item mapping, collision
statistics and a manifest with artifact hashes. No FLASH or SimHash is used.

| Builder | Offline encoding | Default semantic sequence |
| --- | --- | --- |
| PRQ-KMeans | Sequential projection residuals; each level depends on the preceding level | Three tokens, `<a_...><b_...><c_...>` |
| RQ-OPQ | Sequential L2 residual quantization followed by independent OPQ/PQ subspace codes | Three RQ tokens followed by two OPQ tokens |

RQ-OPQ is a hybrid quantizer: its OPQ subcodes are not additional sequential
residual levels. Both builders export an ordered sequence that a recommendation
model can generate autoregressively. This implementation is an item-only L2
adaptation of RQ-OPQ, not a reproduction of the entire OneSearch system.

## Installation

Run commands from the repository root. Python 3.11 or 3.12 is recommended.
The SID builders do not need PyTorch, Transformers, a GPU or language-model
weights. MiniOneRec training still needs the project's regular dependencies.

```bash
python3 -m venv .venv-sid
source .venv-sid/bin/activate
python -m pip install -r sid_generation/requirements-faiss.txt
```

For PRQ-KMeans alone, install `sid_generation/requirements.txt` instead.
The optional FAISS requirements pin `faiss-cpu==1.12.0`; use an isolated CPU
environment rather than replacing a working GPU FAISS installation.

## Inputs

- `--data-path`: a real, finite `(N, D)` NumPy `.npy` array. Values must be
  representable as float32. Each row is one item embedding.
- `--item-ids`: an optional JSON list in embedding-row order, for example
  `["90", "10", "30", "20"]`. Without it, IDs are `0` through `N-1`.
- `--dataset-name`: the filename stem, such as `Industrial_and_Scientific`.
- `--dataset-dir`: an optional directory containing `Dataset.item.json` and
  available `Dataset.train.inter`, `Dataset.valid.inter`, `Dataset.test.inter`.
  Metadata keys must exactly cover the embedding IDs. Every interaction ID is
  checked before fitting. Validated companion files are copied into the export.

Use explicit IDs whenever embedding rows do not follow the default numbering.
The MiniOneRec converter requires canonical nonnegative integer IDs, so the
`--dataset-dir` integration mode enforces that restriction.

PRQ needs at least as many fitting rows as its largest codebook. RQ-OPQ needs
at least `max(codebook_size, 2**nbits)` rows, and `D` must be divisible by
`opq_subspaces`. These are minimum feasibility checks, not a guarantee of good
clustering; FAISS can still warn about insufficient training data.

## Fit PRQ-KMeans

Replace `/path/to/...` with your embedding and processed dataset paths.

```bash
python sid_generation/prq_kmeans.py fit \
  --data-path /path/to/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --dataset-dir /path/to/Industrial_and_Scientific \
  --output-dir outputs/sid/prq/Industrial_and_Scientific \
  --codebook-sizes 256 256 256 \
  --top-k 2 --beta 15 --n-iter 30 --seed 42
```

The implementation normalizes inputs, removes their projection onto the global
mean direction, updates centroids using top-k soft assignments, assigns hard
codes by cosine similarity, and passes normalized projection residuals to the
next level. This is not ordinary centroid subtraction. Empty centroids retain
their previous value; zero directions and near-zero residuals have deterministic
behavior, and exact similarity ties select the lowest centroid index.

## Fit RQ-OPQ

The default configuration exports five semantic tokens:

```bash
python sid_generation/rq_opq.py fit \
  --data-path /path/to/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --dataset-dir /path/to/Industrial_and_Scientific \
  --output-dir outputs/sid/rq_opq/Industrial_and_Scientific \
  --rq-levels 3 --codebook-size 256 \
  --opq-subspaces 2 --nbits 8 \
  --rq-niter 25 --opq-niter 20 --pq-niter 25 \
  --threads 1 --seed 42
```

For a three-token comparison, use `--rq-levels 1` with two OPQ subspaces and a
different output directory. This is a budget adaptation, not the original
OneSearch configuration. OPQ supports 1-8 bits per subcode, including fields
that cross byte boundaries. The OPQ training PQ uses the selected bit width.
The frozen transform and final PQ encode rows in input order.

Use `fit --help` and `encode --help` for all options. The corresponding `.sh`
launchers accept the same arguments and an optional `PYTHON_BIN` environment
variable.

## Reuse a saved quantizer

```bash
python sid_generation/prq_kmeans.py encode \
  --data-path /path/to/Industrial_and_Scientific.emb-qwen-td.npy \
  --dataset-name Industrial_and_Scientific \
  --model-path outputs/sid/prq/Industrial_and_Scientific/Industrial_and_Scientific.tokenizer.npz \
  --output-dir outputs/sid/prq_reencoded/Industrial_and_Scientific
```

RQ-OPQ uses the same interface through `rq_opq.py`. Supply the corresponding
`--item-ids` again for nondefault row mappings. `encode` uses the fitted model
without refitting its mean direction, codebooks or rotation. Add `--dataset-dir`
if the new export must include metadata and interaction splits.

```python
from sid_generation import PRQKMeans, RQOPQ
from sid_generation.sid_common import build_indexes

model = PRQKMeans(codebook_sizes=(256, 256, 256), seed=42).fit(embeddings)
codes = model.encode(embeddings)
raw_index, final_index, statistics = build_indexes(item_ids, codes, model.capacities)
model.save("outputs/sid/model.npz")
```

## Collisions and exported files

The default `--collision-policy append` appends an identity token to **every
item** when any full semantic-code collision occurs. Within each collision
group, suffixes are ranks of lexicographically sorted item IDs. Thus three
semantic tokens can become four, and five can become six. The suffix supplies
identity, not extra semantics; include its length and vocabulary in comparisons.
With no collisions, no suffix is added.

`error` rejects any raw collision without publishing an index. `allow` exports
colliding codes and requires a separate item-level disambiguation/evaluation
protocol; it does not solve the existing evaluator's identity ambiguity.

Export the complete catalog together. Do not independently assign suffixes to
shards and concatenate their indexes. Suffixes are invariant to row order for
fixed IDs and raw codes, but can change when catalog membership changes. Store
the final index as part of the dataset version. The model checkpoint alone does
not manage incremental item identities.

| Filename | Content |
| --- | --- |
| `Dataset.index.json` | Final item-ID to token-list mapping for MiniOneRec |
| `Dataset.raw.index.json` | Semantic mapping before identity suffixes |
| `Dataset.codes.npy` | Integer semantic codes in input-row order |
| `Dataset.item_ids.json` | Item IDs corresponding to those rows |
| `Dataset.tokenizer.npz` | Quantizer state for later encoding |
| `Dataset.sid_stats.json` | Raw/final collisions, lengths, vocabulary and occupancy |
| `Dataset.manifest.json` | Configuration, versions, input hashes and output hashes |

Use the standard `.index.json` filename for training. The `.raw.index.json`
file is for analysis, not the `TokenExtender` index input.

Exports refuse existing reserved filenames unless `--overwrite` is supplied.
A complete staged directory is published in one create or exchange operation.
Replacement removes obsolete companion files owned by the previous manifest,
preserves unrelated files, and refuses ambiguous untracked companions. A
persistent lock beside the output directory coordinates this tool's writers.

Replacing an existing directory requires macOS `renamex_np(RENAME_SWAP)` or
Linux `renameat2(RENAME_EXCHANGE)` support. Unsupported platforms/filesystems
retain the old export and report an error; use a fresh output directory.
Directory staging needs additional disk space. This protects against process
failure, not power-loss durability or arbitrary external edits. Readers opening
multiple files should pin the directory or verify manifest hashes to detect a
version switch between reads.

## MiniOneRec integration

This branch preserves upstream's complete-SID concatenation in `data.py` and
accepts any nonempty SID, including one- or two-token configurations. It also
updates `ts_rec_data.py` to consume the complete SID, including identity suffixes.
Item/title/description tasks and Fusion mappings therefore retain the full identity.
The trainer also constructs independent training and test constraint processors
with their respective beam counts; answer-wrapper offsets are unchanged.

Convert an export containing the metadata and interaction files:

```bash
python convert_dataset.py \
  --dataset_name Industrial_and_Scientific \
  --data_dir outputs/sid/prq/Industrial_and_Scientific \
  --output_dir outputs/sid/converted/prq
```

Point SFT/RL `sid_index_path` at the new `.index.json`, `item_meta_path` at its
matching `.item.json`, and use the newly converted train/validation data.
Rebuild evaluation info mappings and constraint prefixes as well. An export
without `.inter` files only permits metadata/info conversion, not interaction
training. Changing SIDs requires rebuilding the data and retraining; an old-SID
checkpoint is not a model for the new vocabulary.

## Numerical behavior and validation

PRQ scores and the frozen OPQ rotation use fixed float64 dimension reductions.
RQ assignment uses bounded-workspace direct squared differences in float64,
avoiding cancellation from large common offsets. RQ fitting uses a fixed
translation and a shared scalar, then restores centroids to input coordinates.
Inputs that would overflow FAISS intermediates are rejected with a rescaling
message. Loaded RQ-OPQ models are checked for rotation/PQ structure, finite
coefficients, matching dimensions and orthogonality.

These choices address batch-shape-dependent boundary assignments in the tested
environment. They cost CPU time compared with optimized BLAS and do not promise
bitwise equality across NumPy/FAISS versions or hardware. Encoding revision 2
can change boundary codes from revision 1; preserve old indexes and regenerate
complete datasets rather than mixing revisions.

```bash
python -m pip install -r sid_generation/requirements-test.txt
python -m unittest discover -s sid_generation/tests -v
```

Tests use the current repository, with no external checkout or local research
directory. They cover numerical boundaries, order/shard invariance, saved
models, real FAISS bit packing, collisions, failed/overwritten publication,
the actual converter and `TokenExtender`, and complete-SID dataset mappings.
Mapping tests extract repository classes with AST and disable eager tokenization;
they do not validate real Qwen tokenization or model generation.

See [VALIDATION.md](VALIDATION.md) for executed checks and remaining limits.
No recommendation-quality improvement, real-model constrained decoding, SFT or
RL result is claimed by these CPU tests.

## References

- [PRQ-KMeans, Algorithm 1 (v2)](https://arxiv.org/html/2608.24207v2).
- [OneSearch](https://arxiv.org/abs/2509.03236) and its
  [public RQ-OPQ implementation](https://github.com/benchen4395/onesearch-family/blob/1fbb444bad60d0548d022089981ebc4629a1afe9/rq-opq/rq_opq.py).
- [FAISS documentation](https://faiss.ai/).

The builders are independently implemented CPU adaptations. Algorithm references
do not imply reproduction of paper-level recommendation results.
