# FLASH Stage I: SimHash semantic IDs

This optional tokenizer implements Stage I of [FLASH](https://arxiv.org/abs/2610.07402)
using NumPy on CPU. It consumes existing item embeddings and does not require
training a quantizer. This is not a reproduction of FLASH's full recommender:
parallel decoding and semantic-alignment training are not included, and no
recommendation-quality improvement is claimed for the existing autoregressive model.

## Generate a MiniOneRec index

From the repository root, using a Python environment with NumPy installed:

```bash
python rq/generate_indices_flash.py --embeddings data/Office/Office.emb-qwen-td.npy --item-meta data/Office/Office.item.json --output-dir data/Office --dataset Office
```

The upstream NumPy requirement is sufficient; no new dependency is needed.
`python -m rq.flash_sid` and `python -m rq.generate_indices_flash` are also supported.
The default configuration uses three semantic code positions, 256 values per
position, seed 2024, and a fourth collision-resolution token. This short SID is a
MiniOneRec adaptation, not FLASH's default 32-position configuration.

Input must be a real numeric `.npy` array shaped `[N, D]`, without a padding row.
Zero embeddings, non-finite values, dimension mismatches, duplicate item IDs and
float32 overflow are rejected. Input is memory-mapped and encoded in batches;
the full code array, collision groups and JSON index still occupy memory
proportional to catalog size.

### Item alignment

`--item-meta` validates that metadata keys are exactly `0..N-1`; it assumes row i
is item i. The default without an ID argument uses the same row-number mapping.
For any other row order, use `--item-ids row_ids.json` with an explicit JSON list
such as `[42, 7, 99]`, meaning rows 0, 1 and 2 correspond to those item IDs.
Do not use metadata insertion order as an embedding-row mapping. The upstream
text embedding script sorts IDs before writing the matrix. The dataset converter
currently parses interaction IDs as integers, so arbitrary strings or IDs with
leading zeros require additional converter changes.

### Output files

For `--dataset Office`, the generator writes:

- `Office.index.json`: item ID to `["<a_...>", "<b_...>", "<c_...>", "<d_...>"]`.
- `Office.codes.npy`: zero-based raw semantic codes, without the suffix.
- `Office.item_ids.json`: the exact row-to-item mapping.
- `Office.simhash.npz`: the fixed float32 projections and versioned configuration.
- `Office.metrics.json`: raw collisions, bucket sizes, code utilization and entropy.
- `Office.manifest.json`: parameters, alignment convention and projection digest.

Each raw-SID collision group is sorted lexicographically by the canonical string
item ID. The fourth token is the item's rank in that group; singleton groups get
`<d_0>`. With sufficient suffix capacity, complete exported IDs are unique.
The default capacity is 256 items per group; overflow raises an error rather than
silently reusing an ID. Set `--suffix-size` explicitly to increase this capacity.
The raw collision rate is `(N - unique raw SIDs) / N`, before adding suffixes.

Suffix ranks are stable under row reordering for the same catalog, but can change
when items are added or removed. Separate exports cannot be concatenated into one
catalog without resolving suffix collisions. This implementation is intended for
offline full-catalog generation, not a persistent online ID registry.

Existing outputs are protected by default; `--overwrite` explicitly replaces
files for that dataset. Keep the exact `Dataset.index.json` naming convention
because SFT derives the dataset name from the filename.

## Continue the MiniOneRec pipeline

Generate the new index alongside the item metadata, then rerun `convert_dataset.py`
to rebuild train/valid/test CSV and item-info files. Use these files and the same
index for a new SFT run, and use that SFT checkpoint's tokenizer for RL/evaluation.
Existing checkpoints trained with different SID assignments are not interchangeable.
The auxiliary datasets now concatenate all SID tokens, preserving the fourth token.
The decoding `prefix_index` values refer to the response prompt, not SID length.

## Long raw codes and projection reuse

To generate the 32-position configuration without a MiniOneRec index or suffix:

```bash
python rq/generate_indices_flash.py --embeddings items.npy --output-dir output/flash_raw --dataset Items --format raw
```

Raw mode defaults to `--num-codebooks 32 --codebook-size 256`. It reports collisions
without removing them. MiniOneRec mode requires exactly three semantic positions.
Both modes use seed 2024 by default. Gaussian projections are drawn in float64,
then stored as float32, using NumPy's default RNG; sign tests use strict `> 0`
and least-significant-bit-first packing. Inputs are not centered or normalized.
Codebook sizes must be powers of two from 2 through 2**30. The output codes are
zero-based, whereas the paper's equation presents one-based codes.

Reuse the saved projection with the same embedding model and dimension:

```bash
python rq/generate_indices_flash.py --embeddings new_items.npy --item-ids new_ids.json --model-in output/Office.simhash.npz --output-dir output/new --dataset New
```

Explicit seed/codebook arguments conflicting with a saved model are rejected.
Projection reuse stabilizes raw codes; it does not provide cross-export uniqueness
of the catalog-local suffix. Floating-point values extremely close to a hyperplane
can yield different sign bits across BLAS implementations or hardware.

## Tests and references

```bash
python -B -m unittest discover -s tests -p "test_flash_sid*.py" -v
```

Tests cover the independent scalar formula, bit boundaries, saved projections,
batching, input validation, collision handling, item alignment, output protection,
all seven auxiliary dataset SID expressions, and script/module entry points.
They use synthetic data; full SFT/RL and downstream quality are not evaluated.

This is an independent implementation of the construction in
[FLASH, section 2.1](https://arxiv.org/html/2610.07402v1#S2.SS1).
The [reference tokenizer](https://github.com/KevinC2015/Flash/blob/main/genrec/models/Flash/tokenizer.py)
was consulted for RNG and bit-order conventions; its full training code is not
vendored here. FLASH's repository declares CC-BY-NC 4.0 licensing.
