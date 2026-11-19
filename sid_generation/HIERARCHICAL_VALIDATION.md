# Hierarchical K-Means validation

## macOS ARM64 results

Validated on macOS ARM64 with native ARM64 CPython 3.12.4.
The builder ran with NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.7.2,
k-means-constrained 0.9.0, OR-Tools 9.15.6755 and threadpoolctl 3.8.0.
Consumer and regression tests used pytest 9.2.0, pandas 2.3.3, PyTorch
2.9.1 (CPU), transformers 4.57.6, tokenizers 0.22.2 and faiss-cpu 1.12.0.

| Test selection | Passed | Failed | Skipped | Errors (including teardown) |
| --- | ---: | ---: | ---: | ---: |
| Hierarchical K-Means tests | 171 | 0 | 0 | 0 |
| Upstream regressions on the official baseline at `0cbd94d` | 81 | 0 | 0 | 0 |
| Same upstream regressions with Hierarchical K-Means | 81 | 0 | 0 | 0 |

The two upstream runs used the same interpreter, dependencies and test selection.
All 81 test IDs and outcomes matched, including all 15 FAISS-dependent cases
and all eight directory publication tests.

| Area | What the generator tests check |
| --- | --- |
| Strategies | Capacity bounds, a reference global sort for Greedy including ties, per-node Hybrid thresholds and terminal nodes |
| Catalog identity | Explicit item IDs including zero, reordered rows, unchanged SID assignment, full coverage and prefix-free paths |
| Persistence and publication | All three strategies through fit, save, load, encode and diagnose, format and configuration checks, duplicate JSON keys, damaged or inconsistent artifacts, shared locking and overwrite refusal |
| MiniOneRec consumers | Real CSV conversion, TokenExtender registration with a local fast tokenizer, Dataset labels and item mappings, evaluator prefix callbacks and CPU logits masks through newline, EOS and invalid prefixes |

## Real catalog rebuild

Balanced, Greedy and Hybrid each fitted a tree on all 3,686 items in
`Industrial_and_Scientific` using 2,560-dimensional embeddings, K=10 and seed=0.
For each strategy, repeated builds on the same Mac produced byte-identical
tree JSON and SID index JSON. Loading the saved tree and encoding the catalog
in original and reversed row order, with IDs reordered alongside the embeddings,
reproduced both files. For each fit and re-encoding, `diagnose` verified all
four artifact hashes in the manifest. The CLI also loaded and diagnosed existing
format-2 trees for all three strategies and reproduced their indexes with reversed input rows.

| Strategy | Constrained splits | Greedy splits |
| --- | ---: | ---: |
| Balanced | 111 | 0 |
| Greedy | 0 | 111 |
| Hybrid | 110 | 1 |

All three trees met the per-node capacity bounds and covered every item once.
Each produced 3,686 unique four-token SIDs, with 1,110 node tokens, 3,686 leaf
tokens and zero SID or prefix collisions. Hybrid used Greedy only at the root.
In separate boundary checks with the default threshold, a 2,000-item root used
constrained K-Means. A 2,001-item root used Greedy, then all 16 child
nodes used constrained K-Means for their splits.

The consumer tests used small local catalogs and an offline tokenizer. The
validation covered SID construction and consumer compatibility. It did not
include language-model generation, SFT, RL or recommendation quality evaluation.

## Reproducing the tests

Use the Python 3.12 environment from the [installation guide](HIERARCHICAL_KMEANS.md#installation).
From the repository root, install the test dependencies and run the specialized suite.
FAISS is required only for the full upstream comparison.

For a Linux CPU-only environment, install the pinned PyTorch wheel before
the test requirements with
`python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu`.
The results above use PyTorch 2.9.1 on macOS. See version 2.9.1 in the
[official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

Run the specialized suite and the upstream suite in separate Python processes,
as shown below. In this environment, importing PyTorch after a FAISS search
in the same process triggered an OpenMP runtime conflict in both checkouts.

```bash
python -m pip install -r sid_generation/requirements-hierarchical-test.txt -r sid_generation/requirements-test.txt
python -m pytest -q -p no:cacheprovider -o "python_files=test_hierarchical_*.py" sid_generation/tests
```

For the upstream comparison, use Bash and keep the same interpreter active in
both checkouts. The commands below create a baseline in an unused sibling directory.

```bash
git worktree add --detach ../minionerec-baseline 0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1
run_upstream() {
  python -m pytest -q -p no:cacheprovider --tb=short -ra \
    sid_generation/tests/test_sid_generation.py sid_generation/tests/test_rq_regressions.py \
    sid_generation/tests/test_publication.py sid_generation/tests/test_prq_regressions.py \
    sid_generation/tests/test_minionerec_integration.py tests/test_flash_sid_simhash.py \
    tests/test_flash_sid_export.py tests/test_flash_sid_integration.py
}
run_upstream
(cd ../minionerec-baseline && run_upstream)
```
