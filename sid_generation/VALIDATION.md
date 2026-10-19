# SID generator validation

## Executed checks

Checked on 2026-10-19 after synchronizing with upstream `AkaliKong/MiniOneRec`
main commit `1d6dc4f07efb291064e972cba148e406e907b5f4`.

Environment: macOS arm64, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and
faiss-cpu 1.12.0. No language-model weights or GPU were used.

### Automated regressions

```bash
python -m pip install -r sid_generation/requirements-test.txt
python -m unittest discover -s sid_generation/tests -v
```

Result: **50 tests passed, zero failures and zero skips**.

| Area | Checks |
| --- | --- |
| PRQ-KMeans | Projection residuals, global direction removal, deterministic ties, zero/identical vectors, cosine-boundary batch invariance, reordered and sharded encoding, model persistence |
| RQ-OPQ | True L2-nearest assignments with large common offsets, batch boundaries, stable rotation at PQ boundaries, real FAISS 1-8 bit packing, checkpoint structure and orthogonality, invalid/extreme inputs, thread restoration |
| Catalog export | Row-to-ID alignment, collisions and identity suffixes, input validation, manifests, saved-model re-encoding |
| Directory publication | Failed first/replacement commits, obsolete split removal, unrelated-file preservation, ambiguous companions, reserved directory targets, unsupported exchange behavior |
| MiniOneRec integration | Actual converter and TokenExtender, complete SID lengths 1-6, collision suffix identities, Fusion title/description mappings, independent training/test constraint processors |

The integration tests read the current repository directly. They do not require
a nested source checkout, a pinned-tree manifest or private local paths.
Dataset classes are extracted from repository sources with AST; only eager
tokenization is disabled. The tokenizer stand-in raises if asked to tokenize,
so mapping checks cannot be mistaken for real tokenizer validation.

### Upstream compatibility suite

The existing repository suite was also rerun after conflict resolution:

```bash
python -m pip install openai==0.28.1 requests==2.32.3 aiohttp==3.10.3
python -m unittest discover -s tests -v
```

Result: **59 tests collected, 56 passed and 3 skipped, with no failures**.
The three skipped tests require a real `MINIMAX_API_KEY`; no live provider calls
were made. These dependencies were installed in an isolated test directory.
The upstream suite's existing heavy-dependency mocks remain in use, so its
success does not establish real language-model training or provider integration.

### Synthetic catalog workflows

Run the included standalone smoke script in the same environment:

```bash
python sid_generation/tests/smoke_catalog.py \
  --output-dir outputs/sid/synthetic-check
```

Choose a new output directory. The script generates a 10,240-item, 16-dimensional
catalog with explicit shuffled numeric IDs, duplicate embeddings, metadata and
12 synthetic interactions in each of the train/valid/test splits.

| Configuration | Fitting iterations | Expected semantic length |
| --- | --- | ---: |
| PRQ, three 256-entry codebooks | 5 | 3 |
| One RQ level plus two 8-bit OPQ subcodes | RQ 5; OPQ 3; PQ 5 | 3 |
| Three RQ levels plus two 8-bit OPQ subcodes | RQ 5; OPQ 3; PQ 5 | 5 |

All three workflows passed fitting, CLI export, checkpoint reload, identical
re-encoding, item-ID mapping, final SID uniqueness, the real converter on all
three splits, and the repository vocabulary loader. Collision suffixes were
needed in all three catalogs, giving actual final lengths of 4, 4 and 6.
The output contains per-command logs and `verification/smoke_results.json`.

These deliberately shortened fitting runs check functionality. Their collision
rates and runtime are not recommendation-quality or production-scale benchmarks.

## Remaining validation

- Real Qwen tokenization, model-level constrained generation, SFT and RL have
  not been executed for this PR.
- No HR/NDCG improvement, paper-level reproduction or superiority over existing
  SID builders is claimed.
- Existing-directory exchange was exercised on macOS. The Linux path and other
  filesystems still need target-environment validation.
- Other Python, NumPy and FAISS versions and cross-hardware bitwise consistency
  have not been established.
- The identity suffix is assigned for a complete catalog snapshot. Incremental
  identity registration and catalog updates require additional policy.

Use this PR as a reviewed CPU reference implementation, and run the real-model
checks before relying on its training integration in an experiment.
