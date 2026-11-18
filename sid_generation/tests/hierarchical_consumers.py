"""Exercise MiniOneRec's SID consumers on small offline test catalogs."""

import ast
import json
import os
from pathlib import Path

import pytest
import torch

from data import EvalSidDataset, FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset, TitleHistory2SidSFTDataset
from LogitProcessor import ConstrainedLogitsProcessor


ROOT = Path(__file__).resolve().parents[2]
HEADER = "### Response:\n"


def token_extender(index_path):
    """Load the actual class without importing the training dependencies in sft.py."""
    path = ROOT / "sft.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenExtender")
    scope = {"os": os, "json": json}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    assert index_path.name.endswith(".index.json")
    return scope["TokenExtender"](str(index_path.parent), index_path.name.removesuffix(".index.json"))


def evaluator_constraints(index, tokenizer):
    """Run evaluate.py's prefix-table construction through its semantic callback."""
    path = ROOT / "evaluate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    get_hash = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_hash")
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    # Take only the statements between tokenizer initialization and the callback.
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "tokenizer") + 1
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.FunctionDef)
               and n.name == "prefix_allowed_tokens_fn_semantic") + 1
    scope = {"tokenizer": tokenizer, "base_model": "Qwen2.5", "info_titles": [],
             "info_semantic": [HEADER + "".join(tokens) + "\n" for tokens in index.values()]}
    statements = [get_hash, *main.body[start:end]]
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), scope)
    return scope["prefix_allowed_tokens_fn_semantic"]


def validate_dataset_contract(index_path, item_path, train_csv, tokenizer):
    """Read the exported catalog and exercise the actual datasets and their tokenized labels."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    full = {item: "".join(tokens) for item, tokens in index.items()}
    metadata = json.loads(item_path.read_text(encoding="utf-8"))
    catalog = {"item_file": item_path, "index_file": index_path, "tokenizer": tokenizer}
    feature = SidItemFeatDataset(**catalog)
    titles = {sid: metadata[item]["title"] for item, sid in full.items()}
    assert feature.sid2title == titles
    assert feature.title2sid == {title: sid for sid, title in titles.items()}

    fusion = FusionSeqRecDataset(train_file=train_csv, **catalog)
    title = TitleHistory2SidSFTDataset(train_file=train_csv, **catalog)
    assert title.id2sid == full
    sequence = SidSFTDataset(train_file=train_csv, tokenizer=tokenizer)
    evaluation = EvalSidDataset(train_file=train_csv, tokenizer=tokenizer)
    for position, row in sequence.data.iterrows():
        sid = full[str(row["item_id"])]
        assert fusion.get_history(row.copy())["target_sid"] == sid
        target = tokenizer.encode(sid + "\n", add_special_tokens=False) + [tokenizer.eos_token_id]
        for dataset in (sequence, evaluation, title):
            assert dataset.get_history(row.copy())["output"] == sid + "\n"
            prepared = dataset[position]
            assert [token for token in prepared["labels"] if token != -100] == target
            assert prepared["input_ids"][-len(target):] == target


def validate_index(index_path, tokenizer):
    """Check distinct SIDs, token registration and every legal transition on CPU."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert isinstance(index, dict) and index
    assert all(isinstance(path, list) and path and all(isinstance(token, str) for token in path)
               for path in index.values())
    paths = [tuple(path) for path in index.values()]
    unique = set(paths)
    assert len(paths) == len(unique), "duplicate SID"
    assert not any(path[:n] in unique for path in paths for n in range(1, len(path))), "SID prefix collision"

    tokens = token_extender(index_path).get_new_tokens()
    assert tokens == sorted({token for path in paths for token in path})
    assert tokenizer.add_tokens(tokens) == len(tokens)
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    assert len(set(token_ids.values())) == len(tokens)
    for token, token_id in token_ids.items():
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id]

    callback = evaluator_constraints(index, tokenizer)
    header = tokenizer.encode(HEADER, add_special_tokens=False)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    assert len(header) == 3 and newline
    states = {}
    continuations = []
    for path in paths:
        ids = [token_ids[token] for token in path]
        assert tokenizer.encode("".join(path), add_special_tokens=False) == ids
        continuation = ids + newline + [tokenizer.eos_token_id]
        continuations.append(continuation)
        for pos, token in enumerate(continuation):
            state = tuple(continuation[:pos] if pos else header)
            states.setdefault(state, set()).add(token)
    for state, allowed in states.items():
        assert set(callback(0, list(state))) == allowed

    scores = torch.zeros((2, len(tokenizer)), device="cpu")
    for continuation in continuations:
        processor = ConstrainedLogitsProcessor(callback, 2, "Qwen2.5", tokenizer.eos_token_id)
        for pos in range(len(continuation)):
            current = header + continuation[:pos]
            masked = processor(torch.tensor([current, current]), scores)
            expected = states[tuple(continuation[:pos] if pos else header)]
            for beam in masked:
                assert set(torch.isfinite(beam).nonzero().flatten().tolist()) == expected

    invalid = [tokenizer.unk_token_id]
    assert callback(0, invalid) == []
    processor = ConstrainedLogitsProcessor(callback, 1, "Qwen2.5", tokenizer.eos_token_id)
    processor(torch.tensor([header]), scores[:1])
    with pytest.warns(UserWarning, match="No valid tokens"):
        masked = processor(torch.tensor([header + invalid]), scores[:1])
    assert torch.isfinite(masked[0]).nonzero().flatten().tolist() == [tokenizer.eos_token_id]
