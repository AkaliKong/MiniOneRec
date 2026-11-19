"""Integration with a local WordLevel tokenizer, real MiniOneRec datasets and CPU masks."""

import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pytest
import torch

from sid_generation.hierarchical_cli import main as hierarchical_cli
from sid_generation.tests.hierarchical_consumers import (
    evaluator_constraints, token_extender, validate_dataset_contract, validate_index,
)


@pytest.fixture(scope="module", autouse=True)
def single_threaded_cpu_masks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def local_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocabulary = {"[UNK]": 0, "[EOS]": 1, "###": 2, " Response": 3, ":\n": 4, "\n": 5}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"###| Response|:\n|\n|."), behavior="isolated")
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")


def write_companions(folder, ids, history):
    metadata = {item: {"title": f"Item {item}", "description": f"Description {item}"} for item in ids}
    (folder / "Demo.item.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = ["user_id:token\titem_id_list:token_seq\titem_id:token"]
    rows.extend(f"99\t{' '.join(history)}\t{item}" for item in ids)
    (folder / "Demo.train.inter").write_text("\n".join(rows) + "\n", encoding="utf-8")


def convert_training_split(catalog, output):
    script = Path(__file__).resolve().parents[2] / "convert_dataset.py"
    subprocess.run([sys.executable, str(script), "--data_dir", str(catalog), "--dataset_name", "Demo",
                    "--output_dir", str(output)], check=True)
    return output / "train" / "Demo_5_2016-10-2018-11.csv"


class HierarchicalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "Demo.index.json"
        self.index = {
            "0": ["<leaf_000000>"],
            "1": ["<node_000001>", "<leaf_000001>"],
            "2": ["<node_000001>", "<node_000002>", "<leaf_000002>"],
            "3": ["<node_000001>", "<node_000002>", "<node_000003>", "<leaf_000003>"],
            "4": ["<node_000001>", "<node_000002>", "<node_000003>", "<leaf_000004>"],
        }
        self.path.write_text(json.dumps(self.index), encoding="utf-8")

    def test_token_extender_collects_namespaces_and_tokenizer_keeps_adjacency(self):
        extender = token_extender(self.path)
        tokens = extender.get_new_tokens()
        self.assertEqual(tokens, sorted({token for path in self.index.values() for token in path}))
        self.assertIs(extender.get_new_tokens(), tokens)
        tokenizer = local_tokenizer()
        self.assertEqual(tokenizer.add_tokens(tokens), len(tokens))
        for path in self.index.values():
            ids = [tokenizer.convert_tokens_to_ids(token) for token in path]
            self.assertEqual(tokenizer.encode("".join(path), add_special_tokens=False), ids)

    def test_mixed_length_catalog_actual_evaluator_and_cpu_masks(self):
        self.assertEqual({len(path) for path in self.index.values()}, {1, 2, 3, 4})
        validate_index(self.path, local_tokenizer())

    def test_actual_dataset_classes_preserve_all_paths(self):
        tokenizer = local_tokenizer()
        tokenizer.add_tokens(token_extender(self.path).get_new_tokens())
        folder = self.path.parent
        write_companions(folder, self.index, ["0"])
        train_csv = convert_training_split(folder, folder / "converted")
        validate_dataset_contract(self.path, folder / "Demo.item.json", train_csv, tokenizer)

    def test_terminal_leaf_permits_newline_then_eos_only(self):
        tokenizer = local_tokenizer()
        tokenizer.add_tokens(token_extender(self.path).get_new_tokens())
        callback = evaluator_constraints(self.index, tokenizer)
        for path in self.index.values():
            ids = tokenizer.encode("".join(path), add_special_tokens=False)
            newline = tokenizer.encode("\n", add_special_tokens=False)
            self.assertEqual(callback(0, ids), newline)
            self.assertEqual(callback(0, ids + newline), [tokenizer.eos_token_id])
        wrong = tokenizer.encode("<node_000001><leaf_000000>", add_special_tokens=False)
        self.assertEqual(callback(0, wrong), [])

    def test_rejects_sid_prefix_collision(self):
        self.index["5"] = ["<node_000001>"]
        self.path.write_text(json.dumps(self.index), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "prefix collision"):
            validate_index(self.path, local_tokenizer())

    def test_rejects_duplicate_final_sid(self):
        self.index["5"] = list(self.index["0"])
        self.path.write_text(json.dumps(self.index), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "duplicate SID"):
            validate_index(self.path, local_tokenizer())


@pytest.mark.parametrize("method", ["balanced", "greedy", "hybrid"])
def test_generated_strategy_sids_reach_existing_consumers(method, tmp_path):
    ids = [str(7 * i) for i in reversed(range(11))]
    vectors = np.random.default_rng(113).normal(size=(len(ids), 4)).astype("float32")
    np.save(tmp_path / "embeddings.npy", vectors, allow_pickle=False)
    (tmp_path / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    write_companions(tmp_path, ids, ["70", "0"])
    catalog = tmp_path / "catalog"
    hierarchical_cli(["fit", "--method", method, "--branch-factor", "2", "--greedy-threshold", "5",
                      "--data-path", str(tmp_path / "embeddings.npy"), "--item-ids", str(tmp_path / "ids.json"),
                      "--dataset-name", "Demo", "--dataset-dir", str(tmp_path), "--output-dir", str(catalog)])
    path = catalog / "Demo.index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = local_tokenizer()
    validate_index(path, tokenizer)
    train_csv = convert_training_split(catalog, tmp_path / "converted")
    validate_dataset_contract(path, catalog / "Demo.item.json", train_csv, tokenizer)
    with train_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in rows] == ids
    for row in rows:
        assert row["user_id"] == "A99" and row["item_sid"] == "".join(index[row["item_id"]])
        assert ast.literal_eval(row["history_item_sid"]) == ["".join(index[key]) for key in ("70", "0")]


if __name__ == "__main__":
    unittest.main()
