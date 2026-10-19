"""MiniOneRec integration regressions, without language-model weights.

Dataset classes are extracted unchanged from the repository sources with AST.
Only their eager get_inputs() pass is disabled. The tokenizer below only has
ID attributes and raises if asked to tokenize: these are mapping/data-contract
tests, not a claim about Qwen tokenization, training, or model generation.
"""

import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, List
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from sid_common import build_indexes


class NoTokenization:
    bos_token_id = None
    eos_token_id = 99

    def encode(self, text):
        raise AssertionError("these integration tests must not simulate real tokenization")


def dataset_classes(path, ts=False):
    import pandas as pd
    names = {"Tokenizer", "SidSFTDataset", "SidItemFeatDataset", "FusionSeqRecDataset"}
    if not ts:
        names.update({"BaseDataset", "CSVBaseDataset", "JSONBaseDataset", "RLTitle2SidDataset",
                      "RLSid2TitleDataset", "TitleHistory2SidSFTDataset", "PreferenceSFTDataset",
                      "UserPreference2sidSFTDataset"})
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    scope = {"Dataset": object, "List": List, "pd": pd, "json": json, "random": random,
             "tqdm": lambda iterable: iterable}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)
    if ts:
        for name in ("SidSFTDataset", "SidItemFeatDataset", "FusionSeqRecDataset"):
            scope[name].get_inputs = lambda self: None
    else:
        scope["BaseDataset"].get_inputs = lambda self: None
    return scope


def processor_class(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ConstrainedLogitsProcessor")
    # Only the actual constructor runs; tensor annotations and decorators get
    # inert stand-ins. No tensor operations or real generation are claimed.
    scope = {"LogitsProcessor": object, "Callable": Callable, "List": List,
             "torch": SimpleNamespace(Tensor=object, LongTensor=object, FloatTensor=object),
             "add_start_docstrings": lambda *args: lambda fn: fn,
             "LOGITS_PROCESSOR_INPUTS_DOCSTRING": ""}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["ConstrainedLogitsProcessor"]


def construct_trainer_processors(path, cls, *, beam_search):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReReTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_inputs")
    selected = []
    for node in method.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if ((isinstance(target, ast.Name) and target.id == "ccc")
                    or (isinstance(target, ast.Attribute) and target.attr in ("logits_processor", "test_lp_list"))):
                selected.append(node)
    assert len(selected) == 3
    trainer_state = SimpleNamespace(
        prefix_allowed_tokens_fn=lambda batch, ids: [], num_generations=16,
        beam_search=beam_search, test_beam=20, base_model="Qwen2.5",
        processing_class=SimpleNamespace(eos_token_id=99), temperature=1.0,
    )
    scope = {"self": trainer_state, "ConstrainedLogitsProcessor": cls,
             "LogitsProcessorList": list, "TemperatureLogitsWarper": lambda **kwargs: object()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return trainer_state.logits_processor[-1], trainer_state.test_lp_list[0]


@unittest.skipUnless(importlib.util.find_spec("pandas"),
                     "integration regressions need pandas")
class MiniOneRecIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixed = CODE.parent
        cls.scope = dataset_classes(cls.fixed / "data.py")
        cls.ts_scope = dataset_classes(cls.fixed / "ts_rec_data.py", ts=True)

    def assert_catalog_contract(self, index, *, ts=False):
        import pandas as pd
        scope = self.ts_scope if ts else self.scope
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            full = {key: "".join(tokens) for key, tokens in index.items()}
            metadata = {key: {"title": f"Title {key}", "description": f"Description {key}"}
                        for key in index}
            (folder / "Demo.item.json").write_text(json.dumps(metadata), encoding="utf-8")
            (folder / "Demo.index.json").write_text(json.dumps(index), encoding="utf-8")
            rows = [{"history_item_sid": str([full["0"]]), "history_item_title": str(["Title 0"]),
                     "history_item_id": str([0]), "item_sid": full[key], "item_id": int(key)}
                    for key in index]
            frame = pd.DataFrame(rows)
            frame.to_csv(folder / "train.csv", index=False)
            kwargs = {"item_file": str(folder / "Demo.item.json"),
                      "index_file": str(folder / "Demo.index.json")}
            feature = scope["SidItemFeatDataset"](tokenizer=NoTokenization(), **kwargs)
            self.assertEqual(feature.title2sid, {f"Title {key}": value for key, value in full.items()})
            self.assertEqual(feature.sid2title, {value: f"Title {key}" for key, value in full.items()})
            self.assertEqual(len(feature.data), 2 * len(index))
            sequence = scope["SidSFTDataset"](train_file=str(folder / "train.csv"), tokenizer=NoTokenization())
            fusion = scope["FusionSeqRecDataset"](
                train_file=str(folder / "train.csv"), tokenizer=NoTokenization(), **kwargs)
            for key in index:
                row = frame.iloc[int(key)].copy()
                self.assertEqual(sequence.get_history(row.copy())["output"], full[key] + "\n")
                history = fusion.get_history(row.copy())
                self.assertEqual(history["target_title"], f"Title {key}")
                self.assertEqual(history["target_description"], f"Description {key}")
                self.assertEqual(history["target_sid"], full[key])
            if ts:
                return
            rl = scope["RLTitle2SidDataset"](**kwargs)
            self.assertEqual(rl.title2sid, {f"Title {key}": value for key, value in full.items()})
            self.assertEqual(rl.description2sid, {f"Description {key}": value for key, value in full.items()})
            for position in range(len(rl.data)):
                self.assertIn(rl.pre(position)["completion"].strip(), full.values())
            sid_to_title = scope["RLSid2TitleDataset"](**kwargs)
            self.assertEqual(sid_to_title.sid2title, {value: f"Title {key}" for key, value in full.items()})
            title_history = scope["TitleHistory2SidSFTDataset"](
                train_file=str(folder / "train.csv"), tokenizer=NoTokenization(), **kwargs)
            self.assertEqual(title_history.id2sid, full)
            for key in index:
                self.assertEqual(title_history.get_history(frame.iloc[int(key)].copy())["output"], full[key] + "\n")
            for name in ("PreferenceSFTDataset", "UserPreference2sidSFTDataset"):
                dataset = scope[name].__new__(scope[name])
                dataset.indices = index
                self.assertEqual(dataset._convert_to_semantic_ids([0, 1, 99]), [full["0"], full["1"], "99"])

    def test_complete_sid_lengths_one_through_six(self):
        for length in range(1, 7):
            with self.subTest(length=length):
                codes = np.array([[0] * length, [0] * (length - 1) + [1]], dtype=np.int64)
                _, final, _ = build_indexes(["0", "1"], codes, [2] * length, "error")
                self.assert_catalog_contract(final)

    def test_collision_suffix_retains_distinct_identities(self):
        for semantic_length in (3, 5):
            with self.subTest(semantic_length=semantic_length):
                codes = np.zeros((2, semantic_length), dtype=np.int64)
                _, final, stats = build_indexes(["0", "1"], codes, [2] * semantic_length)
                self.assertTrue(stats["identity_suffix_appended"])
                self.assertEqual(len(final["0"]), semantic_length + 1)
                self.assertNotEqual(final["0"], final["1"])
                self.assert_catalog_contract(final)

    def test_ts_rec_entry_point_uses_complete_ids(self):
        for length in range(1, 7):
            with self.subTest(length=length):
                codes = np.array([[0] * length, [0] * (length - 1) + [1]], dtype=np.int64)
                _, final, _ = build_indexes(["0", "1"], codes, [2] * length, "error")
                self.assert_catalog_contract(final, ts=True)


    def test_training_and_test_generations_have_independent_counters(self):
        cls = processor_class(self.fixed / "LogitProcessor.py")
        for beam_search in (False, True):
            with self.subTest(beam_search=beam_search):
                training, testing = construct_trainer_processors(
                    self.fixed / "minionerec_trainer.py", cls, beam_search=beam_search)
                self.assertIsNot(training, testing)
                testing.count = 8
                self.assertEqual(training.count, 0)
                self.assertEqual(testing._num_beams, 20)
                self.assertEqual(training._num_beams, 16 if beam_search else 1)
                self.assertEqual(training.prefix_index, 3)
                self.assertEqual(testing.prefix_index, 3)


if __name__ == "__main__":
    unittest.main()
