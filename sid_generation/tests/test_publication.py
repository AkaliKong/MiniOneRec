"""Directory publication regressions: real exports and commit fault injection."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from prq_kmeans import PRQKMeans
from sid_cli import generate
import sid_publish


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def source(self, name, offset=0, splits=("train", "valid", "test")):
        folder = self.base / name
        folder.mkdir()
        ids = [str(offset + i) for i in range(8)]
        np.save(folder / "emb.npy", np.random.default_rng(offset).normal(size=(8, 4)).astype(np.float32))
        (folder / "ids.json").write_text(json.dumps(ids))
        (folder / "Demo.item.json").write_text(json.dumps({i: {"title": "Item " + i} for i in ids}))
        for split in splits:
            (folder / f"Demo.{split}.inter").write_text(f"user\thistory\ttarget\n0\t{ids[0]} {ids[1]}\t{ids[2]}\n")
        return folder

    def args(self, source, output, overwrite=False, dataset=True):
        return argparse.Namespace(
            operation="fit", data_path=str(source / "emb.npy"), dataset_name="Demo",
            output_dir=str(output), item_ids=str(source / "ids.json"),
            dataset_dir=str(source) if dataset else None, collision_policy="append",
            overwrite=overwrite, codebook_sizes=[2, 2], top_k=2, beta=15., n_iter=2,
            batch_size=8, seed=42,
        )

    def export(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate(args, "prq_kmeans", PRQKMeans)

    @staticmethod
    def contents(folder):
        return {str(path.relative_to(folder)): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_overwrite_removes_obsolete_owned_splits_and_preserves_other_files(self):
        first = self.source("first")
        second = self.source("second", 10, ("train",))
        output = self.base / "out"
        self.export(self.args(first, output))
        (output / "notes.txt").write_text("Keep me")
        (output / "Other.index.json").write_text('{"unrelated":true}')
        (output / "nested").mkdir()
        (output / "nested" / "kept.txt").write_text("Nested file")
        self.export(self.args(second, output, overwrite=True))
        self.assertFalse((output / "Demo.valid.inter").exists())
        self.assertFalse((output / "Demo.test.inter").exists())
        self.assertEqual((output / "notes.txt").read_text(), "Keep me")
        self.assertEqual((output / "Other.index.json").read_text(), '{"unrelated":true}')
        self.assertEqual((output / "nested" / "kept.txt").read_text(), "Nested file")
        index = json.loads((output / "Demo.index.json").read_text())
        self.assertEqual(set(index), {str(i) for i in range(10, 18)})
        manifest = json.loads((output / "Demo.manifest.json").read_text())
        self.assertNotIn("Demo.valid.inter", manifest["outputs"])
        # Actual converter sees only the current train split and current IDs.
        import ast
        converter = CODE.parent / "convert_dataset.py"
        tree = ast.parse(converter.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_dataset")
        scope = {"json": json, "os": __import__("os"), "Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(converter), "exec"), scope)
        loaded = scope["load_dataset"](str(output), "Demo")
        self.assertEqual(set(loaded["splits"]), {"train"})
        self.assertEqual(set(loaded["items"]), set(index))

    def test_encode_without_dataset_removes_previous_owned_companions(self):
        source = self.source("source")
        model_dir, output = self.base / "models", self.base / "out"
        self.export(self.args(source, model_dir))
        self.export(self.args(source, output))
        args = self.args(source, output, overwrite=True, dataset=False)
        args.operation = "encode"
        args.model_path = str(model_dir / "Demo.tokenizer.npz")
        self.export(args)
        for suffix in sid_publish.DATASET_SUFFIXES:
            self.assertFalse((output / ("Demo" + suffix)).exists())
        self.assertTrue((output / "Demo.index.json").is_file())

    def test_directory_target_rejected_before_fit_without_publishing(self):
        source, output = self.source("source"), self.base / "out"
        (output / "Demo.codes.npy").mkdir(parents=True)
        with patch.object(PRQKMeans, "fit") as fitting:
            with self.assertRaisesRegex(ValueError, "regular file"):
                self.export(self.args(source, output, overwrite=True))
        fitting.assert_not_called()
        self.assertFalse((output / "Demo.index.json").exists())
        self.assertEqual(list(output.iterdir()), [output / "Demo.codes.npy"])

    def test_commit_failure_leaves_old_directory_byte_identical(self):
        source, second, output = self.source("source"), self.source("second", 10), self.base / "out"
        self.export(self.args(source, output))
        before = self.contents(output)
        with patch.object(sid_publish, "_exchange_directories", side_effect=OSError("injected commit failure")):
            with self.assertRaisesRegex(OSError, "injected commit failure"):
                self.export(self.args(second, output, overwrite=True))
        self.assertEqual(before, self.contents(output))
        self.assertFalse(list(self.base.glob(".out-publish-*")))

    def test_first_publication_failure_leaves_no_public_directory(self):
        payload, output = self.base / "payload", self.base / "out"
        payload.mkdir()
        names = sid_publish.generated_names("Demo")
        for name in names:
            (payload / name).write_text("new catalog")
        with patch.object(sid_publish.os, "replace", side_effect=OSError("injected create failure")):
            with self.assertRaisesRegex(OSError, "injected create failure"):
                sid_publish.publish_catalog(payload, output, "Demo", names, False, set())
        self.assertFalse(output.exists())
        self.assertTrue(all((payload / name).exists() for name in names))

    def test_preserve_untracked_companion_by_refusing_ambiguous_export(self):
        source, output = self.source("source"), self.base / "out"
        output.mkdir()
        (output / "Demo.valid.inter").write_text("manually supplied validation")
        with self.assertRaisesRegex(ValueError, "untracked dataset companion"):
            self.export(self.args(source, output, overwrite=True, dataset=False))
        self.assertEqual((output / "Demo.valid.inter").read_text(), "manually supplied validation")
        self.assertFalse((output / "Demo.index.json").exists())

    def test_manifest_cannot_claim_unrelated_or_outside_files(self):
        source, output = self.source("source"), self.base / "out"
        self.export(self.args(source, output))
        manifest_path = output / "Demo.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"] += ["notes.txt", "../outside.txt"]
        manifest_path.write_text(json.dumps(manifest))
        (output / "notes.txt").write_text("Keep me")
        (self.base / "outside.txt").write_text("Keep outside")
        self.export(self.args(source, output, overwrite=True, dataset=False))
        self.assertEqual((output / "notes.txt").read_text(), "Keep me")
        self.assertEqual((self.base / "outside.txt").read_text(), "Keep outside")

    def test_unsupported_exchange_keeps_old_version(self):
        source, output = self.source("source"), self.base / "out"
        self.export(self.args(source, output))
        before = self.contents(output)
        with patch.object(sid_publish.sys, "platform", "unsupported"):
            with self.assertRaisesRegex(RuntimeError, "new output directory"):
                self.export(self.args(source, output, overwrite=True))
        self.assertEqual(before, self.contents(output))


if __name__ == "__main__":
    unittest.main()
