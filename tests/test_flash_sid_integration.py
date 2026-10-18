"""Regression checks for complete SID handling and standalone entry points."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "SidItemFeatDataset", "RLTitle2SidDataset", "RLSid2TitleDataset",
    "FusionSeqRecDataset", "TitleHistory2SidSFTDataset", "PreferenceSFTDataset",
    "UserPreference2sidSFTDataset",
}


class CompleteSIDTests(unittest.TestCase):
    def test_all_seven_datasets_keep_three_and_four_token_ids(self):
        tree = ast.parse((ROOT / "data.py").read_text(encoding="utf-8"))
        seen = set()
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in DATASETS:
                continue
            seen.add(cls.name)
            assignments = [n for n in ast.walk(cls) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "combined_sid" for t in n.targets)]
            self.assertEqual(len(assignments), 1, cls.name)
            expression = compile(ast.Expression(assignments[0].value), "<sid-expression>", "eval")
            for tokens in [["<a_1>", "<b_2>", "<c_3>"],
                           ["<a_1>", "<b_2>", "<c_3>", "<d_0>"],
                           ["<a_1>", "<b_2>", "<c_3>", "<d_1>"]]:
                actual = eval(expression, {"__builtins__": {}}, {"sids": tokens})
                self.assertEqual(actual, "".join(tokens), cls.name)
        self.assertEqual(seen, DATASETS)

    def test_converter_retains_collision_suffix(self):
        tree = ast.parse((ROOT / "convert_dataset.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "semantic_tokens_to_id")
        namespace = {"List": list}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<converter>", "exec"), namespace)
        tokens = ["<a_1>", "<b_2>", "<c_3>", "<d_4>"]
        self.assertEqual(namespace["semantic_tokens_to_id"](tokens), "".join(tokens))

    def test_script_and_module_entry_points(self):
        variants = [([str(ROOT / "rq/generate_indices_flash.py")], ROOT / "tests"),
                    (["-m", "rq.generate_indices_flash"], ROOT),
                    (["-m", "rq.flash_sid"], ROOT)]
        for arguments, cwd in variants:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-B", *arguments, "--help"],
                                        cwd=cwd, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--embeddings", result.stdout)


if __name__ == "__main__":
    unittest.main()
