"""Portable plan/preflight tests using only tiny local files and the stdlib."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(os.environ.get("DLM_NEW_METHOD_ROOT", Path(__file__).resolve().parents[1]))


class RunnerTests(unittest.TestCase):
    def fixture(self, root):
        row = {"context": "Two people arrived.", "question": "Who?",
               "ans0": "Person one", "ans1": "Person two", "ans2": "Cannot tell",
               "answer_info": {"ans0": ["one", "Black"], "ans1": ["two", "White"],
                               "ans2": ["unknown", "unknown"]}}
        items = root / "input items.jsonl"
        items.write_text(json.dumps(row) + "\n")
        fit = root / "direction examples.jsonl"
        fit.write_text(json.dumps(dict(row, context="A different training context.")) + "\n")
        arrows = root / "arrows.pt"
        arrows.write_bytes(b"tiny artifact hashed but never loaded by CPU planning")
        return ["--items", str(items), "--fit-examples", str(fit), "--arrows", str(arrows),
                "--data-root", str(root), "--model", str(root / "weights absent")]

    def invoke(self, *arguments):
        # -S removes third-party packages entirely: planning/help must not need torch.
        return subprocess.run([sys.executable, "-S", str(SOURCE / "run.py"), *arguments],
                              capture_output=True, text=True)

    def test_help_without_torch_or_model_dependencies(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--compare", result.stdout)
        self.assertIn("--plan-only", result.stdout)

    def test_plan_without_weights_retains_visible_whitelist_and_exact_budget(self):
        with tempfile.TemporaryDirectory(prefix="new method plan ") as temporary:
            root = Path(temporary)
            output = root / "plan output"
            result = self.invoke(*self.fixture(root), "--out-root", str(output),
                                 "--plan-only", "--compare", "--rotations", "3")
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((output / "plan.json").read_text())
            self.assertFalse(plan["model"]["available"])
            self.assertEqual(plan["cohort"]["n_semantic_items"], 1)
            self.assertEqual(plan["cohort"]["n_visible_items"], 3)
            self.assertEqual(plan["forward_budget"]["actual_per_visible_item"], 261)
            self.assertEqual(plan["forward_budget"]["actual_total"], 783)
            self.assertEqual(plan["arms"], ["clean", "selector_only", "fixed_openloop", "auto_openloop", "auto_pi"])
            for line in (output / "visible_inputs.jsonl").read_text().splitlines():
                self.assertEqual(set(json.loads(line)), {"context", "question", "ans0", "ans1", "ans2"})
            self.assertEqual(json.loads((output / "status.json").read_text()),
                             {"status": "planned", "inference_executed": False})
            self.assertFalse((output / "generations.jsonl").exists())

    def test_live_missing_weights_fails_explicitly_before_heavy_import_or_download(self):
        with tempfile.TemporaryDirectory(prefix="new method missing weights ") as temporary:
            root = Path(temporary)
            output = root / "failed run"
            result = self.invoke(*self.fixture(root), "--out-root", str(output))
            self.assertEqual(result.returncode, 1)
            self.assertIn("model weights are unavailable", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "generations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
