"""Small subprocess checks for the real Bash launcher's argument boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_spaces_and_shell_metacharacters_are_literal_and_offline_is_forced(self):
        with tempfile.TemporaryDirectory(prefix="new method launcher ") as temporary:
            root = Path(temporary)
            launcher = root / "checkout with spaces" / "new method" / "run.sh"
            launcher.parent.mkdir(parents=True)
            launcher.write_bytes((HERE / "run.sh").read_bytes())
            interpreter = root / "python environment with spaces"
            interpreter.write_text(
                f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "print(json.dumps({'args':sys.argv[1:], 'offline':[os.environ.get(k) "
                "for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE']]}))\n"
            )
            interpreter.chmod(0o700)
            marker = root / "should not exist"
            arguments = ["--preset", "smoke", "--model", str(root / "model's weights"),
                         "--data-root", "literal $HOME `uname`", "--out-root",
                         f"$(touch '{marker}')"]
            result = subprocess.run(
                ["bash", str(launcher), *arguments], cwd=root,
                env=dict(os.environ, PYTHON=str(interpreter), HF_HUB_OFFLINE="0"),
                check=True, capture_output=True, text=True,
            )
            output = json.loads(result.stdout)
            self.assertEqual(output["args"], ["-u", str(launcher.parent / "run.py"), *arguments])
            self.assertEqual(output["offline"], ["1", "1", "1"])
            self.assertFalse(marker.exists())

    def test_missing_interpreter_fails_before_runner(self):
        result = subprocess.run(
            ["bash", str(HERE / "run.sh"), "--help"],
            env=dict(os.environ, PYTHON="/no/such/research/python"),
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 127)
        self.assertIn("Set PYTHON", result.stderr)


if __name__ == "__main__":
    unittest.main()
