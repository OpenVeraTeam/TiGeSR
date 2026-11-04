"""Import regression checks: no model downloads, GPU, or Paddle runtime required.

Run these after installing requirements/train.txt, not in the separate OCR
environment. Missing third-party packages are failures rather than mock imports.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class ReleaseImportTests(unittest.TestCase):
    def assert_imports_without_paddle(self, modules):
        repository = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["HF_HUB_OFFLINE"] = "1"
        code = "; ".join(
            ["import sys"]
            + [f"import {module}" for module in modules]
            + ["assert 'paddle' not in sys.modules", "assert 'paddleocr' not in sys.modules"]
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inference_imports_without_paddle(self):
        self.assert_imports_without_paddle([
            "models.pure_ctrlnet", "models.FullPipeline",
            "models.DiffTSR.DiffTSR_base", "scripts.inference",
        ])

    def test_training_and_degradation_imports(self):
        self.assert_imports_without_paddle([
            "scripts.train_stage1", "scripts.train_stage2", "data.deg_mine",
        ])


if __name__ == "__main__":
    unittest.main()
