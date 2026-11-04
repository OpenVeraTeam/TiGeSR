"""Hi-SAM checkpoint completion without constructing its large image encoder."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from hi_sam.modeling.build import _load_hisam_checkpoint


class TinyHiSam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2, bias=False)
        self.adapter = torch.nn.Linear(2, 1, bias=False)


class HiSamAssetTests(unittest.TestCase):
    def test_parallel_builder_exposes_checkpoint_paths(self):
        from hi_sam.modeling.build import build_sam_vit_h_parallel
        with patch("hi_sam.modeling.build._build_sam_parallel", return_value="model") as builder:
            self.assertEqual(build_sam_vit_h_parallel("trained.pth", "base.pth"), "model")
        self.assertEqual(builder.call_args.kwargs["checkpoint"], "trained.pth")
        self.assertEqual(builder.call_args.kwargs["sam_checkpoint"], "base.pth")

    def test_complete_wrapped_checkpoint_needs_no_original_sam(self):
        model = TinyHiSam()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        with patch("hi_sam.modeling.build.torch.load", return_value={"model": state}) as loader:
            result = _load_hisam_checkpoint(model, "complete.pth", SimpleNamespace(model_type="vit_h"))
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(result.missing_keys, [])
        for tensor in model.state_dict().values():
            self.assertTrue(torch.equal(tensor, torch.ones_like(tensor)))

    def test_partial_checkpoint_reports_missing_original_sam(self):
        model = TinyHiSam()
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing-sam.pth"
            with patch("hi_sam.modeling.build.torch.load", return_value={"adapter.weight": torch.ones(1, 2)}):
                with self.assertRaisesRegex(FileNotFoundError, "original SAM checkpoint"):
                    _load_hisam_checkpoint(model, "partial.pth", SimpleNamespace(model_type="vit_h", sam_checkpoint=missing_path))

    def test_original_sam_only_fills_missing_tensors_and_extras_are_allowed(self):
        model = TinyHiSam()
        trained = {"adapter.weight": torch.full((1, 2), 3.0), "unused.weight": torch.ones(1)}
        original = {"backbone.weight": torch.full((2, 2), 5.0), "adapter.weight": torch.zeros(1, 2)}
        with tempfile.TemporaryDirectory() as directory:
            sam_path = Path(directory) / "sam.pth"
            sam_path.touch()
            with patch("hi_sam.modeling.build.torch.load", side_effect=[trained, original]):
                result = _load_hisam_checkpoint(model, "partial.pth", SimpleNamespace(model_type="vit_h", sam_checkpoint=sam_path))
        torch.testing.assert_close(model.adapter.weight, torch.full((1, 2), 3.0))
        torch.testing.assert_close(model.backbone.weight, torch.full((2, 2), 5.0))
        self.assertEqual(result.unexpected_keys, ["unused.weight"])
        self.assertNotIn("backbone.weight", trained)

    def test_still_missing_parameters_raise_instead_of_random_initialization(self):
        model = TinyHiSam()
        with tempfile.TemporaryDirectory() as directory:
            sam_path = Path(directory) / "sam.pth"
            sam_path.touch()
            with patch("hi_sam.modeling.build.torch.load", side_effect=[{}, {"backbone.weight": torch.ones(2, 2)}]):
                with self.assertRaisesRegex(ValueError, "remains incomplete"):
                    _load_hisam_checkpoint(model, "partial.pth", SimpleNamespace(model_type="vit_h", sam_checkpoint=sam_path))


if __name__ == "__main__":
    unittest.main()
