"""CPU regression tests for the glyph losses used by stage-1 training."""

import math
import importlib.util
import unittest
from types import SimpleNamespace

import torch

from seg_loss.focal_loss import BinaryFocalLoss
from seg_loss.dice_loss import BinaryDiceLoss, DiceLoss
from models.DiffTSR.TDM.models.text_prediction import My_Text_Prediction


class SegLossTests(unittest.TestCase):
    def test_original_focal_weighting_and_backward(self):
        logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
        target = torch.tensor([[[[0.0, 1.0]]]])
        loss = BinaryFocalLoss()(logits, target)
        self.assertAlmostEqual(loss.item(), 0.5 * math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertLess(logits.grad[0, 0, 0, 1].item(), 0)

    def test_focal_ignore_index(self):
        logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
        target = torch.tensor([[[[-1.0, 1.0]]]])
        loss = BinaryFocalLoss(ignore_index=-1)(logits, target)
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 0, 0].item(), 0)

    def test_binary_dice_perfect_prediction(self):
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        loss = BinaryDiceLoss()(target, target, use_sigmoid=False)
        self.assertEqual(loss.item(), 0)

    def test_multiclass_dice_backward(self):
        logits = torch.randn(2, 3, 4, 4, requires_grad=True)
        labels = torch.randint(0, 3, (2, 4, 4))
        target = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
        loss = DiceLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_supplied_text_does_not_require_transocr_weights(self):
        tokenizer = My_Text_Prediction(max_length=30, ckpt_path="missing-optional-transocr.pth")
        label = tokenizer.get_padding_label_from_text("TIGER中文")
        self.assertEqual(tuple(label.shape), (30,))
        self.assertEqual(label.dtype, torch.int64)
        self.assertIsNone(tokenizer.trans_ocr_model)

    @unittest.skipUnless(importlib.util.find_spec("diffusers"), "requires diffusion dependencies")
    def test_training_pipeline_returns_four_differentiable_losses(self):
        from models.DiffTSR.DiffTSR_base import UNet2OutPipeline_wEncoder
        from models.DiffTSR.sobel_edge_loss import DifferentiableEdgeDetector

        class TinyVAE(torch.nn.Module):
            def encode(self, image):
                return SimpleNamespace(sample=lambda: image)

            def decode(self, latent):
                return latent

        class TinyUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(9, 6, 1)

            def forward(self, x, timesteps, context):
                prediction = self.conv(x) + context.mean() / 10
                return prediction.chunk(2, dim=1)

        class TinyScheduler:
            def add_noise(self, sample, noise, timestep):
                return sample + noise

            def step(self, prediction, timestep, sample, return_dict):
                original = sample - prediction
                return original, original

        # Exercise the real training forward without allocating the full model
        # or requiring pretrained assets, CUDA, or Hi-SAM.
        model = UNet2OutPipeline_wEncoder.__new__(UNet2OutPipeline_wEncoder)
        torch.nn.Module.__init__(model)
        model.device = "cpu"
        model.scale_factor = 0.18215
        model.VAE_model = TinyVAE()
        model.Text_Prediction = My_Text_Prediction(max_length=30)
        model.transformer_encoder = torch.nn.Embedding(6736, 8)
        model.IDM_Unet = TinyUNet()
        model.edge_det = DifferentiableEdgeDetector().eval().requires_grad_(False)
        model.scheduler = TinyScheduler()
        image = torch.rand(2, 3, 8, 8)
        mask = torch.randint(0, 2, image.shape).float()
        losses = model(image, image, mask, ["TIGER", "中文"])
        self.assertEqual(len(losses), 4)
        for loss in losses:
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(loss.requires_grad)
        sum(losses).backward()
        self.assertTrue(torch.isfinite(model.IDM_Unet.conv.weight.grad).all())
        self.assertTrue(torch.isfinite(model.transformer_encoder.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
