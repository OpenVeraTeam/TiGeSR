"""Annotation geometry and online/cache routing checks without model assets."""

from copy import deepcopy
from types import MethodType
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import cv2  # Import native extensions before patch.dict snapshots sys.modules.

from utils.annotations import inference_size, resize_annotation


def annotation(angle=0, size=(400, 200)):
    return {
        "doc_preprocessor_res": {"angle": angle},
        "dt_polys": [[[20, 40], [60, 40], [60, 80], [20, 80]]],
        "rec_texts": ["TIGER"],
        "tiger_image_size": list(size),
    }


class AnnotationGeometryTests(unittest.TestCase):
    def test_inference_size_uses_shared_multiples_of_eight(self):
        self.assertEqual(inference_size(4000, 3000), (1024, 768))
        self.assertEqual(inference_size(3000, 4000), (768, 1024))
        self.assertEqual(inference_size(1, 4000), (8, 1024))
        with self.assertRaises(ValueError):
            inference_size(0, 50)

    def test_all_document_angles_resize_in_the_rotated_frame(self):
        for angle in (-1, 0, 90, 180, 270):
            with self.subTest(angle=angle):
                source = annotation(angle)
                before = deepcopy(source)
                resized = resize_annotation(source, (400, 200), (200, 96))
                scale = [0.48, 0.5] if angle in (90, 270) else [0.5, 0.48]
                expected = np.asarray(source["dt_polys"]) * scale
                np.testing.assert_allclose(resized["dt_polys"], expected)
                self.assertEqual(resized["tiger_image_size"], [200, 96])
                self.assertEqual(resized["rec_scores"], [1.0])
                self.assertEqual(resized["textline_orientation_angles"], [0])
                self.assertEqual(source, before)

    def test_cache_metadata_prevents_double_scaling(self):
        resized = resize_annotation(annotation(90), (400, 200), (200, 96))
        same = resize_annotation(resized, (400, 200), (200, 96))
        self.assertEqual(resized, same)

    def test_legacy_cache_uses_supplied_original_size(self):
        source = annotation(270)
        del source["tiger_image_size"]
        resized = resize_annotation(source, (400, 200), (200, 96))
        np.testing.assert_allclose(resized["dt_polys"][0][0], [9.6, 20])

    def test_invalid_polygon_and_orientation_fail_early(self):
        bad = annotation()
        bad["dt_polys"] = []
        with self.assertRaises(ValueError):
            resize_annotation(bad, (400, 200), (200, 96))
        bad = annotation(45)
        with self.assertRaises(ValueError):
            resize_annotation(bad, (400, 200), (200, 96))
        bad = annotation()
        bad["dt_polys"][0].pop()
        with self.assertRaises(ValueError):
            resize_annotation(bad, (400, 200), (200, 96))

    def test_cache_generator_records_prerotation_size_for_90_degrees(self):
        from PIL import Image
        from scripts import make_ocr_annotations
        from types import SimpleNamespace

        inputs = []

        class FakeOCR:
            def predict(self, bgr):
                inputs.append(bgr.copy())
                return [{
                    "doc_preprocessor_res": {"angle": 90},
                    "dt_polys": np.asarray([[[2, 4], [6, 4], [6, 8], [2, 8]]]),
                    "rec_texts": ["TIGER"],
                    "rec_scores": np.asarray([0.99]),
                    "textline_orientation_angles": np.asarray([0]),
                }]

        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "inputs"
            output_dir = Path(directory) / "annotations"
            input_dir.mkdir()
            Image.new("RGB", (80, 40), (255, 128, 0)).save(input_dir / "sample.png")
            argv = ["make_ocr_annotations.py", "--input-dir", str(input_dir),
                    "--output-dir", str(output_dir), "--max-side", "64"]
            with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCR=lambda **kwargs: FakeOCR())}), patch.object(sys, "argv", argv):
                make_ocr_annotations.main()
            cache = json.loads((output_dir / "sample_res.json").read_text(encoding="utf-8"))
        self.assertEqual(inputs[0].shape, (32, 64, 3))
        np.testing.assert_array_equal(inputs[0][0, 0], [0, 128, 255])
        self.assertEqual(cache["tiger_image_size"], [64, 32])
        self.assertEqual(cache["doc_preprocessor_res"]["angle"], 90)
        # These are already coordinates at the OCR resize, so loading a cache
        # for the same inference size must leave the polygons untouched.
        again = resize_annotation(cache, (80, 40), (64, 32))
        self.assertEqual(again["dt_polys"], cache["dt_polys"])


class AnnotationRoutingTests(unittest.TestCase):
    def test_document_rotation_roundtrip_with_and_without_tiling(self):
        from models.FullPipeline import TextSR_multiscale

        class IdentityEnhancer(torch.nn.Module):
            def forward(self, image, mask, width, height):
                self.assertions = (tuple(image.shape[-2:]), (height, width))
                return image.clone()

        model = TextSR_multiscale.__new__(TextSR_multiscale)
        torch.nn.Module.__init__(model)
        model.execution_device = torch.device("cpu")
        model.net_pix2pix = IdentityEnhancer()
        image = torch.arange(3 * 8 * 16).reshape(1, 3, 8, 16).float() / (3 * 8 * 16)
        for angle in (-1, 0, 90, 180, 270):
            for tiling in (False, True):
                with self.subTest(angle=angle, tiling=tiling):
                    record = {"doc_preprocessor_res": {"angle": angle},
                              "dt_polys": [], "rec_texts": [], "rec_scores": [],
                              "textline_orientation_angles": []}
                    restored, mask, glyphs = model.inference_wanno(image, record, tiling=tiling)
                    self.assertEqual(tuple(restored.shape), tuple(image.shape))
                    self.assertEqual(tuple(mask.shape), tuple(image.shape))
                    self.assertEqual(tuple(glyphs.shape), tuple(image.shape))
                    self.assertEqual(mask.count_nonzero().item(), 0)
                    self.assertTrue(torch.isfinite(restored).all())
                    self.assertEqual(*model.net_pix2pix.assertions)
                    if not tiling:
                        torch.testing.assert_close(restored, image)

    def test_online_and_cached_ocr_route_to_identical_geometry(self):
        import models.FullPipeline as pipeline_module

        class FakeOCR:
            def __init__(self, result):
                self.result = result
                self.inputs = []

            def predict(self, image):
                self.inputs.append(image.copy())
                return [deepcopy(self.result)]

        for angle in (-1, 0, 90, 180, 270):
            with self.subTest(angle=angle):
                image = torch.empty(1, 3, 16, 32)
                image[:, 0] = 0.2
                image[:, 1] = 0.4
                image[:, 2] = 0.8
                original = annotation(angle, size=(80, 24))
                cached = resize_annotation(original, (80, 24), (32, 16))
                online = deepcopy(cached)
                # Native Paddle results have no TiGeSR metadata.
                del online["tiger_image_size"]
                fake_ocr = FakeOCR(online)
                model = pipeline_module.TextSR_multiscale.__new__(pipeline_module.TextSR_multiscale)
                torch.nn.Module.__init__(model)
                model.ocr = None
                received = []

                def fake_inference_wanno(self, lq, anno, tiling):
                    received.append((lq, deepcopy(anno), tiling))
                    return "restored"

                model.inference_wanno = MethodType(fake_inference_wanno, model)
                with patch.object(pipeline_module, "PaddleOCR", return_value=fake_ocr) as factory:
                    self.assertEqual(model.inference(image, tiling=False), "restored")
                    factory.assert_called_once()
                self.assertEqual(len(fake_ocr.inputs), 1)
                self.assertEqual(fake_ocr.inputs[0].dtype, np.uint8)
                np.testing.assert_array_equal(fake_ocr.inputs[0][0, 0], [204, 102, 51])
                with patch.object(pipeline_module, "PaddleOCR", side_effect=AssertionError("OCR must not run for a cache")):
                    self.assertEqual(model.inference(image, anno=cached, tiling=False), "restored")
                self.assertEqual(len(fake_ocr.inputs), 1)
                self.assertIs(received[0][0], image)
                self.assertIs(received[1][0], image)
                self.assertEqual(received[0][1], received[1][1])
                self.assertEqual(received[0][2], False)
                self.assertEqual(received[1][2], False)


if __name__ == "__main__":
    unittest.main()
