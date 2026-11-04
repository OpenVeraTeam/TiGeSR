"""CPU regression checks for the generated SD3 prompt-cache contract."""

from pathlib import Path
import tempfile
import unittest

import torch

from utils.prompt_cache import load_prompt_cache


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prompt.pt"

    @staticmethod
    def valid_cache():
        return {
            "prompt_embeds": torch.ones(1, 333, 4096),
            "pooled_prompt_embeds": torch.ones(1, 2048),
        }

    def test_missing_cache_has_generation_command(self):
        with self.assertRaisesRegex(FileNotFoundError, "prepare_sd3_prompt_cache.py"):
            load_prompt_cache(self.path)

    def test_valid_cache_loaded_on_cpu(self):
        torch.save(self.valid_cache(), self.path)
        cache = load_prompt_cache(self.path)
        self.assertEqual(tuple(cache["prompt_embeds"].shape), (1, 333, 4096))
        self.assertEqual(cache["prompt_embeds"].device.type, "cpu")

    def test_missing_key_is_rejected(self):
        cache = self.valid_cache()
        del cache["pooled_prompt_embeds"]
        torch.save(cache, self.path)
        with self.assertRaisesRegex(ValueError, "pooled_prompt_embeds"):
            load_prompt_cache(self.path)

    def test_guidance_batch_is_rejected(self):
        cache = self.valid_cache()
        cache["prompt_embeds"] = torch.ones(2, 333, 4096)
        torch.save(cache, self.path)
        with self.assertRaisesRegex(ValueError, "prompt_embeds"):
            load_prompt_cache(self.path)

    def test_nonfinite_values_are_rejected(self):
        cache = self.valid_cache()
        cache["pooled_prompt_embeds"][0, 0] = float("nan")
        torch.save(cache, self.path)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_prompt_cache(self.path)


if __name__ == "__main__":
    unittest.main()
