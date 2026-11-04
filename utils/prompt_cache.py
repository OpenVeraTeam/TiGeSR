"""Validation for the SD3 prompt embeddings shared by training and inference."""

from pathlib import Path

import torch


def load_prompt_cache(path):
    """Load one prompt embedding without guessing its transcription from its name."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing SD3 empty-prompt cache: {path}. This is a generated TiGeSR asset, "
            "not part of the official SD3 checkpoint. Generate it once with "
            f'python scripts/prepare_sd3_prompt_cache.py --model-dir "{path.parent}" '
            f'--output "{path}"'
        )
    cache = torch.load(path, map_location="cpu", weights_only=True)
    expected_shapes = {"prompt_embeds": (1, 333, 4096), "pooled_prompt_embeds": (1, 2048)}
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid prompt cache {path}: expected a dictionary of tensors.")
    for key, shape in expected_shapes.items():
        value = cache.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Invalid prompt cache {path}: {key} must have shape {shape}.")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Invalid prompt cache {path}: {key} must contain finite floating point values.")
    return cache
