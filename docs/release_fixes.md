# Release audit: 4 November 2026

The release was compared with the original stage-1 and stage-2 working trees and the repository's issue threads.

## Corrected

- [Missing seg_loss (#3)](https://github.com/OpenVeraTeam/TiGeSR/issues/3): recovered the original focal and Dice implementations. Recovered the omitted Sobel loss, encoder-training class and four-loss forward method; training launchers now use training-specific YAML targets.
- [Environment conflicts (#4)](https://github.com/OpenVeraTeam/TiGeSR/issues/4): pinned direct dependencies and separated PyTorch, PaddleOCR and PyIQA environments. Cached-annotation inference and image-only/cached-OCR evaluation do not import Paddle.
- [OCR and annotation behavior (#5)](https://github.com/OpenVeraTeam/TiGeSR/issues/5): fixed PP-OCRv5 selection, shared image resizing and BGR conversion, recorded cache source dimensions, and routed online/cached OCR through the same crop/restore implementation.
- [Inherited assets (#1, #2, #6)](weights.md): documented the DiffTSR base VAE, set-specific encoder, Hi-SAM, scheduler, SD3 and prompt embeddings. Added a path-mapping downloader, preflight checker, configurable asset paths and an empty-prompt cache generator. Restored the modified Hi-SAM runtime; complete checkpoints no longer unnecessarily open an additional SAM checkpoint, and partial checkpoints report precisely what is required.

Original stage-1 hyperparameters and executed loss formula are preserved. The original source evaluated a next-line `+ dice_loss(...)` expression without adding it to the loss; the restored training implementation retains the actual `20 * focal` behavior rather than silently changing the experiment.

## Validation performed

A new isolated Python 3.11.4 environment on Windows was installed with CPU PyTorch 2.5.1, torchvision 0.20.1 and the core/training dependency profiles.

- `python -m pip check`: no broken requirements.
- `python -m unittest discover -s tests -v`: 26 tests passed.
- Real offline imports: full inference pipeline, both training entry points and the degradation utility, without importing Paddle.
- Tests cover loss gradients into U-Net/encoder, annotation scaling/rotation and online/cache routing, Hi-SAM checkpoint completion, and prompt-cache validation.
- The published HF prompt cache was downloaded and its tensor keys, shapes and finite values checked.
- CLI help, downloader path preview, Python compilation and whitespace checks passed.

The GitHub workflow runs the CPU checks on Linux for future pushes and pull requests.

## Still requiring model-level validation

Full Linux/CUDA training, checkpoint-level inference, actual Paddle OCR execution, and PyIQA metric execution were not run in this audit. The [periodic-texture report (#7)](https://github.com/OpenVeraTeam/TiGeSR/issues/7) needs a reproducible image, command and output before a model-level cause can be established. The published `writing_nocfg` embeddings have no prompt metadata; their tensor contract is valid, but they must not be described as verified empty-prompt embeddings.
