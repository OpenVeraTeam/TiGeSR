# Model assets and paths

The two stage checkpoints alone are not the complete runtime. Run from the repository root:

```bash
python scripts/download_weights.py --dataset uzst
# Or: --dataset realce
```

The downloader maps the existing Hugging Face layout to the paths used by the released code. `--list` previews the mapping without network access.

| Asset | Local path | Source |
| --- | --- | --- |
| Stage 1 denoiser | `weights/<set>/stage1/model.pth` | TiGeSR Hugging Face |
| Stage 1 text encoder | `weights/<set>/stage1/encoder.pth` | TiGeSR Hugging Face |
| Stage 2 enhancer | `weights/<set>/stage2/model.pkl` | TiGeSR Hugging Face |
| Hi-SAM | `weights/BTS-test-test_best.pth` | `SAM/` in TiGeSR Hugging Face |
| Stage 1 scheduler config | `weights/DiffTSR/PNDMscheduler/scheduler_config.json` | `PNDMscheduler/` in TiGeSR Hugging Face |
| Released SD3 prompt embeddings | `weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt` | TiGeSR Hugging Face |
| Stage 1 base VAE | `weights/DiffTSR/DiffTSR_new.ckpt` | Upstream DiffTSR checkpoint |
| SD3.5 Medium | `weights/stable-diffusion-3.5-medium/{transformer,vae,scheduler,...}` | Stability AI model repository |

## DiffTSR base model

Download the official checkpoint from [DiffTSR's checkpoint instructions](https://github.com/YuzheZhang-1999/DiffTSR/tree/main/ckpt). Place it at `weights/DiffTSR/DiffTSR_new.ckpt`, or pass its actual path using `--difftsr_ckpt_path`. The active TIGER loader only reads its `VAE_model` component, so the original checkpoint is usable without renaming the unrelated TDM axial-position keys.

For the released inference examples, `--enc_path` points to our set-specific `encoder.pth`. A separate `Bert_only.ckpt` or `transocr.pth` is not needed: PP-OCRv5 supplies the transcription and the stage-1 text helper only tokenizes it. The optional inherited TransOCR prediction API still requires its original weights if used directly.

## Hi-SAM checkpoint completeness

The loader uses the trained Hi-SAM checkpoint directly when it includes all model tensors. If a checkpoint omits SAM backbone tensors, download the matching [official SAM model](https://github.com/facebookresearch/segment-anything#model-checkpoints). For the default `vit_h` model, place `sam_vit_h_4b8939.pth` in `weights/`. Direct Hi-SAM builder callers may instead provide `args.sam_checkpoint`; `build_sam_vit_h_parallel` also exposes this path as its `sam_checkpoint` argument.

Only missing tensors are filled from the original SAM checkpoint. Trained Hi-SAM tensors are preserved, and incomplete model parameters raise an explicit error rather than falling back to random initialization.

## SD3.5 Medium and prompt embeddings

Download [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) under its model terms. Keep its `transformer/`, `vae/`, and `scheduler/` subfolders together.

The HF release has a cache named `saved_prompt_tokens_writing_nocfg.pt`; its filename does not establish that it contains an empty prompt. Select it explicitly to use that released cache:

```bash
--prompt_cache_path weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt
```

Alternatively, to reconstruct the original code's empty-prompt path, prepare the cache using the complete SD3 text encoders/tokenizers (more memory is required for this one-time step):

```bash
python scripts/prepare_sd3_prompt_cache.py \
  --model-dir weights/stable-diffusion-3.5-medium
```

This creates `saved_prompt_tokens_null_nocfg.pt`. It is not an alias for the writing cache. Selecting either cache changes the text conditioning; keep it fixed when comparing results.

## Check before inference

```bash
python scripts/check_setup.py --dataset uzst \
  --prompt-cache weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt
```

The checker lists all missing modules and asset paths before any large GPU model is constructed. Runtime imports and actual GPU execution are separate checks.
