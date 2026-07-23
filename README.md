# TiGeSR (TIGER): Restore Text First, Enhance Image Later

Official implementation of **TIGER** (Text-Image Guided supEr-Resolution), the two-stage scene-text image super-resolution framework from CVPR 2026. TIGER first restores text glyph structure (stage 1), then uses the restored structure to guide full-image enhancement (stage 2).

Paper: [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Restore_Text_First_Enhance_Image_Later_Two-Stage_Scene_Text_Image_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2510.21590)

## What is included

- `scripts/train_stage1.py`: glyph-restoration training.
- `scripts/train_stage2.py`: SD3/ControlNet enhancement training.
- `scripts/inference.py`: multi-GPU, end-to-end TIGER inference.
- `scripts/make_ocr_annotations.py`: optional OCR annotation cache generation.
- `scripts/evaluate.py`: PSNR, SSIM, LPIPS, and optional OCR accuracy (OCR-A).
- `models/`, `data/`, `utils/`, and `configs/stage1.yaml`: only code required by the above entry points.

Experiment outputs, raw datasets, checkpoints, obsolete ablations, font assets, and vendored PaddleOCR source are intentionally excluded.

## Installation

Use Linux with an NVIDIA GPU. Create an environment with Python 3.11, install the matching PyTorch/CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Download the Stable Diffusion 3.5 Medium files and place them at `weights/stable-diffusion-3.5-medium/`. Download the TIGER checkpoints from [Google Drive](https://drive.google.com/drive/folders/15EJ-1gXHF174wRA5Njwd84EqFkyqZKfx) and place them in `weights/`; they are excluded from git.

The [UZ-ST dataset](https://huggingface.co/datasets/mxluocv/UZ-ST) is available on Hugging Face. Both TIGER stage 1 and stage 2 released checkpoints are trained on UZ-ST.

## Checkpoint layout

The full pipeline needs three learned checkpoints:

```text
weights/
├── stage1_idm.pth       # stage-1 IDM / glyph restoration model
├── stage1_encoder.pth   # stage-1 text encoder
├── stage2_ctrlnet.pkl   # stage-2 ControlNet + VAE adaptation
└── stable-diffusion-3.5-medium/
```

### RealCE checkpoints

The RealCE checkpoint package is distributed separately and is not tracked by Git. Copy it into `weights/realce/` using this layout:

```text
weights/realce/
├── stage1/
│   ├── encoder-96000.pth
│   └── model-96000.pth
└── stage2/
    └── model_135200.pkl
```

For RealCE inference, pass `weights/realce/stage1/model-96000.pth` as `--idm_ckpt_path`, `weights/realce/stage1/encoder-96000.pth` as `--enc_path`, and `weights/realce/stage2/model_135200.pkl` as `--ctrlnet_ckpt_path`.

## End-to-end inference

Run from the repository root. With no annotations, the pipeline runs PaddleOCR itself. With `--anno_path`, it consumes cached annotations named `<image-stem>_res.json`.

```bash
python scripts/inference.py \
  --test_folder /path/to/lr_images \
  --output_dir outputs/demo \
  --difftsr_config configs/stage1.yaml \
  --idm_ckpt_path weights/stage1_idm.pth \
  --enc_path weights/stage1_encoder.pth \
  --ctrlnet_ckpt_path weights/stage2_ctrlnet.pkl \
  --timestep 150 --ctrl_layers 6 --enable_tiling
```

To prepare reusable annotations:

```bash
python scripts/make_ocr_annotations.py --input-dir /path/to/lr_images --output-dir /path/to/annotations
```

## Training

The launchers capture the paper settings (stage 1: 8 epochs, learning rate `5e-5`; stage 2: learning rate `5e-6`, `timestep=150`, `ctrl_layers=6`). Replace all `/path/to/...` placeholders before use:

```bash
bash scripts/train_stage1.sh
bash scripts/train_stage2.sh
```

Stage 1 accepts a comma-separated `--train_folders` list. Stage 2 expects paired data organized like the supplied Real-CE implementation: LR images in `13mm_aligned_1024/`, targets in `52mm_aligned_1024/`, and glyph masks in `mask_52mm_patchedx4/`. The released stage 1 and stage 2 checkpoints are trained on [UZ-ST](https://huggingface.co/datasets/mxluocv/UZ-ST); use its corresponding paired-folder layout and the matching dataset configuration from the original experiment setup.

## Evaluation

Image stems must match across prediction and ground-truth directories. Auxiliary `*_mask.png` and `*_subsr.png` files are ignored.

```bash
python scripts/evaluate.py --pred-dir outputs/demo --gt-dir /path/to/gt
```

For OCR-A, pass a JSON mapping such as `{ "0001.png": "expected text" }`:

```bash
python scripts/evaluate.py --pred-dir outputs/demo --gt-dir /path/to/gt --labels-json /path/to/labels.json
```

## Release checklist

- [ ] Publish SHA-256 checksums for each released checkpoint.
- [ ] Add detailed UZ-ST preprocessing instructions and usage terms.
- [ ] Add citation details, author/contact information, and a `NOTICE` file for all inherited third-party code.
- [ ] Run one clean-environment training smoke test and one end-to-end inference/evaluation run.

## License

The code is released under the Apache License 2.0. Third-party model weights and datasets remain subject to their own licenses.
