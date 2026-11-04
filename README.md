<div align="center">

<img src="assets/tiger-mascot.png" alt="TIGER mascot" width="136" />

# TiGeSR: Restore Text First, Enhance Image Later

**[CVPR 2026] Restore Text First, Enhance Image Later: Two-Stage Scene Text Image Super-Resolution with Glyph Structure Guidance**

Minxing Luo<sup>1,2,*</sup> · Linlong Fan<sup>2,*</sup> · Qiushi Wang<sup>2,3</sup> · Ge Wu<sup>1</sup> · Yiyan Luo<sup>2</sup> · Yuhang Yu<sup>2</sup> · Jinwei Chen<sup>2</sup> · Yaxing Wang<sup>1,4</sup> · Qingnan Fan<sup>2,†</sup> · Jian Yang<sup>1,5,†</sup>

<sup>1</sup> PCA Lab, VCIP, College of Computer Science, Nankai University · <sup>2</sup> vivo BlueImage Lab, vivo Mobile Communication Co., Ltd. · <sup>3</sup> SDS, CUHK, Shenzhen · <sup>4</sup> NKIARI, Shenzhen Futian · <sup>5</sup> PCA Lab, School of Intelligence Science and Technology, Nanjing University

<a href="https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Restore_Text_First_Enhance_Image_Later_Two-Stage_Scene_Text_Image_CVPR_2026_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202026-b31b1b?style=flat-square" alt="CVPR paper" /></a>
<a href="https://arxiv.org/abs/2510.21590"><img src="https://img.shields.io/badge/arXiv-2510.21590-b31b1b?style=flat-square" alt="arXiv" /></a>
<a href="https://huggingface.co/mxluocv/TiGeSR"><img src="https://img.shields.io/badge/Models-Hugging%20Face-ff9d00?style=flat-square" alt="Models" /></a>
<a href="https://huggingface.co/datasets/mxluocv/UZ-ST"><img src="https://img.shields.io/badge/Dataset-UZ--ST-ff9d00?style=flat-square" alt="UZ-ST" /></a>
<a href="https://tony-lowe.github.io/TIGER_project_page/"><img src="https://img.shields.io/badge/Project-Page-267a54?style=flat-square" alt="Project page" /></a>

</div>

> **TIGER** (Text-Image Guided supEr-Resolution) first restores text glyph structure, then uses the restored glyph prior to guide full-image enhancement. It targets scene-text images where both character fidelity and natural scene detail matter.

<p align="center">
  <img src="assets/teaser.png" alt="TiGeSR restores readable scene text while preserving image content" width="100%" />
</p>

## News

- **2026.11** — Restore omitted training/loss and Hi-SAM code; separate model/OCR/evaluation dependencies; align cached and online OCR geometry. See [release fixes and validation](docs/release_fixes.md), [installation](docs/installation.md) and [model assets](docs/weights.md).
- **2026.07** — Code, inference, training, evaluation, UZ-ST alignment utilities, and model links are released.
- **CVPR 2026** — TiGeSR paper.

## Overview

<p align="center">
  <img src="assets/pipeline.png" alt="The two-stage TiGeSR pipeline: glyph restoration followed by glyph-guided image enhancement" width="100%" />
</p>

The first stage reconstructs legible glyph structures from a low-quality scene image. The second stage conditions an image enhancer on both the input and the restored glyph mask, retaining realistic image content while protecting text readability.

## Dependencies and installation

Use Python 3.11 and separate the PyTorch model environment from the PaddleOCR environment. This avoids installing competing CUDA/NCCL packages into one environment.

**H20 / H800 users:** the author tested both GPUs and encountered Paddle GPU runtime failures on both. Use the separate **CPU PP-OCRv5 environment**, then supply its annotations via `--anno_path` for GPU inference. See the [H20 / H800 deployment note](docs/installation.md#2-pp-ocrv5-annotation-environment).

```bash
git clone git@github.com:OpenVeraTeam/TiGeSR.git
cd TiGeSR
conda create -n tiger python=3.11 -y
conda activate tiger
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

See [installation instructions](docs/installation.md) for the separate OCR/evaluation environments and optional training/data packages. The pinned versions are a compatibility baseline; Linux GPU reproduction still needs a full checkpoint run.

## Pre-trained models

The [TiGeSR model repository](https://huggingface.co/mxluocv/TiGeSR) provides **UZ-ST** and **RealCE** checkpoint sets. Download one matching set together with the shared Hi-SAM, scheduler and prompt-cache assets:

```bash
python scripts/download_weights.py --dataset uzst
# Or: python scripts/download_weights.py --dataset realce
```

Also obtain the upstream DiffTSR base checkpoint (for its VAE) and Stable Diffusion 3.5 Medium. Follow the complete [asset list and path instructions](docs/weights.md). A separate `Bert_only.ckpt` is unnecessary when using the set-specific `encoder.pth`.

```text
weights/
├── DiffTSR/
│   ├── DiffTSR_new.ckpt
│   └── PNDMscheduler/scheduler_config.json
├── BTS-test-test_best.pth
├── uzst/
│   ├── stage1/{model.pth,encoder.pth}
│   └── stage2/model.pkl
├── realce/
│   ├── stage1/{model.pth,encoder.pth}
│   └── stage2/model.pkl
└── stable-diffusion-3.5-medium/
    ├── transformer/
    ├── vae/
    ├── scheduler/
    └── saved_prompt_tokens_writing_nocfg.pt
```

Use `configs/stage1.yaml` with `weights/uzst/` and the pruned `configs/realce_stage1.yaml` with `weights/realce/`. Keep each stage-1, encoder and stage-2 set together.

The released cache is named `writing_nocfg`; select it explicitly in the commands below. To instead generate the original code's empty-prompt cache, use [the preparation script](docs/weights.md#sd35-medium-and-prompt-embeddings).

```bash
python scripts/check_setup.py --dataset uzst \
  --prompt-cache weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt
```

## ⚡ Inference

### Run the full TIGER pipeline

Run from the repository root. Generate annotations in the separate OCR environment first, then run the following command in the PyTorch environment. Omitting `--anno_path` enables online OCR and requires PaddleOCR in that environment.

```bash
python scripts/inference.py \
  --test_folder /path/to/lr_images \
  --anno_path /path/to/annotations \
  --output_dir outputs/uzst_demo \
  --difftsr_config configs/stage1.yaml \
  --idm_ckpt_path weights/uzst/stage1/model.pth \
  --enc_path weights/uzst/stage1/encoder.pth \
  --ctrlnet_ckpt_path weights/uzst/stage2/model.pkl \
  --prompt_cache_path weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt \
  --timestep 150 --ctrl_layers 6 --enable_tiling
```

For RealCE, only switch to the matching checkpoint paths and pruned stage-1 config:

```bash
python scripts/inference.py \
  --test_folder /path/to/lr_images \
  --anno_path /path/to/annotations \
  --output_dir outputs/realce_demo \
  --difftsr_config configs/realce_stage1.yaml \
  --idm_ckpt_path weights/realce/stage1/model.pth \
  --enc_path weights/realce/stage1/encoder.pth \
  --ctrlnet_ckpt_path weights/realce/stage2/model.pkl \
  --prompt_cache_path weights/stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt \
  --timestep 150 --ctrl_layers 6 --enable_tiling
```

### Use your own PP-OCRv5 annotations

In the [OCR environment](docs/installation.md#2-pp-ocrv5-annotation-environment), create reusable annotations using the same image preparation and PP-OCRv5 model as online inference:

```bash
python scripts/make_ocr_annotations.py \
  --input-dir /path/to/lr_images \
  --output-dir /path/to/annotations --device cpu
```

You may also supply manually corrected annotations or annotations produced by [PaddleOCR / PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) or [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch). Use one file per image at `<annotation_dir>/<image-stem>_res.json`, then add `--anno_path /path/to/annotations` to inference.

```json
{
  "doc_preprocessor_res": {"angle": 0},
  "dt_polys": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],
  "rec_texts": ["recognized text"],
  "rec_scores": [0.99],
  "textline_orientation_angles": [0]
}
```

For manual JSON, `dt_polys` refer to the original image after the specified document rotation. New caches record the pre-rotation image size in `tiger_image_size`; TIGER uses it to scale coordinates to the inference image. Legacy JSON without this metadata is assumed to describe the original input. `rec_scores` and `textline_orientation_angles` are optional (defaults: `1.0` and `0`). Cached and online OCR now use the same crop/restore code.

## Train

The launchers preserve the original experiment hyperparameters. Install `requirements/train.txt`, then replace the dataset/checkpoint/output paths in the scripts. The training-specific YAMLs select the original four-loss stage-1 implementation.

```bash
# Stage 1, pruned RoPE (RealCE configuration)
bash scripts/train_stage1.sh

# Stage 1, unpruned RoPE (UZ-ST configuration)
bash scripts/train_stage1_unpruned.sh

# Stage 2, glyph-guided image enhancement
bash scripts/train_stage2.sh
```

Stage 1 accepts a comma-separated `--train_folders` list. For stage 2, prepare paired folders following the released RealCE convention: LR images in `13mm_aligned_1024/`, targets in `52mm_aligned_1024/`, and glyph masks in `mask_52mm_patchedx4/`. Stage-1 training uses `configs/stage1_train.yaml` or `configs/realce_stage1_train.yaml`; the shorter YAML names above are for inference. Stage 2 consumes stage-1 glyph guidance. Its `--sd3_dir` and `--prompt_cache_path` options select the SD3 model and prompt embeddings; all original optimization parameters are retained.

## Evaluation

Use the separate [evaluation environment](docs/installation.md#4-image-quality-evaluation-environment), because PyIQA pins an older Transformers version. Prediction and ground-truth filenames must share the same image stem. Auxiliary `*_mask.png` and `*_subsr.png` files are ignored.

```bash
python scripts/evaluate.py \
  --pred-dir outputs/uzst_demo \
  --gt-dir /path/to/ground_truth
```

For OCR accuracy (OCR-A), run the annotation script on the predictions in the OCR environment, then pass the cached results and a filename-to-expected-text JSON map:

```bash
python scripts/evaluate.py \
  --pred-dir outputs/uzst_demo \
  --gt-dir /path/to/ground_truth \
  --labels-json /path/to/labels.json \
  --ocr-annotations /path/to/prediction_annotations
```

## Data preparation utilities

### Synthetic degradation

`data/deg_mine.py` implements the synthetic degradation used in our experiments: two-stage Real-ESRGAN-style blur, noise, JPEG compression, and bicubic upsampling. It writes low-quality images to `lr_normal_blur_kernel/` under the dataset root.

```bash
python data/deg_mine.py --root_path /path/to/rendered_dataset
```

### UZ-ST alignment and color correction

[`tools/uzst_alignment/`](tools/uzst_alignment/) contains the UZ-ST preparation utilities used for SIFT/MAGSAC alignment, LoFTR, RIPE, and color fixing.

```bash
pip install -r tools/uzst_alignment/requirements.txt
```

See [`tools/uzst_alignment/README.md`](tools/uzst_alignment/README.md) for script roles, the optional RIPE dependency, and path configuration notes.

## Acknowledgements

This release includes modified [DiffTSR](https://github.com/YuzheZhang-1999/DiffTSR) and [Hi-SAM](https://github.com/ymy-k/Hi-SAM) code and builds on [Diffusers](https://github.com/huggingface/diffusers), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and their dependencies. We thank their authors and contributors.

## License

The code is released under the [Apache License 2.0](LICENSE). Third-party model weights and datasets remain subject to their respective licenses and terms of use.

## Citation

If you find TiGeSR useful for your research, please cite:

```bibtex
@inproceedings{luo2026restore,
  title={Restore text first, enhance image later: Two-stage scene text image super-resolution with glyph structure guidance},
  author={Luo, Minxing and Fan, Linlong and Wang, Qiushi and Wu, Ge and Luo, Yiyan and Yu, Yuhang and Chen, Jinwei and Wang, Yaxing and Fan, Qingnan and Yang, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={30553--30563},
  year={2026}
}
```
