<div align="center">

# TiGeSR: Restore Text First, Enhance Image Later

<img src="assets/tiger-mascot.png" alt="TIGER mascot" width="136" />

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

- **2026.07** — Code, inference, training, evaluation, UZ-ST alignment utilities, and model links are released.
- **2026.06** — TiGeSR is accepted to CVPR 2026.

## Overview

<p align="center">
  <img src="assets/pipeline.png" alt="The two-stage TiGeSR pipeline: glyph restoration followed by glyph-guided image enhancement" width="100%" />
</p>

The first stage reconstructs legible glyph structures from a low-quality scene image. The second stage conditions an image enhancer on both the input and the restored glyph mask, retaining realistic image content while protecting text readability.

## Dependencies and installation

Tested on Linux with NVIDIA GPUs. Create a Python environment, install a PyTorch build compatible with your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
git clone git@github.com:OpenVeraTeam/TiGeSR.git
cd TiGeSR

conda create -n tiger python=3.11 -y
conda activate tiger
pip install -r requirements.txt
```

Download the Stable Diffusion 3.5 Medium files and place them in `weights/stable-diffusion-3.5-medium/`.

## Pre-trained models

We provide two internally matched checkpoint sets in the [TiGeSR Hugging Face model repository](https://huggingface.co/mxluocv/TiGeSR):

- **UZ-ST**: stage 1 and stage 2 trained on [UZ-ST](https://huggingface.co/datasets/mxluocv/UZ-ST).
- **RealCE**: stage 1 and stage 2 trained on RealCE; its stage-1 model is the pruned RoPE variant.

Download exactly one matching set. Do not mix stage-1 and stage-2 weights across sets.

```bash
# UZ-ST-trained checkpoints
hf download mxluocv/TiGeSR --include "uzst/**" --local-dir weights

# RealCE-trained checkpoints
hf download mxluocv/TiGeSR --include "realce/**" --local-dir weights
```

The expected layout is:

```text
weights/
├── uzst/
│   ├── stage1/
│   │   ├── model.pth
│   │   └── encoder.pth
│   └── stage2/model.pkl
├── realce/
│   ├── stage1/
│   │   ├── model.pth
│   │   └── encoder.pth
│   └── stage2/model.pkl
└── stable-diffusion-3.5-medium/
```

Use `configs/stage1.yaml` with `weights/uzst/`. Use `configs/realce_stage1.yaml` with `weights/realce/`; this is the released pruned RoPE configuration (`RoPE2Out_mask_pruned_config_hybridwEncoder.yaml`).

## ⚡ Inference

### Run the full TIGER pipeline

Run from the repository root. Without `--anno_path`, TIGER runs PP-OCRv5 internally. The following UZ-ST command is the default example:

```bash
python scripts/inference.py \
  --test_folder /path/to/lr_images \
  --output_dir outputs/uzst_demo \
  --difftsr_config configs/stage1.yaml \
  --idm_ckpt_path weights/uzst/stage1/model.pth \
  --enc_path weights/uzst/stage1/encoder.pth \
  --ctrlnet_ckpt_path weights/uzst/stage2/model.pkl \
  --timestep 150 --ctrl_layers 6 --enable_tiling
```

For RealCE, only switch to the matching checkpoint paths and pruned stage-1 config:

```bash
python scripts/inference.py \
  --test_folder /path/to/lr_images \
  --output_dir outputs/realce_demo \
  --difftsr_config configs/realce_stage1.yaml \
  --idm_ckpt_path weights/realce/stage1/model.pth \
  --enc_path weights/realce/stage1/encoder.pth \
  --ctrlnet_ckpt_path weights/realce/stage2/model.pkl \
  --timestep 150 --ctrl_layers 6 --enable_tiling
```

### Use your own PP-OCRv5 annotations

Create reusable annotations from the integrated OCR path:

```bash
python scripts/make_ocr_annotations.py \
  --input-dir /path/to/lr_images \
  --output-dir /path/to/annotations
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

`dt_polys` must be in pixels of the original input image. `rec_scores` and `textline_orientation_angles` are optional. TIGER supplies defaults of `1.0` and `0` when absent, and scales polygons internally when tiling is enabled.

## Train

The launchers preserve the original experiment hyperparameters. Replace only the dataset and output paths in the scripts before launching.

```bash
# Stage 1, pruned RoPE (RealCE configuration)
bash scripts/train_stage1.sh

# Stage 1, unpruned RoPE (UZ-ST configuration)
bash scripts/train_stage1_unpruned.sh

# Stage 2, glyph-guided image enhancement
bash scripts/train_stage2.sh
```

Stage 1 accepts a comma-separated `--train_folders` list. For stage 2, prepare paired folders following the released RealCE convention: LR images in `13mm_aligned_1024/`, targets in `52mm_aligned_1024/`, and glyph masks in `mask_52mm_patchedx4/`. Select `configs/stage1.yaml` for UZ-ST or `configs/realce_stage1.yaml` for RealCE; stage 2 consumes stage-1 glyph guidance rather than this YAML directly.

## Evaluation

Prediction and ground-truth filenames must share the same image stem. Auxiliary `*_mask.png` and `*_subsr.png` files are ignored.

```bash
python scripts/evaluate.py \
  --pred-dir outputs/uzst_demo \
  --gt-dir /path/to/ground_truth
```

For OCR accuracy (OCR-A), pass a JSON map from filename to expected text:

```bash
python scripts/evaluate.py \
  --pred-dir outputs/uzst_demo \
  --gt-dir /path/to/ground_truth \
  --labels-json /path/to/labels.json
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

This release builds on the open-source ecosystems of [Diffusers](https://github.com/huggingface/diffusers), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), and their dependencies. We thank their authors and contributors.

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
