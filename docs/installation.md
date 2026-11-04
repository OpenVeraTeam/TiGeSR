# Installation and dependency scopes

Use Python 3.11. The inference/training code runs with PyTorch; OCR annotation generation runs with PaddlePaddle. The recommended workflow uses separate environments and exchanges PP-OCRv5 JSON files. This prevents the Paddle GPU package from replacing PyTorch's CUDA/NCCL dependencies.

The files below pin direct dependencies to a compatibility baseline. They are not a full transitive lockfile. GPU training and checkpoint-level end-to-end inference still require validation on the target Linux/CUDA machine; dependency metadata checks alone do not establish reproduction.

## 1. PyTorch inference environment

Run from the repository root:

```bash
conda create -n tiger python=3.11 -y
conda activate tiger
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch 2.5.1 and torchvision 0.20.1 are an official matching pair. Select another wheel index from the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v251) if CUDA 12.1 is unsuitable for your driver. Keep the PyTorch/torchvision pair together. This release does not require xFormers.

`requirements.txt` includes `requirements/core.txt`: diffusion models, stage-1 text-model dependencies, image processing, and tensor utilities. It does not install PaddlePaddle, PaddleOCR, BasicSR, or PyIQA. Passing `--anno_path` to inference uses existing OCR annotations and requires no Paddle runtime.

Download and prepare **all** [model assets](weights.md), including the inherited stage-1 base assets. Installing Python packages does not download project checkpoints. Select the published prompt cache explicitly or generate the original empty-prompt cache as described there.

## 2. PP-OCRv5 annotation environment

**H20 / H800 deployment note.** The author tested Paddle GPU on both H20 and H800 and encountered runtime failures on both in the tested environments. For these deployments, use the CPU OCR environment below, save PP-OCRv5 annotations, and run TiGeSR on the GPU with `--anno_path`. This keeps OCR independent of the model's GPU runtime. The exact Paddle/CUDA/driver combinations from those tests have not been recorded here. This report describes the author's observed failures; the [official Paddle compatibility table](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html) lists Hopper support with suitable CUDA builds, and other combinations require separate validation.

Use a second environment:

```bash
conda create -n tiger-ocr python=3.11 -y
conda activate tiger-ocr
python -m pip install --upgrade pip
python -m pip install -r requirements/ocr.txt
python -m pip check

python scripts/make_ocr_annotations.py \
  --input-dir /path/to/lr_images \
  --output-dir /path/to/annotations \
  --device cpu
```

This file uses PaddlePaddle 3.1.1 **CPU** and PaddleOCR 3.2.0, whose OCR pipeline supports PP-OCRv5. Its [`ocr-core` dependency scope](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.2.0/pyproject.toml) avoids installing unrelated document and language-model pipelines. The pinned version makes the OCR behavior more reproducible than an unrestricted upgrade. Refer to the [PaddleOCR installation guide](https://www.paddleocr.ai/v3.2.0/en/version3.x/installation.html) for other supported deployments. If GPU OCR is required, replace the CPU Paddle package with the GPU build appropriate for that machine **inside `tiger-ocr`**. Do not install both Paddle CPU/GPU wheels in one environment.

Then return to `tiger` and add the annotation directory to either inference example in the main README:

```bash
conda activate tiger
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

For manually prepared JSON, follow the PP-OCRv5 annotation schema in the [README](../README.md). Coordinates refer to the document-orientation-corrected input frame. The cache generator records the pre-rotation size in `tiger_image_size` so internal image resizing can scale the polygons correctly. Rebuild old caches if their source resolution is unknown.

## 3. Training and synthetic data

```bash
conda activate tiger
python -m pip install -r requirements/train.txt
python -m pip check
```

This adds LPIPS, TensorBoard, and Matplotlib. The training launchers and hyperparameters remain in `scripts/train_stage1.sh`, `scripts/train_stage1_unpruned.sh`, and `scripts/train_stage2.sh`.

For synthetic degradation without the training extras:

```bash
python -m pip install -r requirements/data.txt
python data/deg_mine.py --root_path /path/to/rendered_dataset
```

SciPy is pinned below 1.14 because the inherited BSRGAN preparation code uses `interp2d`. Optional UZ-ST geometric alignment utilities have their own [requirements and instructions](../tools/uzst_alignment/README.md).

## 4. Image-quality evaluation environment

PyIQA 0.1.13 explicitly requires `transformers==4.37.2`; installing it into the model environment would conflict with the diffusion/PEFT stack. Keep evaluation separate:

```bash
conda create -n tiger-eval python=3.11 -y
conda activate tiger-eval
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/eval.txt
python -m pip check
python scripts/evaluate.py --pred-dir outputs/uzst_demo --gt-dir /path/to/ground_truth
```

PyIQA may download metric-specific pretrained weights on its first run. For OCR-A, generate annotations for the **predicted** images in `tiger-ocr`, then use the cached-annotation option in the evaluation command described in the README.

## What changed from the original all-in-one requirements

- Restored omitted stage-1 Python dependencies: PyTorch Lightning, axial positional embeddings, linear attention, and `zhconv`.
- Removed the default `paddlepaddle-gpu` installation from the PyTorch environment.
- Isolated PyIQA's incompatible Transformers pin in an evaluation environment.
- Replaced open-ended dependency lower bounds with a pinned direct-dependency baseline.
- Kept training, data preparation, OCR, evaluation, and optional alignment dependencies explicit.
