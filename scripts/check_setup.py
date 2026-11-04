"""Check installed modules and required assets before allocating GPU models."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uzst", "realce"), default="uzst")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--difftsr-ckpt", type=Path)
    parser.add_argument("--prompt-cache", type=Path)
    parser.add_argument("--online-ocr", action="store_true")
    args = parser.parse_args()
    missing = []
    names = ["torch", "torchvision", "diffusers", "transformers", "peft", "omegaconf", "kornia", "einops", "cv2", "pytorch_lightning", "linear_attention_transformer", "axial_positional_embedding"]
    if args.online_ocr:
        names += ["paddleocr"]
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(f"Python module: {name}")
    root = args.weights_dir
    required = [
        root / args.dataset / "stage1/model.pth",
        root / args.dataset / "stage1/encoder.pth",
        root / args.dataset / "stage2/model.pkl",
        root / "BTS-test-test_best.pth",
        root / "DiffTSR/PNDMscheduler/scheduler_config.json",
        args.difftsr_ckpt or root / "DiffTSR/DiffTSR_new.ckpt",
        args.prompt_cache or root / "stable-diffusion-3.5-medium/saved_prompt_tokens_null_nocfg.pt",
    ]
    required += [root / "stable-diffusion-3.5-medium" / sub / name
                 for sub, name in (("transformer", "config.json"), ("vae", "config.json"), ("scheduler", "scheduler_config.json"))]
    for path in required:
        if not path.is_file():
            missing.append(f"Asset: {path}")
    print(f"Python: {sys.version.split()[0]}")
    for item in missing:
        print(f"MISSING {item}")
    if missing:
        print("See docs/installation.md and docs/weights.md. No GPU models were loaded.")
        return 1
    print("Required module names and asset paths are present. This does not run inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
