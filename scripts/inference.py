"""Multi-GPU TIGER inference, optionally using a Paddle-free annotation cache."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.annotations import inference_size, resize_annotation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_folder", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--difftsr_config", default="configs/stage1.yaml")
    parser.add_argument("--idm_ckpt_path", required=True)
    parser.add_argument("--enc_path", required=True)
    parser.add_argument("--ctrlnet_ckpt_path", required=True)
    parser.add_argument("--difftsr_ckpt_path", default="weights/DiffTSR/DiffTSR_new.ckpt")
    parser.add_argument("--scheduler_path", default="weights/DiffTSR/PNDMscheduler")
    parser.add_argument("--hi_sam_ckpt", default="weights/BTS-test-test_best.pth")
    parser.add_argument("--sd3_dir", default="weights/stable-diffusion-3.5-medium")
    parser.add_argument("--prompt_cache_path", default=None)
    parser.add_argument("--anno_path", type=Path)
    parser.add_argument("--ocr_device", default="cpu")
    parser.add_argument("--max_side", type=int, default=1024)
    parser.add_argument("--timestep", type=float, default=150)
    parser.add_argument("--ctrl_layers", type=int, default=6)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def process_images(rank, world_size, args, img_names):
    import torch
    import torchvision.transforms.functional as tF
    from torchvision.utils import save_image
    from PIL import Image
    from models.FullPipeline import TextSR_multiscale

    torch.cuda.set_device(rank)
    torch.manual_seed(args.seed + rank)
    net_sr = TextSR_multiscale(
        args.difftsr_config,
        target_timestep=args.timestep,
        ckpt_dir=args.sd3_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        num_ctrl_layers=args.ctrl_layers,
        difftsr_ckpt_path=args.difftsr_ckpt_path,
        scheduler_path=args.scheduler_path,
        hi_sam_ckpt=args.hi_sam_ckpt,
        prompt_cache_path=args.prompt_cache_path,
    )
    net_sr.ocr_device = args.ocr_device
    net_sr.load_model(args.idm_ckpt_path, args.ctrlnet_ckpt_path, args.enc_path)
    net_sr.set_eval()
    net_sr.to_device(torch.device(f"cuda:{rank}"))

    for img_name in img_names[rank::world_size]:
        stem = Path(img_name).stem
        output_path = args.output_dir / f"{stem}.png"
        if output_path.exists():
            continue
        with Image.open(args.test_folder / img_name) as source:
            image = source.convert("RGB")
            original_size = image.size
            target_size = inference_size(*original_size, args.max_side)
            image = image.resize(target_size)
        input_img = tF.to_tensor(image).unsqueeze(0).to(f"cuda:{rank}")
        anno = None
        if args.anno_path is not None:
            anno_file = args.anno_path / f"{stem}_res.json"
            anno = json.loads(anno_file.read_text(encoding="utf-8"))
            anno = resize_annotation(anno, original_size, target_size)
        with torch.no_grad():
            sr_img, mask, sub_sr = net_sr.inference(input_img, anno=anno, tiling=args.enable_tiling)
        save_image(sr_img.cpu(), output_path)
        save_image(mask.cpu(), args.output_dir / f"{stem}_mask.png")
        save_image(sub_sr.cpu(), args.output_dir / f"{stem}_subsr.png")
        gc.collect()
        torch.cuda.empty_cache()


def main(args):
    required = [Path(path) for path in (args.idm_ckpt_path, args.enc_path,
                args.ctrlnet_ckpt_path, args.difftsr_ckpt_path, args.hi_sam_ckpt,
                args.difftsr_config)]
    required += [Path(args.scheduler_path) / "scheduler_config.json",
                 Path(args.prompt_cache_path) if args.prompt_cache_path else
                 Path(args.sd3_dir) / "saved_prompt_tokens_null_nocfg.pt"]
    required += [Path(args.sd3_dir) / sub / name for sub, name in
                 (("transformer", "config.json"), ("vae", "config.json"),
                  ("scheduler", "scheduler_config.json"))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model assets:\n  " + "\n  ".join(missing) +
                                "\nSee docs/weights.md and scripts/check_setup.py.")
    import torch
    import torch.multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("TIGER inference requires an NVIDIA CUDA GPU.")
    image_names = sorted(p.name for p in args.test_folder.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not image_names:
        raise ValueError(f"No input images found in {args.test_folder}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world_size = min(num_gpus, len(image_names))
    mp.spawn(process_images, args=(world_size, args, image_names), nprocs=world_size, join=True)


if __name__ == "__main__":
    main(parse_args())
