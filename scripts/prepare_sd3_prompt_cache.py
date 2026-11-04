"""Cache SD3's encoded empty prompt, as used by the original TiGeSR training code.

This requires the official model's two CLIP encoders and T5 encoder. It does not
download or load the image transformer or VAE. The output is not a zero tensor:
an empty string still produces learned text-encoder features.
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="weights/stable-diffusion-3.5-medium")
    parser.add_argument("--output", help="Defaults to <model-dir>/saved_prompt_tokens_null_nocfg.pt")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda; text encoders require substantial memory")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output) if args.output else Path(args.model_dir) / "saved_prompt_tokens_null_nocfg.pt"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to regenerate it.")

    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype = getattr(torch, args.dtype)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if pipeline.text_encoder_3 is None:
        raise ValueError("The full T5 text encoder is required; a CLIP-only cache changes TiGeSR conditioning.")
    pipeline.to(args.device)
    with torch.inference_mode():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=torch.device(args.device),
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
    cache = {
        "prompt_embeds": prompt_embeds.detach().float().cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().float().cpu(),
        "metadata": {
            "model": args.model_dir,
            "prompt": "",
            "max_sequence_length": 256,
            "do_classifier_free_guidance": False,
            "generation_dtype": args.dtype,
        },
    }
    for key in ("prompt_embeds", "pooled_prompt_embeds"):
        if not torch.isfinite(cache[key]).all():
            raise ValueError(f"Generated non-finite {key}; try --dtype float32.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    print(f"Saved {output}")
    for key in ("prompt_embeds", "pooled_prompt_embeds"):
        print(f"  {key}: {tuple(cache[key].shape)}")


if __name__ == "__main__":
    main()
