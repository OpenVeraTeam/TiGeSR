"""Download released TIGER assets into the paths actually used by the code."""
import argparse
from pathlib import Path
import shutil

REPO = "mxluocv/TiGeSR"
SHARED = {
    "PNDMscheduler/scheduler_config.json": "DiffTSR/PNDMscheduler/scheduler_config.json",
    "SAM/BTS-test-test_best.pth": "BTS-test-test_best.pth",
    "weights/Stable-Diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt":
        "stable-diffusion-3.5-medium/saved_prompt_tokens_writing_nocfg.pt",
}


def asset_map(dataset):
    result = dict(SHARED)
    for suffix in ("stage1/model.pth", "stage1/encoder.pth", "stage2/model.pkl"):
        path = f"{dataset}/{suffix}"
        result[path] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uzst", "realce"), default="uzst")
    parser.add_argument("--output-dir", type=Path, default=Path("weights"))
    parser.add_argument("--list", action="store_true", help="Print paths without downloading.")
    args = parser.parse_args()
    for remote, relative in asset_map(args.dataset).items():
        destination = args.output_dir / relative
        print(f"{REPO}/{remote} -> {destination}")
        if args.list:
            continue
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(REPO, remote)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print("The upstream DiffTSR base checkpoint and licensed SD3.5 Medium model are separate downloads. See docs/weights.md.")


if __name__ == "__main__":
    main()
