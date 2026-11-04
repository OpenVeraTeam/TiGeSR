"""Evaluate TIGER predictions using paired full-reference and OCR metrics.

The optional label JSON must be a mapping from filename (with or without its
extension) to its ground-truth transcription.
"""
import argparse
import json
from pathlib import Path



IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_index(directory: Path):
    return {path.stem: path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES and not path.stem.endswith(("_mask", "_subsr"))}


def normalized(text: str) -> str:
    return "".join(str(text).split()).lower()


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR, SSIM, LPIPS and optional OCR-A.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, help="Optional filename-to-text JSON for OCR-A.")
    parser.add_argument("--ocr-annotations", type=Path, help="Cached OCR on predicted images; avoids importing Paddle.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    import pyiqa
    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions, targets = image_index(args.pred_dir), image_index(args.gt_dir)
    shared = sorted(predictions.keys() & targets.keys())
    if not shared:
        raise RuntimeError("No matching prediction/ground-truth image stems were found.")

    metrics = {name: pyiqa.create_metric(name, device=args.device) for name in ("psnr", "ssim", "lpips")}
    totals = {name: 0.0 for name in metrics}
    for stem in shared:
        for name, metric in metrics.items():
            totals[name] += float(metric(str(predictions[stem]), str(targets[stem])).item())

    result = {name: value / len(shared) for name, value in totals.items()}
    result["num_paired_images"] = len(shared)

    if args.labels_json:
        labels = json.loads(args.labels_json.read_text(encoding="utf-8"))
        ocr = None
        if args.ocr_annotations is None:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(ocr_version="PP-OCRv5", use_doc_orientation_classify=True, use_doc_unwarping=False, use_textline_orientation=True, device="cpu")
        correct = evaluated = 0
        for stem in shared:
            target = labels.get(predictions[stem].name, labels.get(stem))
            if target is None:
                continue
            if args.ocr_annotations is not None:
                cache = args.ocr_annotations / f"{stem}_res.json"
                record = json.loads(cache.read_text(encoding="utf-8"))
            else:
                record = ocr.predict(str(predictions[stem]))[0]
            prediction = "".join(record.get("rec_texts", []))
            correct += normalized(prediction) == normalized(target)
            evaluated += 1
        if evaluated:
            result["ocr_a"] = correct / evaluated
            result["num_ocr_images"] = evaluated

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
