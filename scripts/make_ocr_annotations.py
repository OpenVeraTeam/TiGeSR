"""Create cached PaddleOCR annotations consumed by full-pipeline inference."""
import argparse
from pathlib import Path

from paddleocr import PaddleOCR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=False, use_textline_orientation=True, device=args.device)
    for result in ocr.predict([str(path) for path in images]):
        result.save_to_json(str(args.output_dir))


if __name__ == "__main__":
    main()
