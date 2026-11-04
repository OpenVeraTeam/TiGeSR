"""Create PP-OCRv5 caches in a separate OCR-only environment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.annotations import inference_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-side", type=int, default=1024)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ocr = PaddleOCR(ocr_version="PP-OCRv5", use_doc_orientation_classify=True,
                    use_doc_unwarping=False, use_textline_orientation=True, device=args.device)
    for path in images:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            size = inference_size(*rgb.size, args.max_side)
            rgb = rgb.resize(size)
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        record = ocr.predict(bgr)[0]
        result = {
            "doc_preprocessor_res": {"angle": int(record.get("doc_preprocessor_res", {}).get("angle", 0))},
            "dt_polys": np.asarray(record.get("dt_polys", [])).tolist(),
            "rec_texts": list(record.get("rec_texts", [])),
            "rec_scores": np.asarray(record.get("rec_scores", [])).tolist(),
            "textline_orientation_angles": np.asarray(record.get("textline_orientation_angles", [])).tolist(),
            "tiger_image_size": list(size),
            "ocr_version": "PP-OCRv5",
        }
        (args.output_dir / f"{path.stem}_res.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
