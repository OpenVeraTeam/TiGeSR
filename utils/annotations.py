"""PP-OCRv5 geometry shared by cached and online inference (no Paddle import)."""
from copy import deepcopy


def inference_size(width, height, max_side=1024):
    if min(width, height, max_side) <= 0:
        raise ValueError("Image dimensions and max_side must be positive.")
    scale = max_side / max(width, height)
    return max(8, round(width * scale) // 8 * 8), max(8, round(height * scale) // 8 * 8)


def resize_annotation(annotation, source_size, target_size):
    """Scale polygons in the document-orientation-corrected coordinate frame.

    Sizes describe the image BEFORE the document rotation. New caches record
    that size explicitly; legacy PP-OCR JSON is assumed to refer to the input
    image passed by the user.
    """
    result = deepcopy(annotation)
    source_size = result.get("tiger_image_size", source_size)
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("Annotation image dimensions must be positive.")
    angle = result.get("doc_preprocessor_res", {}).get("angle", 0)
    if angle not in (-1, 0, 90, 180, 270):
        raise ValueError(f"Unsupported document orientation: {angle}")
    if angle in (90, 270):
        sw, sh, tw, th = sh, sw, th, tw
    polys, texts = result.get("dt_polys", []), result.get("rec_texts", [])
    result["rec_texts"] = list(texts)
    if len(polys) != len(texts):
        raise ValueError("dt_polys and rec_texts must have matching lengths.")
    result["dt_polys"] = []
    for polygon in polys:
        if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
            raise ValueError("Each dt_polys entry must contain four [x, y] points.")
        result["dt_polys"].append([[float(x) * tw / sw, float(y) * th / sh] for x, y in polygon])
    result["doc_preprocessor_res"] = {"angle": angle}
    result["rec_scores"] = list(result.get("rec_scores", []))
    result["textline_orientation_angles"] = list(result.get("textline_orientation_angles", []))
    for key, default in (("rec_scores", 1.0), ("textline_orientation_angles", 0)):
        result[key] += [default] * max(0, len(texts) - len(result[key]))
    result["tiger_image_size"] = list(target_size)
    return result
