# UZ-ST alignment utilities

These are the alignment and color-correction utilities used to prepare UZ-ST multi-lens data. The original scripts are retained with their source names and their original path configuration blocks; edit those path settings before running.

| Script | Purpose |
| --- | --- |
| `easy.py` | SIFT/MAGSAC cascade alignment. |
| `matching4multiLens.py` | Multi-lens SIFT and LoFTR matching. |
| `sift_loftr.py` | Two-stage SIFT + LoFTR alignment. |
| `sift_ripe.py` | SIFT + RIPE alignment with wavelet color correction. |
| `color_fix.py`, `color_fix_same_dir.py`, `color_fix_ultralens.py` | AdaIN or wavelet color correction variants. |
| `utils/wavelet.py` | Shared AdaIN and wavelet color-fix implementation. |

Install the extra dependencies after the main project environment is ready:

```bash
pip install -r tools/uzst_alignment/requirements.txt
```

`sift_ripe.py` additionally requires RIPE. Clone it next to the scripts so that the original `RIPE.ripe` import resolves:

```bash
git clone https://github.com/fraunhoferhhi/RIPE.git tools/uzst_alignment/RIPE
```

LoFTR weights are downloaded by Kornia on first use. The alignment outputs and external RIPE source/weights are intentionally excluded from this repository.
