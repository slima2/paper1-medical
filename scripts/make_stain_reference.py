#!/usr/bin/env python
"""Pick a representative TCGA-LUAD H&E tile to use as the stain-normalization target.

The CPTAC cohort-transfer must map CPTAC tiles into the colour space the model was
trained on (raw TCGA-LUAD + ImageNet norm; the pipeline has no stain normalization).
So the Macenko/Reinhard *target* has to be a real, well-stained TCGA-LUAD tile -- not
a canonical/foreign reference. This helper produces one deterministically.

Two input modes:
  # (a) from saved training tiles (png/jpg/tif) -- preferred if you have them:
  python make_stain_reference.py --tiles-dir /path/to/tcga_luad_tiles \
      --out "$DATA_ROOT/luad_cptac/tcga_luad_reference.png"

  # (b) from one TCGA-LUAD WSI (svs/tif) -- grabs a tissue-rich 224px level-0 tile:
  python make_stain_reference.py --wsi /path/to/TCGA-XX-XXXX.svs \
      --out "$DATA_ROOT/luad_cptac/tcga_luad_reference.png"

Selection criterion (both modes): among tissue-rich, well-exposed candidates,
take the one whose colour is closest to the *median* candidate (a "typical"
well-stained tile), which is a robust, non-arbitrary Macenko/Reinhard target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 224
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def tissue_stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (tissue_fraction, mean_saturation, mean_value) for an RGB uint8 tile."""
    rgb = arr.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0.0)
    val = mx
    # tissue = not-background: reasonably saturated and not blown-out white
    tissue = (sat > 0.10) & (val < 0.92)
    return float(tissue.mean()), float(sat[tissue].mean() if tissue.any() else 0.0), float(val.mean())


def load_candidates_from_dir(tiles_dir: Path, sample: int) -> list[tuple[Path, np.ndarray]]:
    paths = [p for p in tiles_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not paths:
        sys.exit(f"no image tiles under {tiles_dir}")
    rng = np.random.default_rng(0)
    if len(paths) > sample:
        paths = list(rng.choice(np.array(paths, dtype=object), size=sample, replace=False))
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            if min(im.size) < TILE:
                continue
            # centre-crop to a square TILE for comparable colour statistics
            im = im.resize((TILE, TILE)) if im.size != (TILE, TILE) else im
            out.append((p, np.asarray(im)))
        except Exception:
            continue
    if not out:
        sys.exit("no readable tiles of sufficient size")
    return out


def load_candidates_from_wsi(wsi: Path, sample: int) -> list[tuple[Path, np.ndarray]]:
    import openslide  # requires openslide-python
    slide = openslide.OpenSlide(str(wsi))
    w, h = slide.level_dimensions[0]
    rng = np.random.default_rng(0)
    out = []
    tries = 0
    while len(out) < sample and tries < sample * 20:
        tries += 1
        x = int(rng.integers(0, max(1, w - TILE)))
        y = int(rng.integers(0, max(1, h - TILE)))
        tile = slide.read_region((x, y), 0, (TILE, TILE)).convert("RGB")
        arr = np.asarray(tile)
        frac, _, _ = tissue_stats(arr)
        if frac > 0.6:
            out.append((Path(f"{wsi.stem}@{x}_{y}"), arr))
    slide.close()
    if not out:
        sys.exit("could not sample tissue-rich tiles from the WSI")
    return out


def pick_reference(cands: list[tuple[Path, np.ndarray]]) -> tuple[Path, np.ndarray]:
    # Keep tissue-rich, well-exposed candidates, then take the one closest to the
    # median colour (per-channel mean over tissue pixels) -- the "typical" tile.
    scored = []
    for p, arr in cands:
        frac, sat, val = tissue_stats(arr)
        if frac < 0.5 or sat < 0.12 or not (0.35 < val < 0.9):
            continue
        scored.append((p, arr, arr.reshape(-1, 3).mean(0)))
    if not scored:  # relax if the gate was too strict
        scored = [(p, arr, arr.reshape(-1, 3).mean(0)) for p, arr in cands]
    means = np.stack([m for _, _, m in scored])
    median = np.median(means, axis=0)
    dists = np.linalg.norm(means - median, axis=1)
    j = int(dists.argmin())
    return scored[j][0], scored[j][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tiles-dir", default=None, help="dir of saved TCGA-LUAD tiles")
    src.add_argument("--wsi", default=None, help="one TCGA-LUAD WSI (svs/tif)")
    ap.add_argument("--out", required=True, help="output reference tile (png)")
    ap.add_argument("--sample", type=int, default=400, help="candidate tiles to consider")
    args = ap.parse_args()

    if args.tiles_dir:
        cands = load_candidates_from_dir(Path(args.tiles_dir), args.sample)
    else:
        cands = load_candidates_from_wsi(Path(args.wsi), args.sample)

    ref_id, ref_arr = pick_reference(cands)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ref_arr, "RGB").save(out)
    frac, sat, val = tissue_stats(ref_arr)
    print(f"reference tile: {ref_id}")
    print(f"  tissue_frac={frac:.2f} mean_sat={sat:.2f} mean_val={val:.2f}")
    print(f"wrote {out}")
    print("next: pass this to run_cptac_inference.py via --stain-target")


if __name__ == "__main__":
    main()
