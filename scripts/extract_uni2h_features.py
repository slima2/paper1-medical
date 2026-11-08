#!/usr/bin/env python3
"""Re-extract per-tile embeddings with UNI2-h, preserving tile order exactly.

The assignment of a gene to a regime is a property of the triple (vocabulary,
encoder, cohort), so the natural challenge to every out-of-ontology read is
that a stronger encoder would move it. This script produces the embeddings
needed to answer that, by swapping CTransPath for UNI2-h and leaving everything
else identical.

    <out>/slides/<slide_id>/embeddings.npy      (T, 1536) float32

ORDERING IS THE WHOLE RISK
--------------------------
The benchmark aligns the two channels by row index:

    n = min(embeddings.shape[0], pattern_probs.shape[0])
    embeddings, pattern_probs = embeddings[:n], pattern_probs[:n]

There is no key, no coordinate, no check. If tiles are enumerated in a
different order than when pattern_probs.npy was written, every tile's
vocabulary vector is paired with a different tile's embedding, the concat and
Choquet conditions are silently corrupted, and the run still completes and
reports plausible numbers. This script therefore refuses to write a slide whose
tile count does not match its existing pattern_probs.npy, and enumerates tiles
in sorted filename order, which is what the original extraction almost
certainly used. Run --discover first and confirm the counts line up before
committing GPU time.

Usage
-----
    # 1. inspect the layout and check counts against pattern_probs.npy
    python scripts/extract_uni2h_features.py --discover \
        --tiles-root /path/to/tiles --reference /home/rapids/notebooks/slima/data

    # 2. extract
    python scripts/extract_uni2h_features.py \
        --tiles-root /path/to/tiles \
        --reference /home/rapids/notebooks/slima/data \
        --out /home/rapids/notebooks/slima/data_uni \
        --model-dir /home/rapids/notebooks/slima/models/uni2-h \
        --batch-size 256 --fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
EMBED_DIM = 1536

# Exactly the configuration used for UNI2-h elsewhere in this project.
TIMM_KWARGS = dict(
    img_size=224,
    patch_size=14,
    depth=24,
    num_heads=24,
    init_values=1e-5,
    embed_dim=1536,
    mlp_ratio=2.66667 * 2,
    num_classes=0,
    no_embed_class=True,
    reg_tokens=8,
    dynamic_img_size=True,
)


def find_tiles(tiles_root: Path, slide_id: str) -> list[Path]:
    """Tiles for one slide, in sorted filename order.

    Sorted order is the contract with the existing pattern_probs.npy. Do not
    change it without re-extracting the vocabulary channel too.
    """
    slide_dir = tiles_root / slide_id
    if not slide_dir.is_dir():
        return []
    return sorted(
        (p for p in slide_dir.rglob("*") if p.suffix.lower() in TILE_SUFFIXES),
        key=lambda p: str(p.relative_to(slide_dir)),
    )


def discover(args: argparse.Namespace) -> int:
    ref_slides = sorted((args.reference / "slides").iterdir())
    print(f"reference cohort: {len(ref_slides)} slides under {args.reference}")
    if not ref_slides:
        print("  nothing found; is --reference the directory containing slides/ ?")
        return 1

    sample = ref_slides[0]
    print(f"  example slide dir: {sample.name}")
    for f in sorted(sample.iterdir()):
        print(f"    {f.name}")

    print(f"\ntile root: {args.tiles_root}")
    if not args.tiles_root.is_dir():
        print("  MISSING. Tiles are required; see the module docstring.")
        return 1
    children = sorted(p for p in args.tiles_root.iterdir() if p.is_dir())
    print(f"  {len(children)} subdirectories, first few: "
          f"{[p.name for p in children[:3]]}")

    print("\ncount check against pattern_probs.npy (first 10 slides)")
    mismatched = matched = missing = 0
    for slide_dir in ref_slides:
        probs_path = slide_dir / "pattern_probs.npy"
        if not probs_path.exists():
            continue
        n_probs = int(np.load(probs_path, mmap_mode="r").shape[0])
        tiles = find_tiles(args.tiles_root, slide_dir.name)
        if not tiles:
            missing += 1
            status = "NO TILES"
        elif len(tiles) == n_probs:
            matched += 1
            status = "ok"
        else:
            mismatched += 1
            status = "MISMATCH"
        if matched + mismatched + missing <= 10:
            print(f"  {slide_dir.name:<28} probs={n_probs:<6} tiles={len(tiles):<6} {status}")

    print(f"\n  matched {matched}   mismatched {mismatched}   no tiles {missing}")
    if mismatched or missing:
        print("\n  Do not extract until this is resolved. A mismatch means the tile")
        print("  enumeration here differs from the one that produced the vocabulary")
        print("  channel, and the two channels would be misaligned row by row.")
        return 1
    print("\n  Counts agree. Ordering is still an assumption: sorted filename order")
    print("  is used here and is almost certainly what the original extraction did,")
    print("  but if the original used a coordinate list or a shuffled manifest, say")
    print("  so and this script must read that manifest instead.")
    return 0


def build_model(args: argparse.Namespace):
    import timm
    import torch
    from timm.layers import SwiGLUPacked

    kwargs = dict(TIMM_KWARGS, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    ckpt = args.model_dir / "pytorch_model.bin"
    if ckpt.exists():
        model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kwargs)
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"loaded UNI2-h from {ckpt}")
    else:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs)
        print("loaded UNI2-h from HuggingFace")

    device = torch.device(args.device)
    model.eval().to(device)
    if args.fp16:
        model = model.half()

    from timm.data import create_transform, resolve_data_config
    transform = create_transform(**resolve_data_config({}, model=model))
    print(f"transform: {transform}")
    return model, transform, device


def extract(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    model, transform, device = build_model(args)
    ref_slides = sorted((args.reference / "slides").iterdir())
    (args.out / "slides").mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for i, slide_dir in enumerate(ref_slides, 1):
        slide_id = slide_dir.name
        out_path = args.out / "slides" / slide_id / "embeddings.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        probs_path = slide_dir / "pattern_probs.npy"
        if not probs_path.exists():
            print(f"[{i}/{len(ref_slides)}] {slide_id}: no pattern_probs.npy, skip")
            failed += 1
            continue
        n_probs = int(np.load(probs_path, mmap_mode="r").shape[0])

        tiles = find_tiles(args.tiles_root, slide_id)
        if len(tiles) != n_probs:
            print(f"[{i}/{len(ref_slides)}] {slide_id}: REFUSING, "
                  f"{len(tiles)} tiles vs {n_probs} pattern rows")
            failed += 1
            continue

        feats = np.empty((n_probs, EMBED_DIM), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(tiles), args.batch_size):
                batch = tiles[start:start + args.batch_size]
                imgs = torch.stack([
                    transform(Image.open(p).convert("RGB")) for p in batch
                ]).to(device)
                if args.fp16:
                    imgs = imgs.half()
                out = model(imgs).float().cpu().numpy()
                feats[start:start + len(batch)] = out

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".npy.tmp")
        np.save(tmp, feats)
        tmp.replace(out_path)
        done += 1
        print(f"[{i}/{len(ref_slides)}] {slide_id}: {feats.shape} written")

    print(f"\nextracted {done}, skipped {skipped}, failed {failed}")

    # The vocabulary channel is unchanged, so link rather than copy it.
    print("\nNext: the new cohort needs the vocabulary channel too. Either")
    print(f"  cp {args.reference}/labels.csv {args.out}/labels.csv")
    print(f"  for d in {args.reference}/slides/*/; do s=$(basename $d); "
          f"ln -sf $d/pattern_probs.npy {args.out}/slides/$s/; "
          f"ln -sf $d/pattern_labels.npy {args.out}/slides/$s/; done")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True,
                   help="existing cohort dir containing slides/ and labels.csv")
    p.add_argument("--tiles-root", type=Path, required=True,
                   help="directory with one subdirectory of tile images per slide")
    p.add_argument("--out", type=Path, help="output cohort dir (not for --discover)")
    p.add_argument("--model-dir", type=Path,
                   default=Path("/home/rapids/notebooks/slima/models/uni2-h"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--discover", action="store_true")
    args = p.parse_args()

    if args.discover:
        return discover(args)
    if not args.out:
        p.error("--out is required unless --discover is given")
    return extract(args)


if __name__ == "__main__":
    sys.exit(main())
