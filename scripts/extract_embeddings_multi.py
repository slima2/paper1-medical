#!/usr/bin/env python3
"""Re-extract the embedding channel under encoders that never saw the vocabulary.

The audit established that the published embedding channel is CTransPath
fine-tuned end-to-end on the six ANORAK growth patterns (191/191 tensors
overwritten from best_fuzzyarcloss_v2.pth). That makes E a representation
optimised to render P decodable, so reading PI ~ B2 as redundancy is circular
and a null result for unique-to-P is close to guaranteed by the data processing
inequality rather than discovered.

This script produces the two arms that break the circle, reusing the tiling of
extract_embeddings.py verbatim so the three cohorts are comparable row by row:

    ctranspath_raw   768-d   published checkpoint, no projection, RGB only
                             -- literally what the Methods section claims
    uni2h           1536-d   ViT-H/14 self-supervised, never saw ANORAK
                             -- the independent encoder the argument needs
    ctranspath_anorak 512-d  the existing arm, reproduced for provenance

WHY THE TILE ORDER IS SAFE
--------------------------
The benchmark pairs the two channels by row index with no key, so a different
tile enumeration would silently misalign every tile's vocabulary vector with
another tile's embedding. That risk is avoided rather than checked here,
because the enumeration is reproduced exactly:

    coords = [(x, y) for y in range(0, h, 224) for x in range(0, w, 224)]

with shuffle=False, and the background test is a pure function of the RGB tile

    (arr.mean(axis=2) > 0.9).mean() > 0.9

which is independent of the encoder. The kept subsequence is therefore
identical across all three arms by construction. The count is still asserted
against pattern_probs.npy, and unlike the original this script REFUSES to write
a mismatched slide: the benchmark does not skip such slides as the original
comment assumes, it truncates to the shorter array and trains on misaligned
channels.

Usage
-----
    python scripts/extract_embeddings_multi.py \
        --encoder uni2h \
        --svs_dir "/home/rapids/notebooks/slima/TGCA LUAD LUSC/TCGA LUAD LUSC" \
        --reference_dir /home/rapids/notebooks/slima/data \
        --out_dir /home/rapids/notebooks/slima/data_uni2h \
        --uni2h_dir /home/rapids/notebooks/slima/models/uni2-h \
        --num_gpus 6
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError:
    print("[ERROR] pip install timm")
    sys.exit(1)
try:
    import openslide
except ImportError:
    print("[ERROR] pip install openslide-python")
    sys.exit(1)

# ── identical to extract_embeddings.py; do not change ────────────────────────
TILE_SIZE = 224
BATCH_SIZE = 256
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
USE_FP16 = True
BACKGROUND_THRESHOLD = 0.9

MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ENCODERS = {
    "ctranspath_raw": 768,
    "ctranspath_anorak": 512,
    "uni2h": 1536,
}


# ══════════════════════════════════════════════════════════════════════════════
# CTransPath, verbatim from extract_embeddings.py
# ══════════════════════════════════════════════════════════════════════════════

class ConvStem(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        d1, d2 = embed_dim // 8, embed_dim // 4
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, d1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d1), nn.GELU(),
            nn.Conv2d(d1, d2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d2), nn.GELU(),
            nn.Conv2d(d2, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class CTransPathBackbone(nn.Module):
    def __init__(self, checkpoint_path=None, in_channels=3):
        super().__init__()
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, num_classes=0
        )
        self.model.patch_embed = ConvStem(in_chans=3, embed_dim=96)
        self.out_features = self.model.num_features  # 768

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_ckpt(checkpoint_path)

        if in_channels != 3:
            old = self.model.patch_embed.proj[0]
            new_conv = nn.Conv2d(in_channels, old.out_channels,
                                 old.kernel_size, old.stride, old.padding,
                                 bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight
                new_conv.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
            self.model.patch_embed.proj[0] = new_conv

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = (ckpt.get("model", ckpt.get("state_dict", ckpt))
              if isinstance(ckpt, dict) else ckpt)
        remap = {}
        for k, v in sd.items():
            if any(s in k for s in ("head.", "fc.", "attn_mask",
                                    "relative_position_index")):
                continue
            nk = k
            for i in range(3):
                if f"layers.{i}.downsample." in k:
                    nk = k.replace(f"layers.{i}.downsample.",
                                   f"layers.{i+1}.downsample.")
                    break
            remap[nk] = v
        ms = self.model.state_dict()
        filt = {k: v for k, v in remap.items()
                if k in ms and v.shape == ms[k].shape}
        if filt:
            self.model.load_state_dict(filt, strict=False)
        print(f"    CTransPath: loaded {len(filt)}/{len(ms)} tensors "
              f"from {Path(path).name}")

    def forward(self, x):
        return self.model(x)


def build_uni2h(model_dir: Path):
    """UNI2-h, with the timm configuration used elsewhere in this project."""
    from timm.layers import SwiGLUPacked

    kwargs = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
        num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
    )
    ckpt = model_dir / "pytorch_model.bin"
    if ckpt.exists():
        model = timm.create_model("vit_giant_patch14_224",
                                  pretrained=False, **kwargs)
        state = torch.load(str(ckpt), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"    UNI2-h: loaded from {ckpt.name}, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            raise RuntimeError(
                f"UNI2-h load is incomplete ({len(missing)} missing tensors). "
                "Refusing to extract with a partially initialised encoder."
            )
    else:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h",
                                  pretrained=True, **kwargs)
        print("    UNI2-h: loaded from HuggingFace")
    return model


def build_encoder(name: str, args, device: torch.device):
    """Returns (model, in_channels, embed_dim)."""
    if name == "uni2h":
        model = build_uni2h(Path(args.uni2h_dir))
        in_ch, embed = 3, 1536

    elif name == "ctranspath_raw":
        # The published checkpoint, three channels, no projection. This is the
        # encoder the manuscript describes.
        model = CTransPathBackbone(checkpoint_path=args.ctranspath_ckpt,
                                   in_channels=3)
        in_ch, embed = 3, 768

    elif name == "ctranspath_anorak":
        # Reproduces the published arm: backbone AND projection overwritten by
        # the growth-pattern checkpoint.
        ckpt = torch.load(args.model_ckpt, map_location="cpu",
                          weights_only=False)
        cfg = ckpt.get("config", {})
        embed = int(cfg.get("EMBED_DIM", 512))
        in_ch = 4 if cfg.get("USE_MASK_AS_CHANNEL", True) else 3
        backbone = CTransPathBackbone(checkpoint_path=args.ctranspath_ckpt,
                                      in_channels=in_ch)
        proj = nn.Sequential(nn.Linear(backbone.out_features, embed),
                             nn.LayerNorm(embed), nn.GELU(), nn.Dropout(0.0))
        model = nn.Sequential(backbone, proj)
        raw = ckpt.get("model", ckpt.get("model_state", {}))
        model.load_state_dict({k.replace("module.", ""): v
                               for k, v in raw.items()}, strict=False)
    else:
        raise ValueError(name)

    model.to(device).eval()
    if USE_FP16:
        model.half()
    return model, in_ch, embed


# ══════════════════════════════════════════════════════════════════════════════
# Tile dataset, verbatim apart from in_channels being a parameter
# ══════════════════════════════════════════════════════════════════════════════

class SlideTileDataset(Dataset):
    def __init__(self, slide_path, coords, tile_size=224, bg_thr=0.9,
                 in_channels=3):
        self.slide_path = slide_path
        self.coords = coords
        self.tile_size = tile_size
        self.bg_thr = bg_thr
        self.in_channels = in_channels
        self._slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        x, y = self.coords[idx]
        tile = self._slide.read_region(
            (x, y), 0, (self.tile_size, self.tile_size)
        ).convert("RGB")
        arr = np.asarray(tile, dtype=np.float32) / 255.0

        # The discard rule. Independent of the encoder, so the kept
        # subsequence is identical across arms.
        if (arr.mean(axis=2) > 0.9).mean() > self.bg_thr:
            return torch.empty(0), x, y, False

        chw = ((arr - MEANS) / STDS).transpose(2, 0, 1)
        if self.in_channels == 4:
            gray = np.asarray(tile.convert("L"), dtype=np.uint8)
            mask = (gray < gray.mean()).astype(np.float32)
            if mask.mean() < 0.2:
                mask[:] = 1.0
            chw = np.concatenate([chw, mask[None]], axis=0)

        return torch.from_numpy(chw.copy()), x, y, True


def _collate(batch):
    tensors, xs, ys, valids = zip(*batch)
    idx = [i for i, v in enumerate(valids) if v]
    if not idx:
        return torch.empty(0), [], [], 0
    return (torch.stack([tensors[i] for i in idx]),
            [xs[i] for i in idx], [ys[i] for i in idx], len(idx))


@torch.no_grad()
def extract_slide(svs_path: Path, model, device, in_ch: int, embed: int,
                  reference_dir: Path, out_dir: Path) -> str:
    slide_id = svs_path.stem
    ref_dir = reference_dir / "slides" / slide_id
    out_slide = out_dir / "slides" / slide_id
    emb_out = out_slide / "embeddings.npy"

    if emb_out.exists():
        return f"[SKIP] {slide_id}"
    probs_path = ref_dir / "pattern_probs.npy"
    if not probs_path.exists():
        return f"[SKIP-NO-PROBS] {slide_id}"
    n_probs = int(np.load(probs_path, mmap_mode="r").shape[0])

    try:
        sl = openslide.OpenSlide(str(svs_path))
        w, h = sl.level_dimensions[0]
        sl.close()
    except Exception as exc:
        return f"[ERR] {slide_id}: {exc}"

    coords = [(x, y)
              for y in range(0, h, TILE_SIZE)
              for x in range(0, w, TILE_SIZE)]
    ds = SlideTileDataset(str(svs_path), coords, TILE_SIZE,
                          BACKGROUND_THRESHOLD, in_ch)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        prefetch_factor=PREFETCH_FACTOR,
                        persistent_workers=True, collate_fn=_collate,
                        drop_last=False)

    chunks = []
    t0 = time.time()
    for imgs, _vx, _vy, cnt in loader:
        if cnt == 0:
            continue
        if USE_FP16:
            imgs = imgs.half()
        imgs = imgs.to(device, non_blocking=True)
        chunks.append(model(imgs).float().cpu().numpy())
    elapsed = time.time() - t0

    if not chunks:
        return f"[EMPTY] {slide_id}"
    emb = np.vstack(chunks).astype(np.float32)

    # Refuse rather than warn. The benchmark truncates to the shorter array
    # and trains on misaligned channels; it does not skip.
    if emb.shape[0] != n_probs:
        return (f"[REFUSED] {slide_id}: {emb.shape[0]} tiles vs "
                f"{n_probs} pattern rows. Not written.")
    if emb.shape[1] != embed:
        return (f"[REFUSED] {slide_id}: embed_dim {emb.shape[1]} != {embed}")

    out_slide.mkdir(parents=True, exist_ok=True)
    tmp = emb_out.with_suffix(".npy.tmp")
    np.save(tmp, emb)
    os.replace(tmp, emb_out)
    return f"[OK] {slide_id}: {emb.shape[0]} x {emb.shape[1]}, {elapsed:.0f}s"


def gpu_worker(gpu_id: int, svs_files: list, args):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    print(f"\n[GPU {gpu_id}] building {args.encoder} ...")
    model, in_ch, embed = build_encoder(args.encoder, args, device)
    print(f"[GPU {gpu_id}] in_channels={in_ch} embed_dim={embed} "
          f"slides={len(svs_files)}\n")
    for i, svs in enumerate(svs_files, 1):
        msg = extract_slide(svs, model, device, in_ch, embed,
                            Path(args.reference_dir), Path(args.out_dir))
        print(f"  [GPU {gpu_id}] ({i}/{len(svs_files)}) {msg}")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODERS))
    ap.add_argument("--svs_dir", required=True)
    ap.add_argument("--reference_dir", required=True,
                    help="existing data/ dir supplying pattern_probs.npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ctranspath_ckpt",
                    default="/home/rapids/notebooks/slima/models/ctranspath.pth")
    ap.add_argument("--model_ckpt",
                    default="/home/rapids/notebooks/slima/outputs/"
                            "ablation_study_v16_optuna/best_fuzzyarcloss_v2.pth")
    ap.add_argument("--uni2h_dir",
                    default="/home/rapids/notebooks/slima/models/uni2-h")
    ap.add_argument("--num_gpus", type=int, default=None)
    ap.add_argument("--max_slides", type=int, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print(f"  embedding extraction, encoder = {args.encoder} "
          f"({ENCODERS[args.encoder]}-d)")
    print("=" * 70)

    ref_slides = Path(args.reference_dir) / "slides"
    out_dir = Path(args.out_dir)
    (out_dir / "slides").mkdir(parents=True, exist_ok=True)

    all_svs = sorted(Path(args.svs_dir).rglob("*.svs"))
    lookup = {p.stem: p for p in all_svs}

    to_process, missing = [], 0
    for sd in sorted(ref_slides.iterdir()):
        if not (sd / "pattern_probs.npy").exists():
            continue
        if (out_dir / "slides" / sd.name / "embeddings.npy").exists():
            continue
        if sd.name in lookup:
            to_process.append(lookup[sd.name])
        else:
            missing += 1
    if args.max_slides:
        to_process = to_process[:args.max_slides]

    print(f"\n{len(all_svs)} SVS found, {len(to_process)} to extract, "
          f"{missing} without an SVS")
    if not to_process:
        print("nothing to do")
        return

    ng = args.num_gpus or (torch.cuda.device_count()
                           if torch.cuda.is_available() else 1)
    ng = min(ng, max(1, torch.cuda.device_count()))
    print(f"{ng} GPU(s), fp16={USE_FP16}\n")

    t0 = time.time()
    if ng == 1:
        gpu_worker(0, to_process, args)
    else:
        assign = [[] for _ in range(ng)]
        for i, sp in enumerate(to_process):
            assign[i % ng].append(sp)
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=gpu_worker, args=(g, assign[g], args))
                 for g in range(ng) if assign[g]]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    done = sum(1 for sd in (out_dir / "slides").iterdir()
               if (sd / "embeddings.npy").exists())
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. {done} slides written.")
    print("\nLink the unchanged vocabulary channel and labels into place:")
    print(f"  cp {args.reference_dir}/labels.csv {out_dir}/labels.csv")
    print(f"  for d in {args.reference_dir}/slides/*/; do s=$(basename $d); "
          f"[ -d {out_dir}/slides/$s ] && ln -sf $d/pattern_probs.npy "
          f"{out_dir}/slides/$s/ && ln -sf $d/pattern_labels.npy "
          f"{out_dir}/slides/$s/; done")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
