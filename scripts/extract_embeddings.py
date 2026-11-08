#!/usr/bin/env python3
"""
extract_embeddings.py
======================
Extracts CTransPath 768-d embeddings for every slide that already has
pattern_probs.npy (from prepare_benchmark_inputs.py) but is missing
embeddings.npy (needed for ABMIL baselines and proposed method).

Re-uses the EXACT same CTransPath backbone + ConvStem from your
inference script, so embeddings are consistent with your classifier.

USAGE
─────
  python extract_embeddings.py \\
      --svs_dir "/home/rapids/notebooks/slima/TGCA LUAD LUSC/TCGA LUAD LUSC" \\
      --data_dir data \\
      --ctranspath_ckpt /home/rapids/notebooks/slima/models/ctranspath.pth \\
      --model_ckpt /home/rapids/notebooks/slima/outputs/ablation_study_v16_optuna/best_fuzzyarcloss_v2.pth

SPEED
─────
  6× H100: set --num_gpus 6  (each GPU handles a different slide)
  Single GPU: default behaviour

OUTPUT
──────
  data/slides/<slide_id>/embeddings.npy   shape: (N_tiles, 768)  float32
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"]      = "2"
os.environ["OMP_NUM_THREADS"]      = "2"

import sys, time, argparse
import numpy as np
from pathlib import Path
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import timm
except ImportError:
    print("[ERROR] pip install timm"); sys.exit(1)
try:
    import openslide
except ImportError:
    print("[ERROR] pip install openslide-python"); sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
TILE_SIZE           = 224
BATCH_SIZE          = 512
NUM_WORKERS         = 4
PREFETCH_FACTOR     = 4
USE_FP16            = True
BACKGROUND_THRESHOLD = 0.9

MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDS  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CTransPath backbone — IDENTICAL to inference script
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
            'swin_tiny_patch4_window7_224', pretrained=False, num_classes=0
        )
        self.model.patch_embed = ConvStem(in_chans=3, embed_dim=96)
        self.out_features = self.model.num_features          # 768

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_ckpt(checkpoint_path)

        if in_channels != 3:
            old = self.model.patch_embed.proj[0]
            new_conv = nn.Conv2d(
                in_channels, old.out_channels,
                old.kernel_size, old.stride, old.padding, bias=False
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old.weight
                new_conv.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
            self.model.patch_embed.proj[0] = new_conv

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
        remap = {}
        for k, v in sd.items():
            if any(s in k for s in ('head.', 'fc.', 'attn_mask', 'relative_position_index')):
                continue
            nk = k
            for i in range(3):
                if f'layers.{i}.downsample.' in k:
                    nk = k.replace(f'layers.{i}.downsample.', f'layers.{i+1}.downsample.')
                    break
            remap[nk] = v
        ms = self.model.state_dict()
        filt = {k: v for k, v in remap.items() if k in ms and v.shape == ms[k].shape}
        if filt:
            self.model.load_state_dict(filt, strict=False)

    def forward(self, x):
        return self.model(x)


# ══════════════════════════════════════════════════════════════════════════════
# Tile dataset — same as inference script
# ══════════════════════════════════════════════════════════════════════════════

class SlideTileDataset(Dataset):
    def __init__(self, slide_path, coords, tile_size=224, bg_thr=0.9, in_channels=4):
        self.slide_path = slide_path
        self.coords     = coords
        self.tile_size  = tile_size
        self.bg_thr     = bg_thr
        self.in_channels = in_channels
        self._slide     = None

    def __len__(self): return len(self.coords)

    def __getitem__(self, idx):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
        x, y = self.coords[idx]
        tile = self._slide.read_region(
            (x, y), 0, (self.tile_size, self.tile_size)
        ).convert("RGB")
        arr = np.asarray(tile, dtype=np.float32) / 255.0

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
    return (
        torch.stack([tensors[i] for i in idx]),
        [xs[i] for i in idx],
        [ys[i] for i in idx],
        len(idx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Build model (backbone + projection head only — no classification head)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_extractor(model_ckpt: str, ctranspath_ckpt: str, device: torch.device):
    """
    Loads CTransPath + projection head from your model checkpoint.
    Returns (model, in_channels, embed_dim).

    The output of this model is the 512-d (or whatever embed_dim is)
    projected embedding — same space your FuzzyArcLoss head operates in.

    We use the projected embedding (not raw 768-d) so Baseline 2 uses
    the same representation space as Proposed. Pass --use_raw_768 to use
    the raw CTransPath output instead.
    """
    ckpt = torch.load(model_ckpt, map_location='cpu', weights_only=False)
    cfg  = ckpt.get('config', {})

    embed   = int(cfg.get('EMBED_DIM', 512))
    in_ch   = 4 if cfg.get('USE_MASK_AS_CHANNEL', True) else 3
    img_sz  = int(cfg.get('IMG_SIZE', 224))

    print(f"  embed_dim={embed}, in_channels={in_ch}, img_size={img_sz}")

    backbone = CTransPathBackbone(checkpoint_path=ctranspath_ckpt, in_channels=in_ch)
    proj = nn.Sequential(
        nn.Linear(backbone.out_features, embed),
        nn.LayerNorm(embed), nn.GELU(), nn.Dropout(0.0),
    )
    model = nn.Sequential(backbone, proj)

    raw = ckpt.get('model', ckpt.get('model_state', {}))
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in raw.items()},
        strict=False
    )
    model.to(device).eval()
    if USE_FP16:
        model.half()
    return model, in_ch, embed


# ══════════════════════════════════════════════════════════════════════════════
# Process one slide → embeddings.npy
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_slide_embeddings(
    svs_path:   Path,
    model:      nn.Module,
    device:     torch.device,
    in_ch:      int,
    out_dir:    Path,
) -> str:
    """
    Extracts per-tile embeddings for one slide.
    ONLY processes tiles that were kept by the inference script
    (same background threshold → same tile count → arrays are aligned).
    Returns a status string.
    """
    slide_id  = svs_path.stem
    slide_dir = out_dir / "slides" / slide_id
    emb_out   = slide_dir / "embeddings.npy"

    if emb_out.exists():
        return f"[SKIP] {slide_id}"

    # only process slides that have pattern_probs (means inference was done)
    if not (slide_dir / "pattern_probs.npy").exists():
        return f"[SKIP-NO-PROBS] {slide_id}"

    try:
        sl = openslide.OpenSlide(str(svs_path))
        w, h = sl.level_dimensions[0]
        sl.close()
    except Exception as e:
        return f"[ERR] {slide_id}: {e}"

    coords = [(x, y) for y in range(0, h, TILE_SIZE) for x in range(0, w, TILE_SIZE)]
    ds = SlideTileDataset(str(svs_path), coords, TILE_SIZE, BACKGROUND_THRESHOLD, in_ch)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
        collate_fn=_collate, drop_last=False,
    )

    all_embs = []
    t0 = time.time()

    for imgs, vx, vy, cnt in loader:
        if cnt == 0:
            continue
        if USE_FP16:
            imgs = imgs.half()
        imgs = imgs.to(device, non_blocking=True)
        emb  = model(imgs)                           # (B, embed_dim)
        all_embs.append(emb.float().cpu().numpy())

    elapsed = time.time() - t0

    if not all_embs:
        return f"[EMPTY] {slide_id}"

    emb_arr = np.vstack(all_embs).astype(np.float32)   # (N_tiles, embed_dim)

    # Sanity check: tile count must match pattern_probs
    probs = np.load(slide_dir / "pattern_probs.npy")
    if emb_arr.shape[0] != probs.shape[0]:
        return (
            f"[MISMATCH] {slide_id}: "
            f"embeddings={emb_arr.shape[0]} vs probs={probs.shape[0]} tiles. "
            f"This means background filtering differed between runs. "
            f"Saving anyway — benchmark will detect and skip this slide."
        )

    slide_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, emb_arr)

    return (
        f"[OK] {slide_id}: {emb_arr.shape[0]} tiles × {emb_arr.shape[1]}-d, "
        f"{elapsed:.0f}s"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GPU worker (one per GPU, mirrors inference script pattern)
# ══════════════════════════════════════════════════════════════════════════════

def gpu_worker(
    gpu_id:         int,
    svs_files:      list,
    data_dir:       str,
    model_ckpt:     str,
    ctranspath_ckpt: str,
):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    print(f"\n[GPU {gpu_id}] Loading feature extractor …")

    model, in_ch, embed_dim = build_feature_extractor(model_ckpt, ctranspath_ckpt, device)
    out_dir = Path(data_dir)

    print(f"[GPU {gpu_id}] embed_dim={embed_dim}, {len(svs_files)} slides to process\n")

    for i, svs_path in enumerate(svs_files):
        msg = extract_slide_embeddings(svs_path, model, device, in_ch, out_dir)
        print(f"  [GPU {gpu_id}] ({i+1}/{len(svs_files)}) {msg}")
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Extract CTransPath embeddings for ABMIL benchmark"
    )
    ap.add_argument(
        "--svs_dir", required=True,
        default="/home/rapids/notebooks/slima/TGCA LUAD LUSC/TCGA LUAD LUSC",
        help="Root directory containing .svs files (searched recursively)",
    )
    ap.add_argument(
        "--data_dir", default="data",
        help="data/ directory created by prepare_benchmark_inputs.py",
    )
    ap.add_argument(
        "--model_ckpt", required=True,
        default="/home/rapids/notebooks/slima/outputs/ablation_study_v16_optuna/best_fuzzyarcloss_v2.pth",
        help="Path to best_fuzzyarcloss_v2.pth (needed to load projection head weights)",
    )
    ap.add_argument(
        "--ctranspath_ckpt", required=True,
        default="/home/rapids/notebooks/slima/models/ctranspath.pth",
        help="Path to ctranspath.pth backbone weights",
    )
    ap.add_argument(
        "--num_gpus", type=int, default=None,
        help="Number of GPUs to use (default: all available)",
    )
    ap.add_argument(
        "--max_slides", type=int, default=None,
        help="Limit number of slides (for testing)",
    )
    args = ap.parse_args()

    print("=" * 65)
    print("  CTransPath Embedding Extractor")
    print("=" * 65)

    # find SVS files that have pattern_probs but no embeddings yet
    slides_dir = Path(args.data_dir) / "slides"
    all_svs    = sorted(Path(args.svs_dir).rglob("*.svs"))
    if args.max_slides:
        all_svs = all_svs[:args.max_slides]

    # filter to slides already in data/slides/ (inference was done)
    svs_lookup = {p.stem: p for p in all_svs}
    to_process = []
    for slide_dir in sorted(slides_dir.iterdir()):
        sid = slide_dir.name
        if (slide_dir / "pattern_probs.npy").exists() and \
           not (slide_dir / "embeddings.npy").exists():
            if sid in svs_lookup:
                to_process.append(svs_lookup[sid])
            else:
                print(f"  [WARN] SVS not found for {sid} — cannot extract embeddings")

    print(f"\n{len(all_svs)} total SVS files")
    print(f"{len(to_process)} slides need embeddings extracted")

    if not to_process:
        print("Nothing to do — all slides already have embeddings.npy")
        return

    ng = args.num_gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    ng = min(ng, max(1, torch.cuda.device_count()))
    gname = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n{ng} × {gname}, fp16={USE_FP16}")

    t0 = time.time()

    if ng == 1:
        gpu_worker(
            0, to_process, args.data_dir,
            args.model_ckpt, args.ctranspath_ckpt
        )
    else:
        assign = [[] for _ in range(ng)]
        for i, sp in enumerate(to_process):
            assign[i % ng].append(sp)
        print(f"Distribution: {[len(a) for a in assign]} slides per GPU")
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=gpu_worker,
                args=(g, assign[g], args.data_dir, args.model_ckpt, args.ctranspath_ckpt)
            )
            for g in range(ng) if assign[g]
        ]
        for p in procs: p.start()
        for p in procs: p.join()

    elapsed = time.time() - t0
    done = sum(
        1 for sd in slides_dir.iterdir()
        if (sd / "embeddings.npy").exists()
    )
    print(f"\nDONE in {elapsed/60:.1f} min.  {done} slides now have embeddings.npy")
    print(f"\nNext: python pattern_informed_abmil_benchmark.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
