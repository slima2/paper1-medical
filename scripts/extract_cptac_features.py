#!/usr/bin/env python
"""Dump the two channels (E, P) for every CPTAC-LUAD slide, for regime re-reading.

WHY THIS EXISTS
---------------
run_cptac_inference.py answers "do the frozen TCGA weights transfer?" and keeps
only `result.mutation_scores`. That is a test of the *predictor*. It cannot test
whether the *regime assignment* reproduces on a second cohort, because a regime
is read from the contrast between four conditions (B2/B3/PI/FC) that have to be
TRAINED on the cohort in question. Training them needs the raw channels, which
the inference run discards:

    encoder channel E : the 512-d CTransPath embedding of each retained tile
    source channel  P : the 6-d ANORAK growth-pattern probability of each tile

Both are already computed inside the production tile pass; we only persist them.
This script therefore runs the identical LCHAI pipeline as
run_cptac_inference.py -- same tiling, same backbones, same optional stain
normalisation -- and intercepts the tile-level output instead of the gene
scores. Nothing in the LCHAI repo is modified.

Output: one .npz per slide under $DATA_ROOT/luad_cptac/features/, plus a
manifest.parquet. Feed both to run_cptac_domainA_v2.py.

    slide_id.npz
        emb       float16 [T, 512]   CTransPath embedding per retained tile
        pat       float16 [T, 6]     ANORAK pattern probabilities per tile
        coords    int32   [T, 2]     tile (x, y) at level 0, for L1 read-out
        attn      float32 [T]        production attention, if the pipeline exposes it

DISCOVERING THE TILE-PASS CONTRACT
----------------------------------
The internal return type of `_real_tile_inference` is a private detail of the
LCHAI version you are running, so this script does not hard-code field names.
It locates the two channels by SHAPE (last axis 512 -> embeddings, last axis 6
-> patterns), which is stable across refactors. Run --inspect once on a single
slide to print exactly what the hook received before launching the cohort:

    python extract_cptac_features.py --inspect --limit 1 \
        --lchai-app-dir ... --models-dir ... --wsi-dir ...

If the shape probe fails, --inspect prints the structure and you can pin the
fields explicitly with --emb-key / --pat-key.

Usage (DGX, after `source env.sh`):
    python extract_cptac_features.py \
        --lchai-app-dir "$LCHAI/apps/inference-service" \
        --models-dir    "$LCHAI/models" \
        --wsi-dir       "$DATA_ROOT/luad_cptac/wsi" \
        --device cuda
    # -> $DATA_ROOT/luad_cptac/features/*.npz  +  features/manifest.parquet

To reproduce the stain-normalised arm, add:
    --stain-normalize --stain-method reinhard --stain-target tcga_luad_reference.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the slide enumeration, DICOM handling and stain normalisation that the
# transfer run already validated on this cohort, so the two runs see identical
# pixels and any difference is attributable to the read-out, not the input.
from run_cptac_inference import (  # noqa: E402
    GENES,
    build_stain_normalizer,
    convert_dicom_to_tiff,
    list_slides,
    patient_of,
    resolve_root,
)

EMB_DIM = 512   # CTransPath
PAT_DIM = 6     # ANORAK growth patterns (lepidic, acinar, papillary,
                # micropapillary, solid, cribriform)


# --------------------------------------------------------------------------
# Shape-based discovery of the two channels in whatever the tile pass returns
# --------------------------------------------------------------------------

def _walk(obj, depth: int = 0, max_depth: int = 3):
    """Yield (path, value) for arrays nested in dicts/lists/objects."""
    if depth > max_depth:
        return
    if isinstance(obj, np.ndarray):
        yield ("", obj)
        return
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            yield ("", obj.detach().cpu().numpy())
            return
    except Exception:
        pass
    if isinstance(obj, dict):
        for k, v in obj.items():
            for p, a in _walk(v, depth + 1, max_depth):
                yield (f"{k}.{p}" if p else str(k), a)
    elif isinstance(obj, (list, tuple)):
        # A list of per-tile records is common; stack it if the elements are
        # themselves numeric vectors.
        if obj and all(isinstance(x, (int, float)) for x in obj):
            yield ("", np.asarray(obj))
            return
        for i, v in enumerate(obj[:16]):
            for p, a in _walk(v, depth + 1, max_depth):
                yield (f"[{i}].{p}" if p else f"[{i}]", a)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            for p, a in _walk(v, depth + 1, max_depth):
                yield (f"{k}.{p}" if p else str(k), a)


def _stack_per_tile(path_arrays, want_dim: int, n_tiles: int | None):
    """Find a [T, want_dim] array, or T arrays of [want_dim], among candidates."""
    # Case 1: already a 2-D block with the right trailing dimension.
    for path, arr in path_arrays:
        if arr.ndim == 2 and arr.shape[1] == want_dim:
            if n_tiles is None or arr.shape[0] == n_tiles:
                return arr, path
    # Case 2: per-tile vectors that need stacking, keyed by a common suffix.
    groups: dict[str, list[tuple[int, np.ndarray]]] = {}
    for path, arr in path_arrays:
        if arr.ndim == 1 and arr.shape[0] == want_dim:
            m = re.match(r"\[(\d+)\]\.?(.*)", path)
            if m:
                groups.setdefault(m.group(2), []).append((int(m.group(1)), arr))
    for suffix, items in groups.items():
        if len(items) > 1:
            items.sort(key=lambda t: t[0])
            return np.stack([a for _i, a in items]), f"[*].{suffix}"
    return None, None


def extract_channels(captured, inspect: bool = False):
    """Pull (emb [T,512], pat [T,6], coords [T,2] | None) out of the tile output."""
    cands = [(p, a) for p, a in _walk(captured) if a.size]
    if inspect:
        print("\n--- tile-pass output structure ---")
        print(f"top-level type: {type(captured)}")
        for p, a in cands[:60]:
            print(f"  {p or '<root>':50s} {str(a.dtype):10s} {a.shape}")
        print("--- end structure ---\n")

    emb, emb_path = _stack_per_tile(cands, EMB_DIM, None)
    n_tiles = emb.shape[0] if emb is not None else None
    pat, pat_path = _stack_per_tile(cands, PAT_DIM, n_tiles)
    coords, _ = _stack_per_tile(cands, 2, n_tiles)

    if inspect:
        print(f"resolved embeddings <- {emb_path}")
        print(f"resolved patterns   <- {pat_path}")
    if emb is None or pat is None:
        raise RuntimeError(
            "could not locate the two channels by shape "
            f"(embeddings {EMB_DIM}-d: {'ok' if emb is not None else 'MISSING'}, "
            f"patterns {PAT_DIM}-d: {'ok' if pat is not None else 'MISSING'}). "
            "Re-run with --inspect and pin the fields with --emb-key/--pat-key."
        )
    return emb, pat, coords


def pinned_channels(captured, emb_key: str, pat_key: str):
    """Explicit field lookup, used when shape discovery is ambiguous."""
    table = {p: a for p, a in _walk(captured)}
    missing = [k for k in (emb_key, pat_key) if k not in table]
    if missing:
        raise RuntimeError(f"--emb-key/--pat-key not found: {missing}. "
                           f"Available: {sorted(table)[:40]}")
    return table[emb_key], table[pat_key], None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lchai-app-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--wsi-dir", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default $DATA_ROOT/luad_cptac/features")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k-tiles", type=int, default=200,
                    help="must match the TCGA protocol (200)")
    ap.add_argument("--max-tiles-wsi", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-labeled", action="store_true",
                    help="skip slides whose patient has no row in labels.parquet; "
                         "those slides can never enter a bag, so extracting them "
                         "is pure cost")
    ap.add_argument("--keep-tiff", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print the tile-pass output structure for one slide and exit")
    ap.add_argument("--emb-key", default=None, help="pin the embedding field path")
    ap.add_argument("--pat-key", default=None, help="pin the pattern field path")
    ap.add_argument("--stain-normalize", action="store_true")
    ap.add_argument("--stain-target", default=None)
    ap.add_argument("--stain-method", default="reinhard",
                    choices=["macenko", "reinhard"])
    args = ap.parse_args()

    root = resolve_root(args.data_root)
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir else root / "wsi"
    cache_dir = root / "wsi_tiff"
    tag = f"_stain_{args.stain_method}" if args.stain_normalize else ""
    out_dir = Path(args.out_dir) if args.out_dir else root / f"features{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(args.lchai_app_dir).resolve()))
    try:
        from app.roi_inference_ctranspath_fuzzyarcloss_v3 import (  # type: ignore
            run_inference, InferenceConfig,
        )
        import app.roi_inference_ctranspath_fuzzyarcloss_v3 as roi  # type: ignore
    except Exception as exc:
        sys.exit(f"could not import LCHAI inference from {args.lchai_app_dir}: {exc}")

    models = Path(args.models_dir)
    ctp = models / "backbones" / "ctranspath.pth"
    fuzzy = models / "backbones" / "best_fuzzyarcloss_v2.pth"
    ckpts = models / "checkpoints"
    for p in (ctp, fuzzy, ckpts):
        if not p.exists():
            sys.exit(f"missing checkpoint path: {p}")

    # Optional stain normalisation, applied to the tiles before CTransPath sees
    # them -- identical to the transfer run's arm so the two are comparable.
    normalize_batch = None
    if args.stain_normalize:
        if not args.stain_target:
            sys.exit("--stain-normalize requires --stain-target")
        normalize_batch = build_stain_normalizer(
            args.stain_method, Path(args.stain_target), args.device)
        print(f"[stain] {args.stain_method} ON; target={args.stain_target}")

    # The capture hook: wrap the tile pass, keep its output, pass it through
    # untouched so the rest of the pipeline behaves exactly as in production.
    captured: dict = {}

    orig_tile = roi._real_tile_inference

    def patched(tiles, config):
        if normalize_batch is not None:
            normed = normalize_batch([im for (_x, _y, im) in tiles])
            tiles = [(x, y, normed[i]) for i, (x, y, _im) in enumerate(tiles)]
        out = orig_tile(tiles, config)
        captured["tiles_in"] = [(x, y) for (x, y, _im) in tiles]
        captured["out"] = out
        return out

    roi._real_tile_inference = patched

    items = list_slides(wsi_dir)
    if args.only_labeled:
        labels_path = root / "labels.parquet"
        if not labels_path.exists():
            sys.exit(f"--only-labeled needs {labels_path}\n"
                     f"run: python fetch_cptac_labels.py")
        genotyped = set(pd.read_parquet(labels_path).index.astype(str))
        before = len(items)
        items = [it for it in items if patient_of(it["src"]) in genotyped]
        print(f"--only-labeled: {len(items)}/{before} slides "
              f"({len(genotyped)} genotyped patients)")
        if not items:
            sys.exit("no slide matched a genotyped patient; check the patient-id "
                     "convention in labels.parquet against patient_of()")
    if args.limit:
        items = items[: args.limit]

    manifest_path = out_dir / "manifest.parquet"
    rows: list[dict] = []
    done: set[str] = set()
    if manifest_path.exists() and not args.inspect:
        try:
            prev = pd.read_parquet(manifest_path)
            rows = prev.to_dict("records")
            done = set(prev["slide"].astype(str))
            print(f"resuming: {len(done)} slides already extracted")
        except Exception as exc:
            print(f"[warn] unreadable manifest ({exc}); starting fresh")

    total = len(items)
    print(f"extracting features from {total} slides ({len(done)} done), "
          f"device={args.device}, top_k={args.top_k_tiles}")

    for i, item in enumerate(items, 1):
        slide_id = f"{item['name']}.tif" if item["type"] == "dicom" else item["src"].name
        if slide_id in done:
            continue
        pid = patient_of(item["src"])
        npz_path = out_dir / f"{Path(slide_id).stem}.npz"
        t0 = time.time()
        tif_to_cleanup = None
        try:
            if item["type"] == "dicom":
                wsi = convert_dicom_to_tiff(item["src"], cache_dir / f"{item['name']}.tif")
                if wsi is None:
                    print(f"[{i}/{total}] {pid}: SKIP (conversion failed)")
                    continue
                if not args.keep_tiff:
                    tif_to_cleanup = wsi
            else:
                wsi = item["src"]

            captured.clear()
            cfg = InferenceConfig(
                image_uri=str(wsi),
                image_format=wsi.suffix.lower().lstrip("."),
                device=args.device,
                model_backend="local",
                ctranspath_checkpoint=str(ctp),
                fuzzyarc_checkpoint=str(fuzzy),
                v2_checkpoint_dir=str(ckpts),
                use_choquet=True,
                top_k_tiles=args.top_k_tiles,
                max_tiles_wsi=args.max_tiles_wsi,
                genes=list(GENES),
                shap_decomposition_enabled=False,
                shap_enabled=False,
            )
            result = run_inference(wsi.read_bytes(), cfg)

            if "out" not in captured:
                sys.exit("the tile-pass hook never fired; the LCHAI entrypoint no "
                         "longer routes through _real_tile_inference. Re-point the "
                         "hook at the current tile function.")

            payload = {"tile_out": captured["out"], "result": result}
            if args.emb_key and args.pat_key:
                emb, pat, coords = pinned_channels(payload, args.emb_key, args.pat_key)
            else:
                emb, pat, coords = extract_channels(payload, inspect=args.inspect)

            if args.inspect:
                print(f"embeddings {emb.shape}, patterns {pat.shape}, "
                      f"coords {None if coords is None else coords.shape}")
                print("inspect done; not writing. Re-run without --inspect.")
                return

            if coords is None and captured.get("tiles_in"):
                tin = np.asarray(captured["tiles_in"], dtype=np.int32)
                coords = tin if tin.shape[0] == emb.shape[0] else None

            # Patterns must be a probability simplex per tile for the Choquet
            # integral to be well posed; renormalise if the head emits logits.
            pat = np.asarray(pat, dtype=np.float32)
            if pat.min() < 0.0 or pat.max() > 1.0:
                e = np.exp(pat - pat.max(axis=1, keepdims=True))
                pat = e / e.sum(axis=1, keepdims=True)

            np.savez_compressed(
                npz_path,
                emb=np.asarray(emb, dtype=np.float16),
                pat=pat.astype(np.float16),
                coords=(np.zeros((emb.shape[0], 2), np.int32)
                        if coords is None else np.asarray(coords, np.int32)),
            )
            rows.append({"patient_id": pid, "slide": slide_id,
                         "npz": str(npz_path), "n_tiles": int(emb.shape[0])})
            done.add(slide_id)
            print(f"[{i}/{total}] {pid} {slide_id}: {emb.shape[0]} tiles "
                  f"({time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"[{i}/{total}] {pid} {slide_id}: FAILED - {exc}")
            traceback.print_exc()
        finally:
            if tif_to_cleanup is not None:
                try:
                    tif_to_cleanup.unlink()
                except OSError:
                    pass

        if rows and (i % 10 == 0 or i == total):
            pd.DataFrame(rows).to_parquet(manifest_path)

    if not rows:
        sys.exit("no slides produced features")
    pd.DataFrame(rows).to_parquet(manifest_path)
    n_pat = pd.DataFrame(rows)["patient_id"].nunique()
    print(f"\n{len(rows)} slides / {n_pat} patients -> {out_dir}")
    print(f"next: python run_cptac_domainA_v2.py --features '{out_dir}'")


if __name__ == "__main__":
    main()
