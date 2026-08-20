#!/usr/bin/env python
"""Run LCHAI inference over the CPTAC-LUAD WSIs -> per-patient mutation probs.

This is the compute step of the within-domain cohort-transfer test. It is a
thin, standalone wrapper around LCHAI's production inference entrypoint
(`app.roi_inference_ctranspath_fuzzyarcloss_v3.run_inference`, model_backend
"local"), so every slide is processed with exactly the deployed pipeline:
OpenSlide tiling 224px@level0 + Otsu tissue mask -> CTransPath (512-d) +
ANORAK 6-pattern head -> top-200 tiles -> the best-method-per-gene MIL head
(baseline2 / concat-ABMIL / Fuzzy-Choquet, selected in
app/ml/checkpoints/loader.py). We read `result.mutation_scores` (a
gene->probability dict), aggregate slides to the patient by mean, and write a
predictions table for eval_cptac_transfer.py.

It does NOT modify the LCHAI repo; it only imports from it.

Input WSIs may be:
  * .svs / .tif / .tiff  -- fed to LCHAI directly (native, reliable).
  * DICOM VL Whole Slide Microscopy (as downloaded from IDC) -- converted to a
    pyramidal TIFF first via pyvips' openslideload, which requires libvips
    built against OpenSlide >= 4.0 (DICOM support). If that is unavailable,
    pull the .svs from the GDC instead and point --wsi-dir at those.

Checkpoints required (pass --models-dir pointing at the LCHAI models/ dir):
    models/backbones/ctranspath.pth
    models/backbones/best_fuzzyarcloss_v2.pth
    models/checkpoints/ckpt_*_{GENE}_fold{K}.pth   (the six best per loader.py)

Usage (on the DGX):
    python run_cptac_inference.py \
        --lchai-app-dir "/home/rapids/notebooks/slima/lchai/oncology-xai/apps/inference-service" \
        --models-dir    "/home/rapids/notebooks/slima/lchai/oncology-xai/models" \
        --wsi-dir       "$DATA_ROOT/luad_cptac/wsi" \
        --device cuda
    # -> $DATA_ROOT/luad_cptac/cptac_predictions.parquet
    # then: python eval_cptac_transfer.py --preds "$DATA_ROOT/luad_cptac/cptac_predictions.parquet"
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

GENES = ["TP53", "EGFR", "KRAS", "STK11", "KEAP1", "RBM10"]
# CPTAC case identifiers look like C3L-00001 / C3N-00169.
PATIENT_RE = re.compile(r"C3[LN]-\d{5}", re.IGNORECASE)
WSI_EXTS = {".svs", ".tif", ".tiff", ".bif"}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "luad_cptac"


def patient_of(path: Path) -> str:
    """Extract the CPTAC patient id from a slide path, else use the stem."""
    m = PATIENT_RE.search(str(path))
    return m.group(0).upper() if m else path.stem


# --------------------------------------------------------------------------
# DICOM -> pyramidal TIFF (only used when inputs are DICOM series)
# --------------------------------------------------------------------------

def openslide_supports_dicom() -> bool:
    try:
        import openslide
        ver = tuple(int(x) for x in openslide.__library_version__.split(".")[:2])
        return ver >= (4, 0)
    except Exception:
        return False


def find_dicom_series(wsi_dir: Path) -> list[Path]:
    """Return one representative .dcm per directory that contains DICOM files."""
    series = {}
    for dcm in wsi_dir.rglob("*.dcm"):
        series.setdefault(dcm.parent, dcm)
    return list(series.values())


def convert_dicom_to_tiff(rep_dcm: Path, out_tif: Path) -> Path | None:
    """Convert one DICOM WSI series to a pyramidal TIFF via pyvips/OpenSlide."""
    if out_tif.exists():
        return out_tif
    try:
        import pyvips
    except Exception as exc:
        print(f"[error] pyvips unavailable ({exc}); cannot convert DICOM")
        return None
    if not openslide_supports_dicom():
        print("[error] OpenSlide < 4.0: no DICOM support. Pull .svs from GDC "
              "instead of converting.")
        return None
    try:
        img = pyvips.Image.openslideload(str(rep_dcm), level=0)
        # Preserve microns-per-pixel. A plain tiffsave drops the resolution, so
        # the downstream pipeline reads a bogus openslide.mpp-x and emits a
        # spurious "LOW MAGNIFICATION" warning. Tiling is level-0 regardless,
        # but we keep the metadata accurate (px/mm -> written as px/cm).
        mppx = mppy = None
        try:
            import openslide
            _os = openslide.OpenSlide(str(rep_dcm))
            mppx = float(_os.properties.get("openslide.mpp-x") or 0.0) or None
            mppy = float(_os.properties.get("openslide.mpp-y") or 0.0) or mppx
            _os.close()
        except Exception:
            pass
        if mppx:
            img = img.copy(xres=1000.0 / mppx, yres=1000.0 / (mppy or mppx))
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        img.tiffsave(str(out_tif), pyramid=True, tile=True,
                     tile_width=256, tile_height=256,
                     compression="jpeg", Q=90, bigtiff=True, resunit="cm")
        return out_tif
    except Exception as exc:
        print(f"[error] conversion failed for {rep_dcm.parent.name}: {exc}")
        return None


def list_slides(wsi_dir: Path) -> list[dict]:
    """Enumerate slides as work items WITHOUT converting anything yet.

    Each item is {"type": "native"|"dicom", "src": Path, "name": str}. DICOM
    series are converted lazily inside the main loop (one at a time, then the
    TIFF is deleted) so we never materialise ~500 pyramidal TIFFs at once.
    """
    native = [p for p in wsi_dir.rglob("*") if p.suffix.lower() in WSI_EXTS]
    if native:
        print(f"found {len(native)} native WSI files ({', '.join(sorted(WSI_EXTS))})")
        return [{"type": "native", "src": p, "name": p.stem} for p in sorted(native)]
    series = find_dicom_series(wsi_dir)
    if not series:
        sys.exit(f"no WSIs found under {wsi_dir} (.svs/.tif or DICOM series)")
    print(f"found {len(series)} DICOM series (converted lazily during inference)")
    items = []
    for rep in series:
        # The SM_ series-dir name is the full SeriesInstanceUID; use it whole so
        # a patient's multiple slides never collide (truncating to 12 chars made
        # every SM_1.3.6.1.4... series map to the same TIFF).
        items.append({"type": "dicom", "src": rep,
                      "name": patient_of(rep) + "_" + rep.parent.name})
    return sorted(items, key=lambda d: d["name"])


def _save_predictions(rows: list[dict], out_dir: Path, out_path: Path,
                      per_slide_path: Path) -> int:
    """Persist per-slide rows and the patient-level aggregate. Returns #patients."""
    if not rows:
        return 0
    slide_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide_df.to_parquet(per_slide_path)
    prob_cols = [f"prob_{g}" for g in GENES]
    patient_df = slide_df.groupby("patient_id")[prob_cols].mean().reset_index()
    patient_df.to_parquet(out_path)
    patient_df.to_csv(out_path.with_suffix(".csv"), index=False)
    return patient_df["patient_id"].nunique()


# --------------------------------------------------------------------------
# Optional stain normalization (test-time domain adaptation)
# --------------------------------------------------------------------------

# Reinhard lab (lαβ) colour transform (Reinhard et al. 2001). Batched in torch so
# a whole tile batch is normalized in one GPU pass -- torchstain's per-tile Python
# loop made a full slide ~6x slower, which is untenable over 498 slides.
_RGB2LMS = [[0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]]
_LOGLMS2LAB = [[1 / 3 ** 0.5, 1 / 3 ** 0.5, 1 / 3 ** 0.5],
               [1 / 6 ** 0.5, 1 / 6 ** 0.5, -2 / 6 ** 0.5],
               [1 / 2 ** 0.5, -1 / 2 ** 0.5, 0.0]]


def _lab_mats(device, dtype):
    import torch
    m1 = torch.tensor(_RGB2LMS, device=device, dtype=dtype)
    m2 = torch.tensor(_LOGLMS2LAB, device=device, dtype=dtype)
    return m1, m2, torch.linalg.inv(m1), torch.linalg.inv(m2)


def _rgb01_to_lab(rgb, m1, m2):
    import torch
    lms = torch.clamp(rgb @ m1.T, min=1e-6)
    return torch.log10(lms) @ m2.T


def _lab_to_rgb01(lab, m1inv, m2inv):
    import torch
    lms = torch.pow(10.0, lab @ m2inv.T)
    return torch.clamp(lms @ m1inv.T, 0.0, 1.0)


def build_stain_normalizer(method: str, target_path: Path, device: str = "cuda"):
    """Return a fn list[PIL]->list[PIL] that stain-normalizes tiles to `target_path`.

    The target should be a representative TCGA-LUAD H&E tile, so CPTAC tiles are
    mapped into the colour space the model was actually trained on (raw TCGA +
    ImageNet norm; the pipeline applies no stain normalization).

    method: "reinhard" -- lab mean/std matching, batched on GPU (fast; good for a
                          global colour cast). Own torch implementation.
            "macenko"  -- torchstain stain-matrix deconvolution, per-tile (fidelity
                          at the cost of an SVD per tile; much slower).
    """
    import torch
    import numpy as np
    from PIL import Image

    ref = Image.open(target_path).convert("RGB")

    if method == "reinhard":
        use = "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        ref_arr = np.asarray(ref).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_arr).to(use).reshape(-1, 3)
        m1, m2, _, _ = _lab_mats(use, ref_t.dtype)
        ref_lab = _rgb01_to_lab(ref_t, m1, m2)
        tgt_mean = ref_lab.mean(0)              # (3,)
        tgt_std = ref_lab.std(0) + 1e-6         # (3,)

        def normalize_batch(pils, chunk=128):
            if not pils:
                return pils
            m1_, m2_, m1i, m2i = _lab_mats(use, tgt_mean.dtype)
            out = []
            with torch.no_grad():
                for i in range(0, len(pils), chunk):
                    sub = [np.asarray(p.convert("RGB")) for p in pils[i:i + chunk]]
                    # Guard against non-uniform tile sizes (shouldn't happen at 224).
                    if len({a.shape for a in sub}) != 1:
                        out.extend(pils[i:i + chunk])
                        continue
                    arr = np.stack(sub).astype(np.float32) / 255.0
                    t = torch.from_numpy(arr).to(use)         # N,H,W,3
                    N, H, W, _ = t.shape
                    lab = _rgb01_to_lab(t.reshape(N, H * W, 3), m1_, m2_)
                    mu = lab.mean(dim=1, keepdim=True)
                    sd = lab.std(dim=1, keepdim=True) + 1e-6
                    lab = (lab - mu) / sd * tgt_std + tgt_mean
                    rgb = _lab_to_rgb01(lab, m1i, m2i).reshape(N, H, W, 3)
                    rgb = (rgb * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    out.extend(Image.fromarray(rgb[j], "RGB") for j in range(N))
            return out
        return normalize_batch

    # macenko: per-tile torchstain (slower; opt-in for higher fidelity).
    import torchstain
    from torchvision import transforms
    to255 = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x * 255)])
    norm = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    norm.fit(to255(ref))

    def normalize_batch(pils):
        out = []
        for img in pils:
            try:
                o, _, _ = norm.normalize(I=to255(img.convert("RGB")), stains=False)
                out.append(Image.fromarray(o.numpy().astype("uint8"), "RGB"))
            except Exception:
                out.append(img)  # e.g. near-blank tile the estimator cannot fit
        return out
    return normalize_batch


def install_stain_normalizer(module, method: str, target_path: Path, device: str = "cuda") -> None:
    """Wrap the tile pass so every tile is stain-normalized before CTransPath.

    Patches `_real_tile_inference` on the imported production module WITHOUT
    editing it on disk, so TCGA/production behaviour is untouched and only this
    CPTAC run normalizes.
    """
    normalize_batch = build_stain_normalizer(method, target_path, device)
    orig = module._real_tile_inference

    def patched(tiles, config):
        normed = normalize_batch([im for (_x, _y, im) in tiles])
        tiles = [(x, y, normed[i]) for i, (x, y, _im) in enumerate(tiles)]
        return orig(tiles, config)

    module._real_tile_inference = patched
    print(f"[stain] {method} normalization ON (batched); target={target_path}")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lchai-app-dir", required=True,
                    help="path to LCHAI apps/inference-service (holds the `app` package)")
    ap.add_argument("--models-dir", required=True,
                    help="LCHAI models/ dir with backbones/ and checkpoints/")
    ap.add_argument("--wsi-dir", default=None, help="dir of CPTAC WSIs (default $DATA_ROOT/luad_cptac/wsi)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k-tiles", type=int, default=200)
    ap.add_argument("--max-tiles-wsi", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None, help="process only N slides (smoke test)")
    ap.add_argument("--save-every", type=int, default=10,
                    help="flush predictions to disk every N slides (crash-safe)")
    ap.add_argument("--keep-tiff", action="store_true",
                    help="keep converted DICOM->TIFF files (default: delete after each slide)")
    ap.add_argument("--stain-normalize", action="store_true",
                    help="Macenko-normalize each tile to --stain-target before CTransPath "
                         "(test-time domain adaptation; needs `pip install torchstain`)")
    ap.add_argument("--stain-target", default=None,
                    help="reference H&E tile (ideally a representative TCGA-LUAD tile) that "
                         "defines the target colour space the model was trained on")
    ap.add_argument("--stain-method", default="macenko", choices=["macenko", "reinhard"],
                    help="macenko (standard, ~SVD/tile) or reinhard (LAB mean/std, faster)")
    args = ap.parse_args()

    out_dir = resolve_root(args.data_root)
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir else out_dir / "wsi"
    cache_dir = out_dir / "wsi_tiff"
    tag = f"_stain_{args.stain_method}" if args.stain_normalize else ""
    out_path = Path(args.out) if args.out else out_dir / f"cptac_predictions{tag}.parquet"

    # Import LCHAI's production inference entrypoint.
    sys.path.insert(0, str(Path(args.lchai_app_dir).resolve()))
    try:
        from app.roi_inference_ctranspath_fuzzyarcloss_v3 import (  # type: ignore
            run_inference, InferenceConfig,
        )
    except Exception as exc:
        sys.exit(f"could not import LCHAI inference from {args.lchai_app_dir}: {exc}")

    if args.stain_normalize:
        if not args.stain_target:
            sys.exit("--stain-normalize requires --stain-target <reference H&E tile>")
        import app.roi_inference_ctranspath_fuzzyarcloss_v3 as _roi  # type: ignore
        install_stain_normalizer(_roi, args.stain_method, Path(args.stain_target), args.device)

    models = Path(args.models_dir)
    ctp = models / "backbones" / "ctranspath.pth"
    fuzzy = models / "backbones" / "best_fuzzyarcloss_v2.pth"
    ckpts = models / "checkpoints"
    for p in (ctp, fuzzy, ckpts):
        if not p.exists():
            sys.exit(f"missing checkpoint path: {p}")

    items = list_slides(wsi_dir)
    if args.limit:
        items = items[: args.limit]

    # Resume: a long cohort run can be interrupted; reload what is already done
    # (keyed by the slide id) and skip those, so a re-launch continues.
    per_slide_path = out_path.with_name(out_path.stem + "_per_slide.parquet")
    rows: list[dict] = []
    done: set[str] = set()
    if per_slide_path.exists():
        try:
            prev = pd.read_parquet(per_slide_path)
            rows = prev.to_dict("records")
            done = set(prev["slide"].astype(str))
            print(f"resuming: {len(done)} slides already in {per_slide_path.name}")
        except Exception as exc:
            print(f"[warn] could not read existing predictions ({exc}); starting fresh")

    total = len(items)
    print(f"running inference on {total} slides ({len(done)} done), device={args.device}")

    for i, item in enumerate(items, 1):
        slide_id = f"{item['name']}.tif" if item["type"] == "dicom" else item["src"].name
        if slide_id in done:
            continue
        pid = patient_of(item["src"])
        t0 = time.time()
        tif_to_cleanup = None
        try:
            if item["type"] == "dicom":
                wsi = convert_dicom_to_tiff(item["src"], cache_dir / f"{item['name']}.tif")
                if wsi is None:
                    print(f"[{i}/{total}] {pid} {item['name']}: SKIP (conversion failed)")
                    continue
                if not args.keep_tiff:
                    tif_to_cleanup = wsi
            else:
                wsi = item["src"]
            fmt = wsi.suffix.lower().lstrip(".")

            image_bytes = wsi.read_bytes()
            cfg = InferenceConfig(
                image_uri=str(wsi),
                image_format=fmt,
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
            result = run_inference(image_bytes, cfg)
            scores = dict(result.mutation_scores or {})
            row = {"patient_id": pid, "slide": slide_id, "n_tiles": result.n_tiles}
            for g in GENES:
                row[f"prob_{g}"] = float(scores[g]) if g in scores else np.nan
            rows.append(row)
            done.add(slide_id)
            print(f"[{i}/{total}] {pid} {slide_id}: "
                  + ", ".join(f"{g}={row[f'prob_{g}']:.3f}" for g in GENES if not np.isnan(row[f'prob_{g}']))
                  + f"  ({result.n_tiles} tiles, {time.time()-t0:.0f}s)")
        except Exception as exc:
            print(f"[{i}/{total}] {pid} {slide_id}: FAILED — {exc}")
            traceback.print_exc()
        finally:
            # Free disk: drop the converted TIFF once the slide is scored.
            if tif_to_cleanup is not None:
                try:
                    tif_to_cleanup.unlink()
                except OSError:
                    pass

        # Crash-safe incremental flush.
        if rows and (i % args.save_every == 0 or i == total):
            n_pat = _save_predictions(rows, out_dir, out_path, per_slide_path)
            print(f"    .. flushed {len(rows)} slides / {n_pat} patients to disk")

    if not rows:
        sys.exit("no slides produced predictions")

    n_pat = _save_predictions(rows, out_dir, out_path, per_slide_path)
    print(f"\n{len(rows)} slides -> {n_pat} patients")
    print(f"wrote {out_path}")
    print("next: python eval_cptac_transfer.py --preds "
          f"'{out_path}'")


if __name__ == "__main__":
    main()
