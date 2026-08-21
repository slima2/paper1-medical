#!/usr/bin/env python
"""Preprocess glioma NIfTI volumes into the per-subject slice tensors and the
manifest that run_glioma_domainB.py consumes.

Handles both cohorts (subject folders auto-detected):
  * BraTS 2021 :  BraTS2021_XXXXX/  with *_t1 *_t1ce *_t2 *_flair *_seg
  * BraTS-TCGA-LGG/GBM (Bakas NIfTI) : TCGA-XX-XXXX/ with *_t1 *_t1Gd *_t2
    *_flair and a *ManuallyCorrected*/*GlistrBoost* segmentation.
The t1ce channel maps to _t1Gd when the BraTS-2021-style name is absent.

For each subject folder containing the four co-registered sequences and a
segmentation this script:
  1. z-score normalises each sequence within the brain mask (nonzero voxels),
     leakage-free (per-subject statistics only);
  2. finds the tumour centroid from the segmentation (label > 0), or the brain
     centroid if no segmentation is present;
  3. selects K axial slices spanning the tumour extent (falls back to a band
     around the centroid), centre-crops each to HxW around the tumour, stacks
     the four sequences -> float32 [K, 4, H, W];
  4. saves derived/slices/<subject>.npy.

It then builds derived/manifest.parquet with columns
    subject_id, strat_fold (1..5), slices_path
stratified on MGMT (the scarcest target) when derived/glioma_labels.parquet
exists, else stratified at random.

Constants K=24, H=W=128 match run_glioma_domainB.py (SLICES_PER_SUBJECT, SLICE_HW).

Usage (on the DGX, after `source env.sh`, needs nibabel + scikit-learn):
    python prepare_glioma_slices.py
    python prepare_glioma_slices.py --limit 20   # smoke: first 20 subjects
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    sys.exit(f"nibabel is required (pip install nibabel): {exc}")

K = 24            # SLICES_PER_SUBJECT
HW = 128          # SLICE_HW
N_FOLDS = 5
SEED = 42
SEQS = ("t1", "t1ce", "t2", "flair")

# Candidate filename suffixes per canonical sequence. BraTS 2021 uses
# _t1ce; the Bakas BraTS-TCGA-LGG/GBM NIfTI use _t1Gd. Order matters only for
# disambiguating t1 from t1ce/t1Gd (t1 is matched last, exact).
SEQ_SUFFIXES = {
    "flair": ("_flair", "_fl"),
    "t2":    ("_t2",),
    "t1ce":  ("_t1ce", "_t1gd", "_t1Gd", "_t1c"),
    "t1":    ("_t1",),
}
# Segmentation file suffixes, in priority order (manual first).
SEG_SUFFIXES = ("_seg", "ManuallyCorrected", "GlistrBoost", "_glistrboost")

# Subject-folder id patterns: BraTS 2021 ids and TCGA ids (LGG/GBM NIfTI).
SUBJECT_GLOBS = ("BraTS2021_*", "TCGA-*")


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma"


def find_subject_dirs(glioma_root: Path) -> list[Path]:
    """Recursively find subject dirs (BraTS2021_* or TCGA-*) holding NIfTI."""
    seen: dict[str, Path] = {}
    for pattern in SUBJECT_GLOBS:
        for p in glioma_root.rglob(pattern):
            if p.is_dir() and _find_seq(p, "flair") is not None:
                seen.setdefault(p.name, p)
    return [seen[k] for k in sorted(seen)]


def _find_seq(subj_dir: Path, seq: str) -> Path | None:
    """Match a canonical sequence against any of its known filename suffixes."""
    for suf in SEQ_SUFFIXES[seq]:
        hits = sorted(subj_dir.glob(f"*{suf}.nii*"))
        if hits:
            return hits[0]
    return None


def seq_path(subj_dir: Path, seq: str) -> Path | None:
    return _find_seq(subj_dir, seq)


def seg_path(subj_dir: Path) -> Path | None:
    for suf in SEG_SUFFIXES:
        hits = sorted(subj_dir.glob(f"*{suf}*.nii*"))
        if hits:
            return hits[0]
    return None


def zscore_brain(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if mask.sum() < 100:
        return vol.astype(np.float32)
    mu = vol[mask].mean()
    sd = vol[mask].std() + 1e-6
    out = (vol - mu) / sd
    out[~mask] = 0.0
    return out.astype(np.float32)


def centre_crop(a2d: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    h, w = a2d.shape
    half = size // 2
    y0, x0 = cy - half, cx - half
    out = np.zeros((size, size), dtype=a2d.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y0 + size, h), min(x0 + size, w)
    dy0, dx0 = sy0 - y0, sx0 - x0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = a2d[sy0:sy1, sx0:sx1]
    return out


def select_slice_indices(seg: np.ndarray | None, nz: int) -> tuple[list[int], int, int]:
    """Return (axial slice indices, centroid y, centroid x)."""
    if seg is not None and (seg > 0).any():
        zs = np.where((seg > 0).any(axis=(0, 1)))[0]
        z0, z1 = zs.min(), zs.max()
        ys, xs, _ = np.where(seg > 0)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    else:
        z0, z1 = int(nz * 0.30), int(nz * 0.70)
        cy = cx = None
    if z1 - z0 + 1 >= K:
        idx = np.linspace(z0, z1, K).round().astype(int).tolist()
    else:  # pad the band symmetrically around the centre
        cz = (z0 + z1) // 2
        idx = np.linspace(cz - K // 2, cz + K // 2 - 1, K).round().astype(int)
        idx = np.clip(idx, 0, nz - 1).tolist()
    return idx, cy, cx


def process_subject(subj_dir: Path, out_dir: Path) -> str | None:
    vols = {}
    for s in SEQS:
        p = seq_path(subj_dir, s)
        if p is None:
            print(f"[skip] {subj_dir.name}: missing sequence {s}")
            return None
        vols[s] = zscore_brain(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32))
    seg_p = seg_path(subj_dir)
    seg = np.asarray(nib.load(str(seg_p)).dataobj) if seg_p else None

    ref = vols["flair"]
    nz = ref.shape[2]
    idx, cy, cx = select_slice_indices(seg, nz)
    if cy is None:
        cy, cx = ref.shape[0] // 2, ref.shape[1] // 2

    arr = np.zeros((K, len(SEQS), HW, HW), dtype=np.float32)
    for k, z in enumerate(idx):
        for c, s in enumerate(SEQS):
            arr[k, c] = centre_crop(vols[s][:, :, z], cy, cx, HW)

    out_path = out_dir / f"{subj_dir.name}.npy"
    np.save(out_path, arr)
    return str(out_path)


def build_folds(subjects: list[str], glioma_root: Path) -> dict[str, int]:
    labels_path = glioma_root / "derived" / "glioma_labels.parquet"
    strat = None
    if labels_path.exists():
        lab = pd.read_parquet(labels_path).reindex(subjects)
        # Stratify on the scarcest available binary target (mgmt for the BraTS
        # run; onep19q/idh for the LGG-centric v1 where mgmt is all-NaN).
        for col in ("mgmt", "onep19q", "idh"):
            if col in lab and lab[col].notna().any():
                strat = lab[col].fillna(-1).astype(int).values
                print(f"[ok] folds stratified on '{col}'")
                break
    rng = np.random.default_rng(SEED)
    folds: dict[str, int] = {}
    if strat is not None:
        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            for f, (_, te) in enumerate(skf.split(subjects, strat), start=1):
                for i in te:
                    folds[subjects[i]] = f
            return folds
        except Exception as exc:
            print(f"[warn] stratified fold split failed ({exc}); random folds")
    order = rng.permutation(len(subjects))
    for rank, i in enumerate(order):
        folds[subjects[i]] = (rank % N_FOLDS) + 1
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="process only N subjects")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    glioma_root = resolve_root(args.data_root)
    subj_dirs = find_subject_dirs(glioma_root)
    if not subj_dirs:
        sys.exit(f"no BraTS2021_* or TCGA-* NIfTI subject folders under "
                 f"{glioma_root}; see VASARI_RUN_PLAN.md step 3 (imaging).")
    if args.limit:
        subj_dirs = subj_dirs[: args.limit]

    slice_dir = glioma_root / "derived" / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(subj_dirs, 1):
        out_path = slice_dir / f"{d.name}.npy"
        if out_path.exists() and not args.overwrite:
            rows.append({"subject_id": d.name, "slices_path": str(out_path)})
            continue
        try:
            sp = process_subject(d, slice_dir)
        except Exception as exc:
            print(f"[{i}/{len(subj_dirs)}] {d.name}: FAILED {exc}")
            continue
        if sp:
            rows.append({"subject_id": d.name, "slices_path": sp})
        if i % 25 == 0:
            print(f"  processed {i}/{len(subj_dirs)}")

    if not rows:
        sys.exit("no subjects processed")

    subjects = [r["subject_id"] for r in rows]
    folds = build_folds(subjects, glioma_root)
    for r in rows:
        r["strat_fold"] = folds[r["subject_id"]]

    manifest = pd.DataFrame(rows)[["subject_id", "strat_fold", "slices_path"]]
    out = glioma_root / "derived" / "manifest.parquet"
    manifest.to_parquet(out, index=False)
    print(f"\n{len(manifest)} subjects -> {out}")
    print(manifest["strat_fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
