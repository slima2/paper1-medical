"""Real multiparametric-MRI panel from a TCGA-LGG glioma, with its VASARI
patterns, plotted as the radiological counterpart of the real PTB-XL ECG panel
and the real LUAD histopathology panels.

It shows an actual preoperative case (BraTS-TCGA-LGG NIfTI): the four
co-registered sequences the encoder E ingests in run_glioma_domainB.py
(T1, T1-Gd, T2, FLAIR) at the tumour-centroid axial slice with the expert
segmentation contour, next to the neuroradiologist VASARI scores (the source
channel P) the framework fuses. IDH / 1p19q status is shown in the title.

Run this ON THE DGX (it needs the NIfTI + VASARI on disk):
    cd figures && python gen_glioma_example.py
Optional: --subject TCGA-DU-8166   (default: an IDH-mutant, 1p/19q-codeleted
case if one is available, else the first usable subject).

Output (in this folder):
    glioma_example.png
Then copy glioma_example.png back into the paper's figures/ folder to compile.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception as exc:  # pragma: no cover
    sys.exit(f"nibabel is required: {exc}")

HERE = Path(__file__).resolve().parent

SEQS = ("t1", "t1ce", "t2", "flair")
SEQ_TITLE = {"t1": "T1", "t1ce": "T1-Gd", "t2": "T2", "flair": "FLAIR"}
SEQ_SUFFIXES = {
    "flair": ("_flair", "_fl"),
    "t2": ("_t2",),
    "t1ce": ("_t1ce", "_t1gd", "_t1Gd", "_t1c"),
    "t1": ("_t1",),
}
SEG_SUFFIXES = ("ManuallyCorrected", "GlistrBoost", "_glistrboost", "_seg")

# Standard VASARI MR feature key (Gutman 2013; TCIA VASARI Research Project).
# Used only to render human-readable labels; if a feature-key CSV is present it
# takes precedence, and any unknown column falls back to its raw code.
VASARI_KEY = {
    "f1": "Tumour location", "f2": "Side of epicentre", "f3": "Eloquent brain",
    "f4": "Enhancement quality", "f5": "Proportion enhancing",
    "f6": "Proportion nCET", "f7": "Proportion necrosis", "f8": "Cyst(s)",
    "f9": "Multifocal/-centric", "f10": "T1/FLAIR ratio",
    "f11": "Enhancing-margin thickness", "f12": "Enhancing-margin definition",
    "f13": "Non-enhancing-margin definition", "f14": "Proportion of oedema",
    "f15": "Oedema crosses midline", "f16": "Haemorrhage", "f17": "Diffusion",
    "f18": "Pial invasion", "f19": "Ependymal invasion",
    "f20": "Cortical involvement", "f21": "Deep white-matter invasion",
    "f22": "nCET crosses midline", "f23": "Enhancing tumour crosses midline",
    "f24": "Satellites", "f25": "Calvarial remodelling",
    "f26": "Extent of resection", "f27": "Lesion size (main axis)",
    "f28": "Lesion size (perp. axis)", "f29": "Lesion size (third axis)",
    "f30": "Cortical/eloquent adjacency",
}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma"


def find_seq(subj_dir: Path, seq: str) -> Path | None:
    for suf in SEQ_SUFFIXES[seq]:
        hits = sorted(subj_dir.glob(f"*{suf}.nii*"))
        if hits:
            return hits[0]
    return None


def find_seg(subj_dir: Path) -> Path | None:
    for suf in SEG_SUFFIXES:
        hits = sorted(subj_dir.glob(f"*{suf}*.nii*"))
        if hits:
            return hits[0]
    return None


def subject_dir(glioma_root: Path, subject: str) -> Path | None:
    hits = [p for p in glioma_root.rglob(subject) if p.is_dir()]
    hits = [p for p in hits if find_seq(p, "flair") is not None]
    return hits[0] if hits else None


def all_subject_dirs(glioma_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in glioma_root.rglob("TCGA-*"):
        if p.is_dir() and find_seq(p, "flair") is not None:
            out.setdefault(p.name.upper(), p)
    return out


def pick_subject(glioma_root: Path, dirs: dict[str, Path]) -> str:
    """Prefer an IDH-mutant, 1p/19q-codeleted case (the regime-(ii) target)."""
    labels_path = glioma_root / "derived" / "glioma_labels.parquet"
    if labels_path.exists():
        lab = pd.read_parquet(labels_path)
        lab.index = [str(s).upper() for s in lab.index]
        avail = [s for s in lab.index if s in dirs]
        lab = lab.reindex(avail)
        for query in (
            (lab.get("idh") == 1) & (lab.get("onep19q") == 1),
            (lab.get("onep19q") == 1),
            (lab.get("idh") == 1),
        ):
            hit = lab.index[query.fillna(False)]
            if len(hit):
                return hit[0]
    return sorted(dirs)[0]


def load_vasari_row(glioma_root: Path, subject: str) -> pd.Series | None:
    p = glioma_root / "vasari" / "vasari_features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = [str(s).upper() for s in df.index]
    if subject not in df.index:
        return None
    row = df.loc[subject].drop(labels=[c for c in ("source_file",) if c in df.columns],
                               errors="ignore")
    return pd.to_numeric(row, errors="coerce").dropna()


def feature_label(code: str) -> str:
    return VASARI_KEY.get(str(code).lower(), str(code))


def norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = a.max() if a.max() > lo else lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0, 1)


def tumour_slice(seg: np.ndarray) -> int:
    if seg is not None and seg.sum() > 0:
        per_slice = (seg > 0).sum(axis=(0, 1))
        return int(np.argmax(per_slice))
    return seg.shape[2] // 2 if seg is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()

    glioma_root = resolve_root(args.data_root)
    dirs = all_subject_dirs(glioma_root)
    if not dirs:
        sys.exit(f"no TCGA-* NIfTI subject folders under {glioma_root}")

    subject = (args.subject or pick_subject(glioma_root, dirs)).upper()
    sdir = dirs.get(subject) or subject_dir(glioma_root, subject)
    if sdir is None:
        sys.exit(f"subject {subject} not found with imaging under {glioma_root}")
    print(f"subject: {subject}  dir: {sdir}")

    seg_p = find_seg(sdir)
    seg = np.asarray(nib.load(str(seg_p)).dataobj) if seg_p else None
    z = tumour_slice(seg)

    vols: dict[str, np.ndarray] = {}
    for s in SEQS:
        sp = find_seq(sdir, s)
        if sp is not None:
            vols[s] = np.asarray(nib.load(str(sp)).dataobj)
    if not vols:
        sys.exit(f"no sequences readable for {subject}")

    # genotype for the title
    lab_txt = ""
    lp = glioma_root / "derived" / "glioma_labels.parquet"
    if lp.exists():
        lab = pd.read_parquet(lp); lab.index = [str(s).upper() for s in lab.index]
        if subject in lab.index:
            r = lab.loc[subject]
            def flag(v, pos, neg):
                if pd.isna(v): return "n/a"
                return pos if v == 1 else neg
            lab_txt = (f"IDH {flag(r.get('idh'), 'mutant', 'wildtype')}   |   "
                       f"1p/19q {flag(r.get('onep19q'), 'codeleted', 'intact')}")

    vas = load_vasari_row(glioma_root, subject)

    # ---- figure: 4 sequence panels (top) + VASARI patterns (bottom) ----
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.9], hspace=0.18, wspace=0.06)

    for j, s in enumerate(SEQS):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        if s in vols:
            img = norm01(np.rot90(vols[s][:, :, z]))
            ax.imshow(img, cmap="gray", interpolation="nearest")
            if seg is not None:
                m = np.rot90(seg[:, :, z]) > 0
                ax.contour(m, levels=[0.5], colors=["#ef4444"], linewidths=1.1)
            ax.set_title(SEQ_TITLE[s], fontsize=12, fontweight="bold",
                         color="#111827")
        else:
            ax.text(0.5, 0.5, f"{SEQ_TITLE[s]}\n(missing)", ha="center",
                    va="center", fontsize=10, color="#9ca3af")

    axp = fig.add_subplot(gs[1, :])
    if vas is not None and len(vas):
        # drop the raw size measurements (f27-f29): they are on a different
        # scale and would dominate the axis over the ordinal pattern scores.
        drop = {"f27", "f28", "f29"}
        vas2 = vas[[c for c in vas.index if str(c).lower() not in drop]]
        top = vas2.sort_values(ascending=False).head(8)
        labels = [feature_label(c) for c in top.index]
        y = np.arange(len(top))[::-1]
        axp.barh(y, top.values, color="#2563eb", height=0.62, zorder=3)
        axp.set_yticks(y)
        axp.set_yticklabels(labels, fontsize=9, color="#111827")
        axp.tick_params(left=False)
        vmax = float(max(top.values)) if len(top) else 1.0
        for yi, val in zip(y, top.values):
            axp.text(val + vmax * 0.012, yi, f"{val:g}", fontsize=8,
                     va="center", ha="left", color="#374151")
        axp.set_xlim(0, vmax * 1.15)
        axp.set_ylim(-0.6, len(top) - 0.4)
        axp.set_xticks([])
        for sp in ("top", "right", "bottom", "left"):
            axp.spines[sp].set_visible(False)
        axp.set_title("VASARI expert features (source channel P) - "
                      "highest-scored for this case", fontsize=10.5,
                      fontweight="bold", color="#111827", loc="left")
    else:
        axp.axis("off")
        axp.text(0.5, 0.5, "VASARI scores unavailable for this subject",
                 ha="center", va="center", fontsize=10, color="#9ca3af")

    title = f"TCGA-LGG glioma — preoperative multiparametric MRI (axial slice {z})"
    if lab_txt:
        title += f"\n{lab_txt}"
    fig.suptitle(title, fontsize=12.5, fontweight="bold", color="#111827", y=0.99)

    out = HERE / "glioma_example.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
