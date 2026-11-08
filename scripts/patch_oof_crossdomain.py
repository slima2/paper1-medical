"""Patch the ECG and glioma runners so they write out-of-fold predictions.

E9 of the DGX plan. Both runners (paper1_nature/run_ecg_domainC.py and
paper1_nature/glioma/run_glioma_domainB.py) only save per-fold AUROCs; the
noise-aware rule of the paper needs per-case OOF probabilities so that the
hierarchical bootstrap can put an interval on every fusion contrast.

The patch is minimal and idempotent:
  * train_eval() also returns the test-fold probabilities (and, for glioma,
    the aligned labels);
  * main() writes derived/<domain>/oof/oof_<cond>_<target>_fold<k>.csv right
    after each fold, with columns  record_id, patient_id, target, condition,
    fold, label, prob.

Usage (from paper1_nature/):
    python patch_oof_crossdomain.py            # patch both
    python patch_oof_crossdomain.py --check    # only report what would change
A .bak copy of each runner is written next to it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ECG = HERE / "run_ecg_domainC.py"
GLI = HERE / "glioma" / "run_glioma_domainB.py"

MARK = "# --- OOF patch (E9) ---"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"[{label}] expected exactly 1 occurrence, found {n}:\n{old}")
    return text.replace(old, new)


def patch_ecg(text: str) -> str:
    text = replace_once(
        text,
        "    return float(auroc), inter\n",
        "    return float(auroc), inter, prob\n",
        "ecg: train_eval return",
    )
    text = replace_once(
        text,
        "    results, regimes, interactions = {}, {}, {}\n",
        "    results, regimes, interactions = {}, {}, {}\n"
        f"    {MARK}\n"
        '    oof_dir = ecg_root / "derived" / "domainC" / "oof"\n'
        "    oof_dir.mkdir(parents=True, exist_ok=True)\n",
        "ecg: oof_dir",
    )
    text = replace_once(
        text,
        "                auroc, inter = train_eval(\n"
        "                    c, Xt[tr_mask], Pt[tr_mask], yt[tr_mask],\n"
        "                    Xt[te_mask], Pt[te_mask], yt[te_mask],\n"
        "                    args.epochs, args.batch_size, args.lr, device,\n"
        "                )\n"
        "                per_cond[c].append(auroc)\n",
        "                auroc, inter, prob = train_eval(\n"
        "                    c, Xt[tr_mask], Pt[tr_mask], yt[tr_mask],\n"
        "                    Xt[te_mask], Pt[te_mask], yt[te_mask],\n"
        "                    args.epochs, args.batch_size, args.lr, device,\n"
        "                )\n"
        "                per_cond[c].append(auroc)\n"
        f"                {MARK}\n"
        "                assert len(prob) == int(te_mask.sum()), (len(prob), int(te_mask.sum()))\n"
        "                pd.DataFrame({\n"
        '                    "record_id": keys.index.to_numpy()[te_mask].astype(str),\n'
        '                    "patient_id": keys["patient_id"].to_numpy()[te_mask].astype(int).astype(str),\n'
        '                    "target": target, "condition": c, "fold": int(gf),\n'
        '                    "label": y[te_mask].astype(int), "prob": prob.astype(float),\n'
        '                }).to_csv(oof_dir / f"oof_{c}_{target}_fold{gf}.csv", index=False)\n',
        "ecg: main loop",
    )
    return text


def patch_glioma(text: str) -> str:
    text = replace_once(
        text,
        "    return float(auroc), inter\n",
        "    return float(auroc), inter, prob, ytrue\n",
        "glioma: train_eval return",
    )
    text = replace_once(
        text,
        "    results, regimes, interactions = {}, {}, {}\n",
        "    results, regimes, interactions = {}, {}, {}\n"
        f"    {MARK}\n"
        '    oof_dir = glioma_root / "derived" / "domainB" / "oof"\n'
        "    oof_dir.mkdir(parents=True, exist_ok=True)\n",
        "glioma: oof_dir",
    )
    text = replace_once(
        text,
        "                auroc, inter = train_eval(\n"
        "                    c, tr_ds, te_ds, args.edim, pdim,\n"
        "                    args.epochs, args.batch_size, args.lr, device,\n"
        "                )\n"
        "                per_cond[c].append(auroc)\n",
        "                auroc, inter, prob, ytrue = train_eval(\n"
        "                    c, tr_ds, te_ds, args.edim, pdim,\n"
        "                    args.epochs, args.batch_size, args.lr, device,\n"
        "                )\n"
        "                per_cond[c].append(auroc)\n"
        f"                {MARK}\n"
        "                assert len(prob) == int(te_mask.sum()), (len(prob), int(te_mask.sum()))\n"
        "                _subj = subjects.to_numpy()[te_mask].astype(str)\n"
        "                pd.DataFrame({\n"
        '                    "record_id": _subj, "patient_id": _subj,\n'
        '                    "target": target, "condition": c, "fold": int(gf),\n'
        '                    "label": ytrue.astype(int), "prob": prob.astype(float),\n'
        '                }).to_csv(oof_dir / f"oof_{c}_{target}_fold{gf}.csv", index=False)\n',
        "glioma: main loop",
    )
    return text


def apply(path: Path, fn, check: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        print(f"{path.name}: already patched, nothing to do")
        return
    new = fn(text)
    if check:
        print(f"{path.name}: patch applies cleanly ({new.count(MARK)} insertion points)")
        return
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    path.write_text(new, encoding="utf-8")
    compile(new, str(path), "exec")  # syntax check
    print(f"{path.name}: patched (backup at {bak.name})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for path, fn in ((ECG, patch_ecg), (GLI, patch_glioma)):
        if not path.exists():
            sys.exit(f"missing {path}")
        apply(path, fn, args.check)


if __name__ == "__main__":
    main()
