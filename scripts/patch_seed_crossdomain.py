"""E9b: give the ECG and glioma runners a --seed flag and seed-tagged outputs.

Applies on top of patch_oof_crossdomain.py (it refuses to run if that patch
is absent). After this patch:

  * `--seed S` (default: the module SEED, 42) seeds torch and numpy;
  * out-of-fold CSVs go to derived/<domain>/oof/seed<S>/;
  * results, regimes and interactions are written as
    results_seed<S>.json, regimes_seed<S>.json,
    choquet_interactions_seed<S>.csv.

The seed-42 artefacts of the first OOF run are left where they are; move
them into place once (see DGX_PLAN E9b) so the hierarchical bootstrap
finds five seed directories.

Usage (from paper1_nature/):
    python patch_seed_crossdomain.py --check
    python patch_seed_crossdomain.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ECG = HERE / "run_ecg_domainC.py"
GLI = HERE / "glioma" / "run_glioma_domainB.py"

OOF_MARK = "# --- OOF patch (E9) ---"
MARK = "# --- seed patch (E9b) ---"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"[{label}] expected exactly 1 occurrence, found {n}:\n{old}")
    return text.replace(old, new)


def patch(text: str, root_var: str, domain: str, inter_cols: str) -> str:
    text = replace_once(
        text,
        '    ap.add_argument("--smoke", action="store_true"',
        f'    ap.add_argument("--seed", type=int, default=SEED)  {MARK}\n'
        '    ap.add_argument("--smoke", action="store_true"',
        f"{domain}: --seed arg",
    )
    text = replace_once(
        text,
        "    torch.manual_seed(SEED); np.random.seed(SEED)\n",
        "    torch.manual_seed(args.seed); np.random.seed(args.seed)\n"
        f'    print(f"seed: {{args.seed}}")\n',
        f"{domain}: seeding",
    )
    text = replace_once(
        text,
        f'    oof_dir = {root_var} / "derived" / "{domain}" / "oof"\n',
        f'    oof_dir = {root_var} / "derived" / "{domain}" / "oof" / f"seed{{args.seed}}"\n',
        f"{domain}: oof_dir",
    )
    text = replace_once(
        text,
        '    (out_dir / "results.json").write_text(json.dumps(results, indent=2))\n'
        '    (out_dir / "regimes.json").write_text(json.dumps(regimes, indent=2))\n',
        f'    _tag = f"_seed{{args.seed}}"  {MARK}\n'
        '    (out_dir / f"results{_tag}.json").write_text(json.dumps(results, indent=2))\n'
        '    (out_dir / f"regimes{_tag}.json").write_text(json.dumps(regimes, indent=2))\n',
        f"{domain}: results files",
    )
    text = replace_once(
        text,
        f'        pd.DataFrame(rows).to_csv(out_dir / "choquet_interactions.csv", index=False)\n',
        f'        pd.DataFrame(rows).to_csv(out_dir / f"choquet_interactions{{_tag}}.csv", index=False)\n',
        f"{domain}: interactions file",
    )
    return text


def apply(path: Path, root_var: str, domain: str, inter_cols: str, check: bool) -> None:
    text = path.read_text(encoding="utf-8")
    if OOF_MARK not in text:
        sys.exit(f"{path.name}: apply patch_oof_crossdomain.py first")
    if MARK in text:
        print(f"{path.name}: already patched, nothing to do")
        return
    new = patch(text, root_var, domain, inter_cols)
    if check:
        print(f"{path.name}: patch applies cleanly")
        return
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak_e9b"))
    path.write_text(new, encoding="utf-8")
    compile(new, str(path), "exec")
    print(f"{path.name}: patched (backup {path.name}.bak_e9b)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    apply(ECG, "ecg_root", "domainC", "stmt", args.check)
    apply(GLI, "glioma_root", "domainB", "feat", args.check)


if __name__ == "__main__":
    main()
