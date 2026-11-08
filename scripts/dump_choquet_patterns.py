#!/usr/bin/env python3
"""Re-read the learned Choquet measure under the corrected ANORAK class names.

The tile classifier was trained from an overlay_index whose .mat-id -> class
table was wrong, so the six names it sorted alphabetically are a permutation of
the real ones. The correction was verified by a histopathologist on 4 April 2026
and is recorded in PATTERN_REMAP_4_apr_2026.py; paper 2 was regenerated from it,
paper 1 was not. Anything in paper 1 that names a pattern is therefore reading
the wrong channel, including the KRAS co-activation result.

Training is unaffected: permuting names does not change a six-way optimisation.
Only the reporting layer is wrong, which is why the numbers below are reusable
and only their labels move.

    python scripts/dump_choquet_patterns.py --results-root /path/to/results
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Names the benchmark wrote, in model-output order (WRONG_ID2LABEL).
PATTERN_NAMES = ["acinar", "lepidic", "micropapillary",
                 "mucinous", "papillary", "solid"]

# PATTERN_REMAP_4_apr_2026.NAME_REMAP, verbatim.
NAME_REMAP = {
    "acinar":         "micropapillary",
    "lepidic":        "cribriform",
    "micropapillary": "papillary",
    "mucinous":       "lepidic",
    "papillary":      "solid",
    "solid":          "acinar",
}

CONDITION = "proposed_fuzzy_choquet"


def fix(name: str) -> str:
    return NAME_REMAP.get(name.strip().lower(), name)


def fix_pair(key: str) -> str:
    """Interaction keys arrive as 'a x b'; rename both sides, keep them sorted."""
    for sep in ("\u00d7", "x", "*", "_"):
        if sep in key:
            parts = [p for p in key.split(sep) if p.strip()]
            if len(parts) == 2:
                a, b = sorted(fix(p) for p in parts)
                return f"{a} \u00d7 {b}"
    return fix(key)


def collect(root: Path, gene: str):
    paths = sorted(root.rglob(f"metrics_{CONDITION}_{gene}_fold*.json"))
    shap, inter = defaultdict(list), defaultdict(list)
    used = 0
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = d.get("best_val", d)
        if "fuzzy_shapley_values" not in best:
            continue
        used += 1
        for k, v in best.get("fuzzy_shapley_values", {}).items():
            shap[fix(k)].append(v)
        for k, v in best.get("fuzzy_interactions", {}).items():
            inter[fix_pair(k)].append(v)
    return used, len(paths), shap, inter


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--genes", nargs="+",
                    default=["KRAS", "TP53", "EGFR", "STK11", "KEAP1", "RBM10"])
    args = ap.parse_args()

    print("model-output index -> what the benchmark called it -> what it is")
    for i, old in enumerate(PATTERN_NAMES):
        print(f"  {i}   {old:<16} -> {fix(old)}")

    for gene in args.genes:
        used, found, shap, inter = collect(args.results_root, gene)
        print(f"\n{'=' * 72}\n{gene}   {used} of {found} metrics files carried a "
              f"fuzzy measure\n{'=' * 72}")
        if not used:
            print("  none found; is --results-root the directory with "
                  "metrics_*.json?")
            continue

        print("\n  Shapley value per pattern, corrected names")
        for name in sorted(shap, key=lambda k: -abs(mean(shap[k]))):
            print(f"    {name:<16}{mean(shap[name]):+.4f}")

        print("\n  Pair interactions, strongest first "
              "(positive = synergy, negative = redundancy)")
        for k in sorted(inter, key=lambda k: -abs(mean(inter[k])))[:8]:
            print(f"    {k:<34}{mean(inter[k]):+.4f}")

        if gene == "KRAS":
            lep_sol = next((k for k in inter
                            if set(k.split(" \u00d7 ")) == {"lepidic", "solid"}),
                           None)
            if lep_sol:
                v = mean(inter[lep_sol])
                verdict = "synergy, the proxy reading survives" if v > 0 else \
                          "REDUNDANCY: the invasive-mucinous proxy reading is gone"
                print(f"\n  the pair the paper rests on, lepidic x solid: "
                      f"{v:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
