#!/usr/bin/env python3
"""Sensitivity of the verdict and regime reads to the two free thresholds.

Two parameters carry every categorical claim in the paper and neither is
estimated from the data: the Conclusive cut on the mean fold AUROC
(eq:conclusive_verdict, default 0.70) and the equivalence margin that gates
regimes (i) and (iii) (default +-0.02). This script

  1. derives the margin from the paper's own literature power audit;
  2. re-runs the Conclusive verdict over a grid of cuts;
  3. re-runs the equivalence gate over a grid of margins, on the two contrasts
     the adjudication actually gates on; and
  4. reports the cohort size at which each gating contrast would become
     certifiable at the default margin.

The equivalence gate is PI-B2 and FC-B2, per
bootstrap_luad_contrasts.py::adjudicate_multiseed:

    if expected in {"(i)", "(iii)"}:
        needed = by_name.loc[["PI-B2", "FC-B2"]]
        if all(interval_inside(row, margin) for _, row in needed.iterrows()):
            return "equivalent", ...

FC-PI decides (ii-b) and B3-B2 measures what the vocabulary omits; neither
gates equivalence. Note that 05_methods_CM.tex claimed "all four intervals",
which misdescribed the analysis; that has been corrected in the manuscript.

Usage:  python scripts/threshold_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_PATIENTS = 444
CONTRASTS_CSV = Path(__file__).resolve().parent.parent / "data" / "luad_hierarchical_contrasts.csv"

# The two contrasts the adjudication gates regimes (i) and (iii) on.
EQUIVALENCE_GATE = ["PI-B2", "FC-B2"]

# --- tab_literature_power_audit.tex, the Delta column -----------------------
PUBLISHED_GAINS = {
    "ALK": 0.009,
    "BRAF": 0.041,
    "EGFR": 0.015,
    "ERBB2": 0.046,
    "KRAS": 0.017,
    "METex14": 0.023,
    "architecture (vs. vanilla decoder)": 0.014,
    "tissue stratification (vs. none)": 0.008,
}

# --- tab_mutation_luad.tex, pooled OOF AUROC per condition ------------------
AUROC = {
    "patient": {
        #          B2     B3     PI     FC
        "TP53":  (0.661, 0.579, 0.658, 0.646),
        "EGFR":  (0.582, 0.480, 0.580, 0.567),
        "STK11": (0.591, 0.499, 0.578, 0.608),
        "KEAP1": (0.488, 0.485, 0.446, 0.460),
        "KRAS":  (0.517, 0.486, 0.516, 0.506),
        "RBM10": (0.536, 0.490, 0.552, 0.528),
    },
    "site": {
        "TP53":  (0.640, 0.561, 0.630, 0.636),
        "EGFR":  (0.584, 0.485, 0.576, 0.557),
        "STK11": (0.605, 0.523, 0.593, 0.597),
        "KEAP1": (0.482, 0.488, 0.472, 0.471),
        "KRAS":  (0.507, 0.475, 0.502, 0.501),
        "RBM10": (0.525, 0.517, 0.512, 0.511),
    },
}

GENES = ["TP53", "EGFR", "STK11", "KEAP1", "KRAS", "RBM10"]
PROTOCOLS = ["patient", "site"]
CUT_GRID = [0.60, 0.65, 0.70, 0.75]
MARGIN_GRID = [0.01, 0.02, 0.03, 0.05]
DEFAULT_MARGIN = 0.02


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_contrasts() -> pd.DataFrame:
    df = pd.read_csv(CONTRASTS_CSV)
    # Half-width of the hierarchical bootstrap interval, and the largest
    # absolute bound, which is what the equivalence test compares to delta.
    df["half_width"] = (df["ci_hi"] - df["ci_lo"]) / 2.0
    df["max_abs_bound"] = df[["ci_lo", "ci_hi"]].abs().max(axis=1)
    return df


def derive_margin() -> None:
    rule("1. Where the +-0.02 equivalence margin comes from")
    gains = np.array(list(PUBLISHED_GAINS.values()))
    for name, g in sorted(PUBLISHED_GAINS.items(), key=lambda kv: kv[1]):
        print(f"    {g:.3f}  {name}")
    print(f"\n  n={gains.size}  range {gains.min():.3f}-{gains.max():.3f}"
          f"  median {np.median(gains):.4f}  mean {gains.mean():.4f}")
    print("\n  The mean rounds to 0.02, so the margin is the effect size this")
    print("  literature typically claims for an aggregator change. Declaring")
    print("  equivalence at it means declaring that a fusion arm does not")
    print("  deliver what the field says such a change delivers.")


def verdict_sensitivity() -> None:
    rule("2. Conclusive verdict over a grid of cuts (eq:conclusive_verdict)")
    header = f"  {'gene':<7}{'best(pat)':>10}{'best(site)':>11}   " + "".join(
        f"{c:>14.2f}" for c in CUT_GRID
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    totals = {c: 0 for c in CUT_GRID}
    for gene in GENES:
        bp = max(AUROC["patient"][gene])
        bs = max(AUROC["site"][gene])
        cells = []
        for cut in CUT_GRID:
            hit = [n for n, v in (("pat", bp), ("site", bs)) if v >= cut]
            totals[cut] += 1 if hit else 0
            cells.append(f"{('CONCL:' + '+'.join(hit)) if hit else 'Inconcl.':>14}")
        print(f"  {gene:<7}{bp:>10.3f}{bs:>11.3f}   " + "".join(cells))
    print(f"  {'TOTAL':<7}{'':>10}{'':>11}   " + "".join(
        f"{totals[c]:>14d}" for c in CUT_GRID
    ))
    print("\n  No target is Conclusive at 0.70 or 0.75. At 0.65 TP53 flips on the")
    print("  patient arm alone; only at 0.60 do TP53 and STK11 clear under both.")
    print("  The central negative result is not an artefact of the convention.")


def equivalence_grid(df: pd.DataFrame) -> None:
    rule("3. Equivalence gate (PI-B2 AND FC-B2) over a grid of margins")
    print("  A cell establishes equivalence at margin d iff BOTH gating intervals")
    print("  satisfy max(|ci_lo|,|ci_hi|) <= d. 'EQ' = established.\n")
    gate = df[df.contrast.isin(EQUIVALENCE_GATE)]
    header = (
        f"  {'protocol':<9}{'gene':<7}{'max|PI-B2|':>12}{'max|FC-B2|':>12}"
        f"{'binding':>10}   " + "".join(f"{m:>8.2f}" for m in MARGIN_GRID)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    totals = {m: 0 for m in MARGIN_GRID}
    for protocol in PROTOCOLS:
        for gene in GENES:
            rows = gate[(gate.protocol == protocol) & (gate.gene == gene)]
            bounds = rows.set_index("contrast")["max_abs_bound"]
            pi, fc = bounds["PI-B2"], bounds["FC-B2"]
            binding = max(pi, fc)
            cells = []
            for m in MARGIN_GRID:
                ok = binding <= m
                totals[m] += int(ok)
                cells.append(f"{'EQ' if ok else '.':>8}")
            print(
                f"  {protocol:<9}{gene:<7}{pi:>12.4f}{fc:>12.4f}"
                f"{binding:>10.4f}   " + "".join(cells)
            )
    print(f"  {'TOTAL':<16}{'':>12}{'':>12}{'':>10}   " + "".join(
        f"{totals[m]:>8d}" for m in MARGIN_GRID
    ))
    print("\n  'binding' is the larger of the two, i.e. the smallest margin at which")
    print("  that cell would be declared equivalent.")
    changed = [m for m in MARGIN_GRID if totals[m] != totals[DEFAULT_MARGIN]]
    if changed:
        print(f"\n  *** The reads ARE sensitive to the margin: {changed} give a")
        print("  different count than the default 0.02. Report this honestly.")
        for m in changed:
            flips = []
            for protocol in PROTOCOLS:
                for gene in GENES:
                    rows = gate[(gate.protocol == protocol) & (gate.gene == gene)]
                    b = rows["max_abs_bound"].max()
                    if b <= m:
                        flips.append(f"{gene}/{protocol} ({b:.4f})")
            print(f"    d={m:.2f}: {', '.join(flips) if flips else 'none'}")
    else:
        print("\n  The reads are insensitive to the margin across the whole grid.")


def declarability(df: pd.DataFrame) -> None:
    rule("4. Cohort size at which the default margin becomes certifiable")
    print("  Interval half-width shrinks as 1/sqrt(n) while the point estimate")
    print("  stays put, so equivalence at d=0.02 needs")
    print("      |point| + half_width * sqrt(444/n) <= 0.02.")
    print("  'never' means |point| already exceeds 0.02.\n")
    gate = df[df.contrast.isin(EQUIVALENCE_GATE)]
    header = (
        f"  {'protocol':<9}{'gene':<7}{'contrast':<8}{'point':>9}"
        f"{'half-w':>9}{'n needed':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    worst: dict[tuple[str, str], float | str] = {}
    for protocol in PROTOCOLS:
        for gene in GENES:
            reqs = []
            for contrast in EQUIVALENCE_GATE:
                row = gate[
                    (gate.protocol == protocol)
                    & (gate.gene == gene)
                    & (gate.contrast == contrast)
                ].iloc[0]
                point, half = abs(row["point"]), row["half_width"]
                slack = DEFAULT_MARGIN - point
                if slack <= 0:
                    n_req: float | str = "never"
                    shown = "never"
                else:
                    n_req = N_PATIENTS * (half / slack) ** 2
                    shown = f"{n_req:,.0f}"
                reqs.append(n_req)
                print(
                    f"  {protocol:<9}{gene:<7}{contrast:<8}{row['point']:>9.4f}"
                    f"{half:>9.4f}{shown:>12}"
                )
            worst[(protocol, gene)] = (
                "never" if any(r == "never" for r in reqs) else max(reqs)
            )
    print("\n  Per cell, the binding requirement (both contrasts must certify):")
    for (protocol, gene), v in worst.items():
        shown = v if isinstance(v, str) else f"{v:,.0f}"
        print(f"    {protocol:<9}{gene:<7}{shown:>12}")
    finite = [v for v in worst.values() if not isinstance(v, str)]
    if finite:
        print(f"\n  Smallest cohort that would certify any cell: {min(finite):,.0f}")
        print(f"  Largest among certifiable cells:               {max(finite):,.0f}")


def gate_point_estimates() -> None:
    rule("5. Reference: gate point estimates against the default margin")
    header = (
        f"  {'protocol':<9}{'gene':<7}{'PI-B2':>10}{'FC-B2':>10}"
        f"{'| FC-PI':>12}{'B3-B2':>10}{'  point read':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for protocol in PROTOCOLS:
        for gene in GENES:
            b2, b3, pi, fc = AUROC[protocol][gene]
            g = [pi - b2, fc - b2]
            cells = "".join(
                f"{f'{v:+.3f}' + ('**' if abs(v) > DEFAULT_MARGIN else '  '):>10}"
                for v in g
            )
            cells += f"{fc - pi:>12.3f}{b3 - b2:>10.3f}"
            read = "consistent" if all(abs(v) <= DEFAULT_MARGIN for v in g) else "ruled out"
            print(f"  {protocol:<9}{gene:<7}{cells}{read:>14}")
    print("\n  'consistent' means the point estimates agree with equivalence, so")
    print("  certification turns entirely on interval width (section 4).")


if __name__ == "__main__":
    contrasts = load_contrasts()
    derive_margin()
    verdict_sensitivity()
    equivalence_grid(contrasts)
    declarability(contrasts)
    gate_point_estimates()
