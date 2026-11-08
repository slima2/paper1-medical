"""Recompute the control AUROCs from the finished runs and check the paper.

The manuscript currently cites four numbers from the control arm:

    permutation null   B2 0.515   B3 0.482
    positive control   sex 0.588
    same-run reference TP53 0.647  ("fixes the ceiling at 0.65")

and one comparison that depends on the third to within six thousandths:
Results states that EGFR, at 0.582, sits marginally BELOW the sex control.
If the completed run moves the sex control below 0.582 that sentence inverts,
so it is recomputed here rather than trusted.

Pooled out-of-fold AUROC, aggregated to the patient before scoring, exactly as
the main arm reports it. Run from the directory holding results_controls/.

Usage:
    python verify_controls.py [--root results_controls]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

# What the manuscript says today, so the script can flag its own disagreement.
CITED = {
    ("baseline2", "TP53SHUF"): (0.515, "nulo permutado B2"),
    ("baseline3", "TP53SHUF"): (0.482, "nulo permutado B3"),
    ("baseline2", "SEXF"): (0.588, "control positivo de sexo"),
    ("baseline2", "TP53"): (0.647, "TP53 en la misma corrida"),
}
EGFR_B2 = 0.582   # Table 1, patient-preserved; the comparison at risk

ID_HINTS = ["patient", "case", "submitter", "bcr", "slide", "id"]
LAB_HINTS = ["label", "y_true", "true", "target", "gt", "y"]
PRB_HINTS = ["prob", "score", "y_pred", "pred", "logit", "p1", "p_"]


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = label == 1, label == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = score.argsort(kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def pick(cols: list[str], hints: list[str], exclude: set[str]) -> str | None:
    low = {c: c.lower() for c in cols}
    for h in hints:
        for c in cols:
            if c in exclude:
                continue
            if low[c] == h:
                return c
    for h in hints:
        for c in cols:
            if c in exclude:
                continue
            if h in low[c]:
                return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_controls")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    files = sorted(root.glob("*/oof/oof_*.csv"))
    if not files:
        print(f"ERROR: no hay oof_*.csv bajo {root}", file=sys.stderr)
        return 1
    print(f"{len(files)} ficheros OOF bajo {root}\n")

    head = pd.read_csv(files[0], nrows=5)
    cols = list(head.columns)
    lab = pick(cols, LAB_HINTS, set())
    prb = pick(cols, PRB_HINTS, {lab} if lab else set())
    pid = pick(cols, ID_HINTS, {c for c in (lab, prb) if c})
    print(f"Esquema detectado en {files[0].name}:")
    print(f"   columnas   {cols}")
    print(f"   id         {pid}")
    print(f"   etiqueta   {lab}")
    print(f"   puntuacion {prb}")
    if not (lab and prb):
        print("\nERROR: no identifico etiqueta y puntuacion. Pasa las columnas "
              "correctas editando LAB_HINTS/PRB_HINTS.", file=sys.stderr)
        return 1
    print()

    pat = re.compile(r"oof_(?P<cond>.+?)_(?P<target>[A-Z0-9]+)_fold(?P<fold>\d+)\.csv$")
    rows = []
    for f in files:
        m = pat.match(f.name)
        if not m:
            print(f"   (ignorado, nombre inesperado: {f.name})")
            continue
        run = f.parent.parent.name                      # e.g. patient_seed142
        proto, seed = run.split("_seed")
        d = pd.read_csv(f)
        d = d[[c for c in (pid, lab, prb) if c]].copy()
        d.columns = (["id", "y", "p"] if pid else ["y", "p"])
        rows.append({"proto": proto, "seed": int(seed),
                     "cond": m.group("cond").split("_")[0],
                     "target": m.group("target"), "df": d})

    if not rows:
        return 1

    out = []
    key = ["proto", "seed", "cond", "target"]
    index = {}
    for r in rows:
        index.setdefault(tuple(r[k] for k in key), []).append(r["df"])
    for k, parts in sorted(index.items()):
        d = pd.concat(parts, ignore_index=True)
        level = "portaobjetos"
        if "id" in d.columns:
            # aggregate to the patient before scoring, as the main arm does
            d = d.groupby("id", as_index=False).agg({"y": "max", "p": "mean"})
            level = "paciente"
        out.append({"proto": k[0], "seed": k[1], "cond": k[2], "target": k[3],
                    "n": len(d), "n_pos": int((d["y"] == 1).sum()),
                    "auroc": auroc(d["p"].to_numpy(float),
                                   d["y"].to_numpy(int)),
                    "nivel": level})
    res = pd.DataFrame(out)

    print("=" * 78)
    print("AUROC OOF agrupado, por semilla")
    print("=" * 78)
    for (proto, cond, target), g in res.groupby(["proto", "cond", "target"]):
        per = "  ".join(f"s{int(r.seed)}={r.auroc:.3f}"
                        for r in g.sort_values("seed").itertuples())
        print(f"  {proto:8s} {cond:10s} {target:9s} "
              f"n={g.n.iloc[0]:4d} n+={g.n_pos.iloc[0]:4d}  {per}")

    print()
    print("=" * 78)
    print("Media +- DE entre semillas, y contraste con lo que dice el paper")
    print("=" * 78)
    agg = (res.groupby(["proto", "cond", "target"])["auroc"]
              .agg(["mean", "std", "count"]).reset_index())
    for r in agg.sort_values(["proto", "target", "cond"]).itertuples():
        cited = CITED.get((r.cond, r.target))
        tag = ""
        if cited and r.proto == "patient":
            delta = r.mean - cited[0]
            mark = "OK" if abs(delta) < 0.005 else "CAMBIA"
            tag = (f"   paper dice {cited[0]:.3f} ({cited[1]}) "
                   f"-> {delta:+.3f}  [{mark}]")
        sd = f"{r.std:.3f}" if r.count > 1 else "  -  "
        print(f"  {r.proto:8s} {r.cond:10s} {r.target:9s} "
              f"{r.mean:.3f} +- {sd}  (n={r.count} semillas){tag}")

    # --- the comparison the manuscript depends on -------------------------
    sexr = agg[(agg.proto == "patient") & (agg.cond == "baseline2")
               & (agg.target == "SEXF")]
    print()
    print("=" * 78)
    print("La frase en riesgo")
    print("=" * 78)
    if sexr.empty:
        print("  No encuentro SEXF/baseline2 bajo el protocolo por paciente.")
    else:
        sex = float(sexr["mean"].iloc[0])
        print(f"  Control de sexo (B2, paciente) = {sex:.3f}")
        print(f"  EGFR B2 (Tabla 1)              = {EGFR_B2:.3f}")
        print(f"  EGFR - control                 = {EGFR_B2 - sex:+.3f}")
        if EGFR_B2 < sex:
            print("\n  Resultados dice que EGFR queda POR DEBAJO del control.")
            print("  Se mantiene. No hay que tocar el texto.")
        else:
            print("\n  ATENCION: EGFR queda POR ENCIMA del control.")
            print("  Hay que invertir la frase de Resultados otra vez, y")
            print("  revisar el 0.59 del resumen y de la Discusion.")
    res.to_csv("controls_recomputed.csv", index=False)
    print("\nEscrito controls_recomputed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
