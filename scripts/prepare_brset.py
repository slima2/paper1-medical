"""Prepare the BRSET retina arm: labels, splits, leakage check, power check.

Domain D of the framework. The source pair is the retinal specialist's
structured reading (channel P) against a learned encoder over the fundus
photograph (channel E); the target is patient sex.

Sex is chosen because it is the cleanest regime-(iii) case available in any of
our cohorts: no term in the ophthalmological vocabulary names it, yet the
published BRSET benchmark recovers it at AUC 0.91 with a plain ConvNext V2.
That combination -- an empty vocabulary-side atom against a large encoder-side
one -- is exactly what regime (iii) asserts, and at this cohort size it is the
first such assertion in this work that can be adjudicated rather than read
directionally.

This script does not train anything. It emits the label table and the split
keys the benchmark runner consumes, and it runs the two checks that must pass
before the arm is worth training at all:

  1. Leakage. Regime (iii) requires the vocabulary to be uninformative about
     the target. We verify that empirically, per feature and jointly, rather
     than asserting it from the fact that no feature is named "sex".
  2. Power. We state the minimum detectable effect before seeing any result,
     so that an equivalence read cannot be manufactured after the fact.

Usage:
    python scripts/prepare_brset.py --brset-dir /path/to/brset [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd

# --- the retinal specialist's vocabulary, channel P ------------------------
# Three anatomical criteria and thirteen pathology labels, as defined in the
# BRSET data descriptor. Quality-control fields are deliberately excluded:
# they describe the photograph rather than the retina, so they are confounds
# rather than vocabulary, and folding them in would let the model read the
# camera instead of the eye.
ANATOMY = ["optic_disc", "vessels", "macula"]
PATHOLOGY = [
    "diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd",
    "vascular_occlusion", "hypertensive_retinopathy", "drusens",
    "hemorrhage", "retinal_detachment", "myopic_fundus",
    "increased_cup_disc", "other",
]
QUALITY = ["focus", "illuminaton", "illumination", "image_field", "artifacts"]

TARGET = "patient_sex"          # 1 = male, 2 = female
PATIENT = "patient_id"
IMAGE = "image_id"
SITE = "camera"                 # Canon CR2 / Nikon NF5050, the site analogue
EYE = "exam_eye"                # 1 = right, 2 = left

Z_ALPHA = 1.959963985
Z_POWER = 0.8416212336
K_MDE = Z_ALPHA + Z_POWER
F_TRANSPORT = 0.66              # median paired/analytic SE ratio, LUAD arm
EQUIV_MARGIN = 0.02             # the margin every regime read in this work uses


def resolve(df: pd.DataFrame, wanted: list[str]) -> tuple[list[str], list[str]]:
    """Match requested columns case-insensitively; report what is missing."""
    lower = {c.lower(): c for c in df.columns}
    found = [lower[w.lower()] for w in wanted if w.lower() in lower]
    missing = [w for w in wanted if w.lower() not in lower]
    return found, missing


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. No sklearn dependency."""
    pos, neg = labels == 1, labels == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within ties so a constant feature scores exactly 0.5
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brset-dir", required=True,
                    help="directory holding labels.csv")
    ap.add_argument("--out", default="data_brset")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the shuffled-sex negative control")
    args = ap.parse_args()

    src = pathlib.Path(args.brset_dir)
    csv = src / "labels.csv"
    if not csv.exists():
        hits = list(src.rglob("labels.csv"))
        if not hits:
            print(f"ERROR: no encuentro labels.csv bajo {src}", file=sys.stderr)
            return 1
        csv = hits[0]
    df = pd.read_csv(csv)
    print(f"Leido {csv}  ->  {len(df)} filas, {df.shape[1]} columnas\n")

    for col in (TARGET, PATIENT, IMAGE):
        if col not in df.columns:
            print(f"ERROR: falta la columna obligatoria '{col}'. "
                  f"Columnas presentes: {list(df.columns)}", file=sys.stderr)
            return 1

    vocab, missing = resolve(df, ANATOMY + PATHOLOGY)
    if missing:
        print(f"AVISO: {len(missing)} columnas de vocabulario no aparecen "
              f"con el nombre esperado: {missing}")
        print("       Revisa el diccionario del dataset antes de seguir.\n")
    qfound, _ = resolve(df, QUALITY)
    print(f"Canal P: {len(vocab)} rasgos  {vocab}")
    print(f"Excluidos por ser calidad de imagen, no de retina: {qfound}\n")

    # --- target ----------------------------------------------------------
    # 1 = male, 2 = female in the source encoding; we predict male.
    df = df[df[TARGET].isin([1, 2])].copy()
    df["SEXM"] = (df[TARGET] == 1).astype(int)

    n_img = len(df)
    pats = df.groupby(PATIENT)["SEXM"].agg(["first", "nunique"])
    if (pats["nunique"] > 1).any():
        bad = int((pats["nunique"] > 1).sum())
        print(f"ERROR: {bad} pacientes tienen sexo inconsistente entre "
              f"imagenes. Abortando.", file=sys.stderr)
        return 1
    n_pat = len(pats)
    n_pos = int(pats["first"].sum())
    n_neg = n_pat - n_pos
    print(f"Cohorte: {n_img} imagenes de {n_pat} pacientes "
          f"({n_img / n_pat:.2f} por paciente)")
    print(f"Objetivo SEXM: {n_pos} varones / {n_neg} mujeres "
          f"({100 * n_pos / n_pat:.1f}% positivos)\n")

    # --- negative control, mirroring TP53SHUF in the LUAD arm -------------
    # Permuted at the patient level so the positive count is preserved exactly
    # and the permutation cannot be undone by pooling a patient's two eyes.
    rng = np.random.default_rng(args.seed)
    shuffled = pats["first"].to_numpy().copy()
    rng.shuffle(shuffled)
    df["SEXSHUF"] = df[PATIENT].map(
        dict(zip(pats.index, shuffled))).astype(int)
    assert df.groupby(PATIENT)["SEXSHUF"].first().sum() == n_pos

    # --- check 1: does the vocabulary leak the target? --------------------
    print("=" * 74)
    print("CHECK 1  Fuga: ¿el vocabulario predice el sexo?")
    print("=" * 74)
    print("Regimen (iii) exige que los atomos del lado del vocabulario se")
    print("anulen. Lo verificamos, no lo suponemos.\n")
    y = df["SEXM"].to_numpy()
    rows = []
    for c in vocab:
        v = pd.to_numeric(df[c], errors="coerce").fillna(-1).to_numpy(float)
        a = auroc(v, y)
        rows.append((c, a, abs(a - 0.5)))
    rows.sort(key=lambda r: -r[2])
    for c, a, d in rows:
        flag = "  <-- REVISAR" if d > 0.10 else ""
        print(f"   {c:28s} AUROC {a:.3f}   |desv| {d:.3f}{flag}")
    worst = rows[0]
    print(f"\n   Rasgo mas informativo: {worst[0]} en {worst[1]:.3f}.")
    if worst[2] > 0.10:
        print("   AVISO: algun rasgo se aparta del azar mas de 0.10. El par")
        print("   no es limpio y hay que justificarlo o excluir ese rasgo.")
    else:
        print("   Ningun rasgo aislado se aparta del azar mas de 0.10, asi que")
        print("   el par objetivo/vocabulario es limpio en el mismo sentido")
        print("   que sexo y edad lo son frente a los enunciados SCP-ECG.")
    print("   La version conjunta de este check es la propia condicion B3.\n")

    # --- check 2: power, stated before any training -----------------------
    print("=" * 74)
    print("CHECK 2  Potencia, declarada antes de entrenar")
    print("=" * 74)
    for assumed in (0.85, 0.91, 0.95):
        se = hanley_mcneil_se(assumed, n_pos, n_neg)
        mde = K_MDE * F_TRANSPORT * se
        half = Z_ALPHA * F_TRANSPORT * se
        ok = "SI" if half < EQUIV_MARGIN else "NO"
        print(f"   AUROC supuesto {assumed:.2f}  ->  MDE {mde:.4f}   "
              f"semianchura IC95 {half:.4f}   "
              f"¿cabe en +-{EQUIV_MARGIN}? {ok}")
    print(f"\n   Referencia: las ganancias de agregador publicadas que")
    print(f"   auditamos van de 0.008 a 0.046 (tab:literature_power_audit).")
    print(f"   Referencia LUAD: el mismo calculo da 0.04-0.11 por gen.\n")

    # --- emit --------------------------------------------------------------
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = [IMAGE, PATIENT, "SEXM", "SEXSHUF"] + vocab
    for extra in (SITE, EYE):
        if extra in df.columns:
            keep.insert(2, extra)
    df[keep].to_csv(out / "labels.csv", index=False)
    print(f"Escrito {out / 'labels.csv'}  ({len(keep)} columnas)")

    if SITE in df.columns:
        counts = df.groupby(SITE)[PATIENT].nunique()
        print(f"\nAnalogo del protocolo por sitio, columna '{SITE}':")
        for k, v in counts.items():
            print(f"   {k:24s} {v:5d} pacientes "
                  f"({100 * v / n_pat:.1f}%)")
        # A site-preserved split is only well defined if the grouping key
        # partitions patients. A patient photographed on both cameras would
        # otherwise land on both sides of the split, which is the same
        # leak the site-preserved protocol exists to prevent.
        spanning = df.groupby(PATIENT)[SITE].nunique()
        n_span = int((spanning > 1).sum())
        if n_span:
            print(f"\n   AVISO: {n_span} pacientes ({100 * n_span / n_pat:.1f}%)"
                  f" tienen imagenes de mas de una camara, asi que '{SITE}'")
            print("   NO particiona pacientes y no sirve tal cual como clave")
            print("   de sitio. Opciones: asignar a cada paciente su camara")
            print("   mayoritaria, o descartar del brazo por sitio a los que")
            print("   abarcan ambas. Decidir antes de entrenar y declararlo.")
        else:
            print("\n   Cada paciente aparece bajo una sola camara, asi que la")
            print("   clave particiona limpiamente y sirve para el protocolo")
            print("   preservado por sitio, igual que el sitio TCGA en LUAD.")
    else:
        print(f"\nAVISO: no hay columna '{SITE}'. Sin ella no hay analogo del")
        print("       protocolo por sitio y el brazo pierde la paridad con LUAD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
