"""Census and prep for the tumour-vs-normal positive control.

Why this runs before any GPU time. The main arm uses TP53 both as a
measurement and as the evidence that the pipeline works, which is circular.
A tumour-vs-normal control on the same frozen encoder and the same ABMIL head
breaks the circularity, because the expected value is near ceiling and any
result far below it indicts the pipeline rather than the cohort.

Two things can sink it, and neither is visible from the manuscript:

  1. The normal slides were dropped during cohort correction. If features were
     extracted only for the 458 tumour slides, there is nothing to train on and
     the control costs a re-extraction rather than an afternoon.

  2. TCGA matched normals come from the SAME patients as the tumours. The rest
     of the pipeline aggregates slides to the patient by mean probability
     before scoring. For this target that is fatal: a patient contributing both
     a tumour and a normal slide has no single label. This control must be
     scored at the slide level, or restricted to patients contributing one
     type. The script measures how much each choice costs.

Labels come from the TCGA barcode itself (field 4, the sample type), not from
a metadata file, so they cannot drift from the slide identity.

Usage, on the DGX:
    python prepare_tumor_control.py \
        --inventory "/home/rapids/notebooks/slima/TGCA MAF/luad_slide_ids_available.txt" \
        --data-dir "$PWD/data_control" \
        --out "$PWD/data_tumor_control"
"""

from __future__ import annotations

import argparse
import collections
import math
import pathlib
import re
import sys

# TCGA sample type codes (barcode field 4, first two digits).
TUMOUR = {"01": "primary solid tumour", "02": "recurrent solid tumour",
          "05": "additional new primary", "06": "metastatic",
          "07": "additional metastatic"}
NORMAL = {"10": "blood derived normal", "11": "solid tissue normal",
          "12": "buccal normal", "14": "bone marrow normal"}

BARCODE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z]?")
FEAT_EXT = (".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5")

Z975, Z80 = 1.959963985, 0.841621234


def hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                      + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))


def parse(slide_id: str) -> tuple[str, str] | None:
    m = BARCODE.search(slide_id)
    return (m.group(1), m.group(2)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True,
                    help="text file, one slide id per line, BEFORE the "
                         "tumour-only filter")
    ap.add_argument("--data-dir", required=True,
                    help="directory holding the extracted features")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inv = pathlib.Path(args.inventory)
    if not inv.exists():
        print(f"ERROR: no existe {inv}", file=sys.stderr)
        return 1
    slides = [ln.strip() for ln in inv.read_text().splitlines() if ln.strip()]
    print(f"Inventario: {len(slides)} portaobjetos en {inv.name}\n")

    # --- 1. sample-type census -------------------------------------------
    by_type: dict[str, list[str]] = collections.defaultdict(list)
    unparsed = []
    for s in slides:
        p = parse(s)
        (by_type[p[1]] if p else unparsed).append(s)
    print("Censo por tipo de muestra (campo 4 del codigo de barras TCGA):")
    for code in sorted(by_type):
        name = TUMOUR.get(code) or NORMAL.get(code) or "desconocido"
        arm = "tumor" if code in TUMOUR else (
            "normal" if code in NORMAL else "?")
        print(f"   {code}  {name:24s} {arm:7s} {len(by_type[code]):5d}")
    if unparsed:
        print(f"   sin codigo de barras reconocible: {len(unparsed)}")
        for s in unparsed[:5]:
            print(f"      {s}")

    tum = [s for c in TUMOUR for s in by_type.get(c, [])]
    nor = [s for c in NORMAL for s in by_type.get(c, [])]
    print(f"\n   tumor  {len(tum)}\n   normal {len(nor)}")
    if not nor:
        print("\nPARADA: el inventario no contiene normales, asi que este\n"
              "fichero es ya la lista filtrada. Apunta --inventory al\n"
              "inventario previo al filtro (el de 687 imagenes) o al\n"
              "manifiesto del GDC.", file=sys.stderr)
        return 1

    # --- 2. do the features exist? ---------------------------------------
    data = pathlib.Path(args.data_dir)
    have: set[str] = set()
    if data.exists():
        for f in data.rglob("*"):
            if f.suffix.lower() in FEAT_EXT:
                p = parse(f.stem)
                if p:
                    have.add(f.stem)
    print(f"\nFicheros de caracteristicas hallados bajo {data}: {len(have)}")

    def covered(ids: list[str]) -> list[str]:
        idx = {h: h for h in have}
        out = []
        for s in ids:
            if s in idx:
                out.append(s)
            else:
                hit = [h for h in have if s in h or h in s]
                if hit:
                    out.append(hit[0])
        return out

    tum_ok, nor_ok = covered(tum), covered(nor)
    print(f"   tumor  con caracteristicas: {len(tum_ok)}/{len(tum)}")
    print(f"   normal con caracteristicas: {len(nor_ok)}/{len(nor)}")
    if len(nor_ok) < max(10, 0.5 * len(nor)):
        print("\n   AVISO: faltan caracteristicas para la mayoria de los\n"
              "   normales. Este control exige extraerlas primero, lo que\n"
              "   ya no es una tarde de GPU. Decide eso antes de seguir.")

    # --- 3. the patient-overlap problem ----------------------------------
    pat_t = {parse(s)[0] for s in tum_ok}
    pat_n = {parse(s)[0] for s in nor_ok}
    both = pat_t & pat_n
    print(f"\nPacientes con solo tumor   {len(pat_t - pat_n):5d}")
    print(f"Pacientes con solo normal  {len(pat_n - pat_t):5d}")
    print(f"Pacientes con AMBOS        {len(both):5d}")
    if both:
        pct = 100 * len(both) / len(pat_t | pat_n)
        print(f"\n   {pct:.0f}% de los pacientes aportan las dos etiquetas.")
        print("   Agregar a paciente por probabilidad media, que es lo que")
        print("   hace el resto del pipeline, les asignaria una etiqueta que")
        print("   no existe. Dos salidas:")
        print("     (a) puntuar A NIVEL DE PORTAOBJETOS solo para este")
        print("         control, manteniendo el agrupamiento por paciente en")
        print("         los pliegues para que no haya fuga. Recomendada: no")
        print("         descarta datos y la fuga sigue controlada.")
        print("     (b) quedarse con un tipo por paciente, lo que deja")
        print(f"         {len(pat_t | pat_n) - len(both)} pacientes.")

    # --- 4. is the design even powered? ----------------------------------
    n_pos, n_neg = len(tum_ok), len(nor_ok)
    print("\nPotencia, a nivel de portaobjetos:")
    if n_pos and n_neg:
        for a in (0.90, 0.95, 0.99):
            se = hanley_mcneil_se(a, n_pos, n_neg)
            print(f"   si el AUROC real es {a:.2f}: IC95 de ancho "
                  f"+-{Z975 * se:.3f}, cota inferior {a - Z975 * se:.3f}")
        se0 = hanley_mcneil_se(0.5, n_pos, n_neg)
        print(f"   minimo detectable contra el azar: {(Z975 + Z80) * se0:.3f}")
        print("\n   Un control positivo solo sirve si su cota inferior queda")
        print("   muy por encima del techo del brazo principal (0.67). Si la")
        print("   lista de arriba lo cumple, el control es concluyente.")

    # --- 5. emit ----------------------------------------------------------
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = tum_ok + nor_ok
    lines = ["slide_id,patient_id,TUMOR"]
    for s in keep:
        pid, code = parse(s)
        lines.append(f"{s},{pid},{1 if code in TUMOUR else 0}")
    (out / "labels.csv").write_text("\n".join(lines) + "\n")
    (out / "tumor_control_slides.txt").write_text("\n".join(keep) + "\n")
    print(f"\nEscrito {out / 'labels.csv'}  ({len(keep)} filas)")
    print(f"Escrito {out / 'tumor_control_slides.txt'}")
    print("\nCopia o enlaza las caracteristicas a este --data_dir antes de")
    print("lanzar, y anade la columna TUMOR al labels.csv que consume el")
    print("benchmark si prefieres reutilizar data_control tal cual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
