"""Build the data_dir for the tumour-vs-normal positive control.

Everything this needs already exists on the DGX: each slide has its own
directory under data/slides holding embeddings.npy and pattern_probs.npy, and
that includes the 61 solid-tissue-normal slides dropped during cohort
correction. So this writes a labels.csv and a slide list, and nothing is
extracted or recomputed.

Three things it refuses to guess, and checks instead:
  - the slide-id format the benchmark expects, read off the existing
    data_control/labels.csv rather than assumed;
  - that every slide it emits actually has both .npy files;
  - that no tissue-source site contributes only one class, which would let
    the model learn the centre instead of the tissue.

Run from luad_controls/. Writes data_tumor_control/ and prints the launch.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

BARCODE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z0-9]?(?:-[A-Z0-9]+)*")
PATIENT = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
TSS = re.compile(r"TCGA-([A-Z0-9]{2})-")
STYPE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})")
NEEDED = ("embeddings.npy", "pattern_probs.npy")


def field(rx, s):
    m = rx.search(s)
    return m.group(0) if m else None


def group1(rx, s):
    m = rx.search(s)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides-root", default="/home/rapids/notebooks/slima/data/slides")
    ap.add_argument("--src-labels", default="data_control/labels.csv")
    ap.add_argument("--cohort", default="luad_cohort_slides.txt")
    ap.add_argument("--out", default="data_tumor_control")
    args = ap.parse_args()

    root = args.slides_root
    if not os.path.isdir(root):
        print(f"ERROR: no existe {root}", file=sys.stderr)
        return 1

    # --- 1. which slide directories actually carry features ---------------
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    ok = [d for d in dirs
          if all(os.path.exists(os.path.join(root, d, f)) for f in NEEDED)]
    print(f"directorios de portaobjetos: {len(dirs)}")
    print(f"   con {' + '.join(NEEDED)}: {len(ok)}")
    if len(ok) < len(dirs):
        print(f"   sin caracteristicas completas: {len(dirs) - len(ok)}")

    by_type = collections.Counter(group1(STYPE, d) or "??" for d in ok)
    print(f"   por tipo de muestra: {dict(sorted(by_type.items()))}")

    tum_dirs = [d for d in ok if group1(STYPE, d) == "01"]
    nor_dirs = [d for d in ok if group1(STYPE, d) == "11"]
    print(f"   tumor {len(tum_dirs)}   normal {len(nor_dirs)}")
    if not nor_dirs:
        print("\nPARADA: no hay normales con caracteristicas.", file=sys.stderr)
        return 1

    # --- 2. learn the id format the benchmark already uses -----------------
    with open(args.src_labels, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"ERROR: {args.src_labels} vacio", file=sys.stderr)
        return 1
    cols = list(rows[0].keys())
    print(f"\ncolumnas de {args.src_labels}:\n   {cols}")

    def find_col(*hints):
        for h in hints:
            for c in cols:
                if c.lower() == h:
                    return c
        for h in hints:
            for c in cols:
                if h in c.lower():
                    return c
        return None

    c_slide = find_col("slide_id", "slide")
    c_pat = find_col("patient_id", "patient", "case_id", "case")
    c_site = find_col("site", "tss", "source_site")
    print(f"   slide={c_slide}  patient={c_pat}  site={c_site}")
    if not c_slide:
        print("ERROR: no identifico la columna de portaobjetos", file=sys.stderr)
        return 1

    sample = rows[0][c_slide]
    dirset = set(ok)
    fmt = "dir" if sample in dirset else "barcode"
    print(f"   ejemplo de id: {sample}")
    print(f"   formato detectado: {fmt}"
          + ("  (nombre de directorio completo, con UUID)" if fmt == "dir"
             else "  (codigo de barras sin UUID)"))

    def emit_id(d: str) -> str:
        return d if fmt == "dir" else (field(BARCODE, d) or d)

    # sanity: the format we will emit must resolve back to a real directory
    probe = emit_id(nor_dirs[0])
    resolves = probe in dirset or any(probe in d for d in dirset)
    print(f"   comprobacion de resolucion sobre un normal: "
          f"{'OK' if resolves else 'FALLA'}  ({probe})")
    if not resolves:
        print("ERROR: el id que emitiria no resuelve a un directorio.",
              file=sys.stderr)
        return 1

    # --- 3. positives: the paper's own 458-slide cohort --------------------
    idx = {}
    for d in ok:
        idx.setdefault(field(BARCODE, d), d)
    with open(args.cohort) as fh:
        cohort = [ln.strip() for ln in fh if ln.strip()]
    pos = []
    missing = []
    for s in cohort:
        b = field(BARCODE, s) or s
        (pos.append(idx[b]) if b in idx else missing.append(s))
    print(f"\ncohorte del paper: {len(cohort)} portaobjetos")
    print(f"   con caracteristicas: {len(pos)}")
    if missing:
        print(f"   SIN caracteristicas: {len(missing)}  p.ej. {missing[:3]}")

    neg = nor_dirs
    print(f"negativos (normales): {len(neg)}")

    # --- 4. the site confound ---------------------------------------------
    tss_p = collections.Counter(group1(TSS, d) for d in pos)
    tss_n = collections.Counter(group1(TSS, d) for d in neg)
    solo_n = set(tss_n) - set(tss_p)
    solo_p = set(tss_p) - set(tss_n)
    print(f"\ncentros: tumor {len(tss_p)}, normal {len(tss_n)}, "
          f"compartidos {len(set(tss_p) & set(tss_n))}")
    if solo_n:
        print(f"   AVISO: {sorted(solo_n)} aportan SOLO normales. El modelo\n"
              f"   podria aprender el centro en vez del tejido.")
    if solo_p:
        print(f"   {len(solo_p)} centros aportan solo tumor. Bajo el protocolo\n"
              f"   por sitio, los pliegues que retengan esos centros no tienen\n"
              f"   negativos; el AUROC agrupado sobre todos los pliegues sigue\n"
              f"   definido, pero conviene leer sobre todo el arm por paciente.")

    # --- 5. patients must not carry both labels ---------------------------
    pp = {field(PATIENT, d) for d in pos}
    pn = {field(PATIENT, d) for d in neg}
    both = pp & pn
    print(f"\npacientes: solo tumor {len(pp - pn)}, solo normal {len(pn - pp)}, "
          f"ambos {len(both)}")
    level = "paciente"
    if both:
        level = "portaobjetos"
        print(f"   {len(both)} pacientes aportan las dos etiquetas, asi que\n"
              f"   este control hay que puntuarlo a nivel de portaobjetos.")
    else:
        print("   Ningun paciente aporta las dos, asi que se puede puntuar a\n"
              "   nivel de paciente igual que el brazo principal.")

    # --- 6. emit -----------------------------------------------------------
    out = args.out
    os.makedirs(out, exist_ok=True)
    link = os.path.join(out, "slides")
    if not os.path.exists(link):
        try:
            os.symlink(root, link)
            print(f"\nenlace {link} -> {root}")
        except OSError as e:
            print(f"\nNo pude crear el enlace {link} ({e}).")
            print(f"Hazlo a mano:  ln -s {root} {link}")

    header = [c_slide] + ([c_pat] if c_pat else []) \
        + ([c_site] if c_site else []) + ["TUMOR"]
    with open(os.path.join(out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for d, y in [(d, 1) for d in pos] + [(d, 0) for d in neg]:
            row = [emit_id(d)]
            if c_pat:
                row.append(field(PATIENT, d))
            if c_site:
                row.append(group1(TSS, d))
            row.append(y)
            w.writerow(row)
    with open(os.path.join(out, "tumor_control_slides.txt"), "w") as fh:
        fh.write("\n".join(emit_id(d) for d in pos + neg) + "\n")

    n = len(pos) + len(neg)
    print(f"escrito {out}/labels.csv  ({n} filas, {len(pos)} tumor / "
          f"{len(neg)} normal)")
    print(f"escrito {out}/tumor_control_slides.txt")
    print(f"\nPuntuar a nivel de {level}.")
    print("\nLanzar con:\n")
    print("  for M in patient site; do for S in 42 142 242; do")
    print('    python "pattern_informed_abmil_benchmark_v2 (3).py" \\')
    print(f'      --data_dir "$PWD/{out}" \\')
    print(f'      --results_dir "$PWD/results_tumor/${{M}}_seed${{S}}" \\')
    print(f'      --slide_list "$PWD/{out}/tumor_control_slides.txt" \\')
    print('      --n_folds 5 --seed "$S" --abmil_epochs 50 '
          '--inner_val_frac 0.15 \\')
    print('      --split_mode "$M" \\')
    print('      --conditions baseline2_abmil_embeddings '
          'baseline3_abmil_patterns \\')
    print("      --genes TUMOR")
    print("  done; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
