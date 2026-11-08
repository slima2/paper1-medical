"""Check each numeric sentence in Results against Table 1 as printed.

Table 1 (tables/tab_mutation_luad.tex) is transcribed below exactly as it is
typeset. Every assertion the prose makes about it is then re-derived from the
transcription, so a mismatch means the prose and the table disagree, not that
someone misread one of them.
"""

from __future__ import annotations

GENES = ["TP53", "EGFR", "STK11", "KEAP1", "KRAS", "RBM10"]

# n_+ as printed in the Table 1 header (patient level).
N_POS = {"TP53": 220, "EGFR": 67, "STK11": 58,
         "KEAP1": 79, "KRAS": 124, "RBM10": 30}

# (mean, SD) per condition, exactly as typeset.
PATIENT = {
    "B2": {"TP53": (.661, .033), "EGFR": (.582, .060), "STK11": (.591, .019),
           "KEAP1": (.488, .029), "KRAS": (.517, .026), "RBM10": (.536, .047)},
    "B3": {"TP53": (.579, .029), "EGFR": (.480, .023), "STK11": (.499, .017),
           "KEAP1": (.485, .038), "KRAS": (.486, .031), "RBM10": (.490, .037)},
    "PI": {"TP53": (.658, .020), "EGFR": (.580, .046), "STK11": (.578, .050),
           "KEAP1": (.446, .032), "KRAS": (.516, .025), "RBM10": (.552, .027)},
    "FC": {"TP53": (.646, .031), "EGFR": (.567, .060), "STK11": (.608, .021),
           "KEAP1": (.460, .020), "KRAS": (.506, .019), "RBM10": (.528, .038)},
}
SITE = {
    "B2": {"TP53": (.640, .024), "EGFR": (.584, .038), "STK11": (.605, .019),
           "KEAP1": (.482, .021), "KRAS": (.507, .028), "RBM10": (.525, .055)},
    "B3": {"TP53": (.561, .032), "EGFR": (.485, .052), "STK11": (.523, .044),
           "KEAP1": (.488, .059), "KRAS": (.475, .039), "RBM10": (.517, .045)},
    "PI": {"TP53": (.630, .043), "EGFR": (.576, .050), "STK11": (.593, .024),
           "KEAP1": (.472, .019), "KRAS": (.502, .045), "RBM10": (.512, .057)},
    "FC": {"TP53": (.636, .026), "EGFR": (.557, .040), "STK11": (.597, .056),
           "KEAP1": (.471, .015), "KRAS": (.501, .023), "RBM10": (.511, .071)},
}

# B3-B2 point and interval, and whether the table typesets it bold.
B3B2_PATIENT = {
    "TP53":  (-.082, (-.134, -.027), True),
    "EGFR":  (-.102, (-.178, -.030), True),
    "STK11": (-.092, (-.170, -.021), True),
    "KEAP1": (-.003, (-.076, +.070), False),
    "KRAS":  (-.031, (-.094, +.024), False),
    "RBM10": (-.046, (-.161, +.073), False),
}
B3B2_SITE = {
    "TP53":  (-.079, (-.131, -.024), True),
    "EGFR":  (-.099, (-.174, -.019), True),
    "STK11": (-.083, (-.181, +.001), False),
    "KEAP1": (+.006, (-.067, +.093), False),
    "KRAS":  (-.032, (-.093, +.035), False),
    "RBM10": (-.007, (-.103, +.108), False),
}

CONCLUSIVE_THRESHOLD = 0.70  # Methods, eq:conclusive_verdict

checks: list[tuple[str, bool, str]] = []


def check(claim: str, ok: bool, detail: str) -> None:
    checks.append((claim, ok, detail))


def best(gene: str, table) -> tuple[str, float]:
    c = max(table, key=lambda k: table[k][gene][0])
    return c, table[c][gene][0]


# --- RESULTS L37-39 -------------------------------------------------------
clears = [g for g in GENES
          if PATIENT["B2"][g][0] > .60 or SITE["B2"][g][0] > .60]
clears_both = [g for g in GENES
               if PATIENT["B2"][g][0] > .60 and SITE["B2"][g][0] > .60]
check("'TP53 is the only gene whose embedding baseline clears 0.60 "
      "under BOTH protocols'",
      clears_both == ["TP53"],
      f"clears bajo ambos: {clears_both} "
      f"(bajo al menos uno seria {clears}, por eso 'either' era ambiguo)")

low = [g for g in GENES if PATIENT["B2"][g][0] <= .52]
low_site = [g for g in GENES if SITE["B2"][g][0] <= .52]
low_best = [g for g in GENES if best(g, PATIENT)[1] <= .52]
check("'two of six sit at or below 0.52'",
      len(low) == 2 and len(low_site) == 2 and len(low_best) == 2,
      f"B2 paciente: {len(low)} {low}; B2 sitio: {len(low_site)} {low_site}; "
      f"mejor condicion: {len(low_best)} {low_best}")

# --- RESULTS L40-42: which genes resolve B3-B2 ----------------------------
res_p = [g for g in GENES if B3B2_PATIENT[g][1][1] < 0]
res_s = [g for g in GENES if B3B2_SITE[g][1][1] < 0]
check("'B3-B2 resolves on TP53 and EGFR under both protocols, STK11 "
      "under one'",
      res_s == ["TP53", "EGFR"] and res_p == ["TP53", "EGFR", "STK11"],
      f"paciente: {res_p}; sitio: {res_s}")

bold_p = [g for g in GENES if B3B2_PATIENT[g][2]]
bold_s = [g for g in GENES if B3B2_SITE[g][2]]
check("pie de Tabla 1: bold = TP53/EGFR/STK11 en (a), solo TP53/EGFR en (b)",
      bold_p == res_p and bold_s == res_s,
      f"negrita en panel a: {bold_p}; en panel b: {bold_s}; "
      f"coinciden con los intervalos que excluyen cero")

# --- RESULTS L72-78 y L97-98: controles -----------------------------------
SEX_CONTROL = 0.588   # "A positive control, patient sex, returns 0.588"
EGFR_B2 = PATIENT["B2"]["EGFR"][0]
check("'EGFR at 0.582 sits marginally BELOW the 0.588 the demographic "
      "control returns'",
      EGFR_B2 < SEX_CONTROL,
      f"EGFR B2 = {EGFR_B2:.3f} vs control de sexo = {SEX_CONTROL:.3f} "
      f"-> diferencia {EGFR_B2 - SEX_CONTROL:+.3f}")

# --- RESULTS L89-91 -------------------------------------------------------
check("'fusion moves the AUROC by <=0.02 in point estimate' (TP53 y EGFR)",
      all(abs(PATIENT[c][g][0] - PATIENT["B2"][g][0]) <= .02
          for g in ("TP53", "EGFR") for c in ("PI", "FC")),
      "; ".join(f"{g} {c}-B2={PATIENT[c][g][0] - PATIENT['B2'][g][0]:+.3f}"
                for g in ("TP53", "EGFR") for c in ("PI", "FC")))

check("'the pattern channel alone falls 0.082 and 0.102 below the embedding'",
      abs(B3B2_PATIENT["TP53"][0] + .082) < 1e-9
      and abs(B3B2_PATIENT["EGFR"][0] + .102) < 1e-9,
      f"TP53 {B3B2_PATIENT['TP53'][0]}, EGFR {B3B2_PATIENT['EGFR'][0]}")

# --- RESULTS L141-152 -----------------------------------------------------
check("'STK11 (best 0.608)'", abs(best("STK11", PATIENT)[1] - .608) < 1e-9,
      f"mejor paciente = {best('STK11', PATIENT)}")
check("'RBM10 (best 0.552)'", abs(best("RBM10", PATIENT)[1] - .552) < 1e-9,
      f"mejor paciente = {best('RBM10', PATIENT)}")
check("'STK11 B3 falling 0.092 below B2'",
      abs(B3B2_PATIENT["STK11"][0] + .092) < 1e-9,
      f"{B3B2_PATIENT['STK11'][0]}")

# --- RESULTS L236-246 -----------------------------------------------------
gap_best = abs(best("KEAP1", PATIENT)[1] - best("RBM10", PATIENT)[1])
gap_b2 = abs(PATIENT["B2"]["KEAP1"][0] - PATIENT["B2"]["RBM10"][0])
check("'KEAP1 and RBM10 sit within 0.05 AUROC of each other'",
      gap_best <= .05 or gap_b2 <= .05,
      f"mejor condicion: {gap_best:.3f}; solo B2: {gap_b2:.3f}")

keap = [PATIENT[c]["KEAP1"][0] for c in PATIENT] + \
       [SITE[c]["KEAP1"][0] for c in SITE]
check("'KEAP1: both channels at or below 0.49 under both protocols'",
      max(keap) <= .49, f"maximo KEAP1 en cualquier condicion = {max(keap):.3f}")

sds = [PATIENT[c]["RBM10"][1] for c in PATIENT] + \
      [SITE[c]["RBM10"][1] for c in SITE]
check("'RBM10 across-seed spread of +-0.027 to +-0.071'",
      abs(min(sds) - .027) < 1e-9 and abs(max(sds) - .071) < 1e-9,
      f"rango observado {min(sds):.3f}-{max(sds):.3f}")

check("'RBM10 pattern does not exceed embedding (0.490 against 0.536)'",
      PATIENT["B3"]["RBM10"][0] == .490 and PATIENT["B2"]["RBM10"][0] == .536,
      f"B3={PATIENT['B3']['RBM10'][0]}, B2={PATIENT['B2']['RBM10'][0]}")

check("'FC-MIL scores 0.506 on KRAS'",
      PATIENT["FC"]["KRAS"][0] == .506, f"{PATIENT['FC']['KRAS'][0]}")

# --- verdicto Conclusive contra el criterio de Metodos --------------------
conclusive = [g for g in GENES
              if max(best(g, PATIENT)[1], best(g, SITE)[1])
              >= CONCLUSIVE_THRESHOLD]
check(f"la banda Conclusive de LUAD esta vacia (criterio: mean AUROC "
      f">= {CONCLUSIVE_THRESHOLD})",
      conclusive == [],
      f"genes que alcanzan el umbral: "
      f"{conclusive if conclusive else 'ninguno'}; "
      f"maximo del estudio = {max(best(g, PATIENT)[1] for g in GENES):.3f} "
      f"(TP53 B2, paciente)")

# --- unidades -------------------------------------------------------------
check("'1,200 gene-condition-fold units'",
      5 * 2 * 6 * 4 * 5 == 1200, f"5x2x6x4x5 = {5 * 2 * 6 * 4 * 5}")

# --- n_+ de RBM10: suelo de estimacion ------------------------------------
ESTIMATION_FLOOR = 32  # Methods: umbral fijo de positivos
check("RBM10 cae por debajo del suelo de estimacion a nivel de paciente",
      N_POS["RBM10"] < ESTIMATION_FLOOR,
      f"n_+ paciente = {N_POS['RBM10']} < suelo {ESTIMATION_FLOOR}; "
      f"su recuento de portaobjetos (32) coincide con el valor del suelo "
      f"por casualidad, aclarado en tab_mutation_prevalence")

if __name__ == "__main__":
    ok = sum(1 for _, o, _ in checks if o)
    print(f"{ok}/{len(checks)} afirmaciones concuerdan con la Tabla 1\n")
    for claim, o, detail in checks:
        mark = "OK  " if o else "FALLA"
        print(f"[{mark}] {claim}")
        print(f"        {detail}")
