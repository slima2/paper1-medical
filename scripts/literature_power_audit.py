"""Power audit of reported MIL-aggregator gains in mutation prediction.

The question is not whether a published gain is real. It is whether the study
that reports it could have distinguished it from zero. We answer that by
transporting a variance component measured in our own five-seed, two-protocol
run onto each published design.

Transport is done in two stages rather than by rescaling our standard error
directly, because the standard error of an AUROC depends on the AUROC itself
and on both class counts, and published studies sit at higher AUROC than ours.

  Stage 1. For each of our six genes, compute the Hanley-McNeil analytic SE of
           a single AUROC at our observed AUROC, n+ and n-. Compare it with the
           SE of the *paired difference* that our hierarchical bootstrap
           actually measured. Their ratio is an inflation factor f that
           absorbs (a) the paired correlation between two aggregators scoring
           the same patients and (b) the seed-to-seed component that a
           single-seed cross-validation never sees.

  Stage 2. For a published study, compute its Hanley-McNeil SE from its own
           reported AUROC and class counts, multiply by the median f, and
           convert to a minimum detectable effect at alpha=.05 two-sided,
           80% power.

Every published number used here is transcribed from the primary source and
carries its citation in LITERATURE_ROWS. Nothing is inferred from abstracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Z_ALPHA = 1.959963985  # two-sided 0.05
Z_POWER = 0.8416212336  # 80% power
K_MDE = Z_ALPHA + Z_POWER


def hanley_mcneil_se(auroc: float, n_pos: int, n_neg: int) -> float:
    """Analytic SE of a single AUROC (Hanley & McNeil 1982, Radiology 143:29-36)."""
    a = auroc
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (
        a * (1.0 - a)
        + (n_pos - 1) * (q1 - a * a)
        + (n_neg - 1) * (q2 - a * a)
    ) / (n_pos * n_neg)
    return math.sqrt(var)


# --------------------------------------------------------------------------
# Stage 1 inputs: our own run.
#   auroc   = pooled out-of-fold AUROC of B2, mean over five seeds
#             (bootstrap_v4_final/luad_pooled_oof_aurocs_by_seed.csv)
#   se_diff = SE of the paired contrast, taken as the mean of the PI-B2 and
#             FC-B2 hierarchical interval half-widths divided by 1.96
#             (bootstrap_v4_final/luad_hierarchical_contrasts.csv, patient)
# --------------------------------------------------------------------------
OURS = {
    "TP53":  dict(auroc=0.6608, n_pos=220, n_neg=224, se_diff=0.01307),
    "KRAS":  dict(auroc=0.5170, n_pos=124, n_neg=320, se_diff=0.02202),
    "KEAP1": dict(auroc=0.4878, n_pos=79,  n_neg=365, se_diff=0.02810),
    "EGFR":  dict(auroc=0.5816, n_pos=67,  n_neg=377, se_diff=0.02552),
    "STK11": dict(auroc=0.5907, n_pos=58,  n_neg=386, se_diff=0.02420),
    "RBM10": dict(auroc=0.5364, n_pos=30,  n_neg=414, se_diff=0.03757),
}


@dataclass
class Claim:
    study: str
    citation: str
    task: str
    n_pos: int
    n_neg: int
    baseline: str
    baseline_auroc: float
    proposed: str
    proposed_auroc: float
    reports_ci: bool
    source_note: str
    verified: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.proposed_auroc - self.baseline_auroc


# --------------------------------------------------------------------------
# Stage 2 inputs. VERIFIED means the numbers were read from the primary text,
# not from an abstract, a summary or a leaderboard aggregator.
# --------------------------------------------------------------------------
LITERATURE_ROWS: list[Claim] = [
    # Brattoli, Shi, Park, Lee, Yoo & Pereira (Lunit), arXiv:2508.02431.
    # Internal cohort of 1,223 NGS-assessed surgical resection WSIs, stratified
    # 5-fold CV. Table 1 gives per-mutation AUROC for each aggregator and the
    # "#Samples Pos/Neg" row gives the class counts used below. ViT-based FM
    # (H-optimus-0) row. No confidence interval or test appears in the paper.
    *[
        Claim(
            study="Brattoli et al. 2025",
            citation="arXiv:2508.02431, Table 1 (ViT FM row)",
            task=f"{gene} (NSCLC, internal 1,223 WSI)",
            n_pos=npos,
            n_neg=nneg,
            baseline="ABMIL",
            baseline_auroc=abmil,
            proposed="AsymTransDec",
            proposed_auroc=ours,
            reports_ci=False,
            source_note="primary text, arXiv HTML",
        )
        for gene, npos, nneg, abmil, ours in [
            ("ALK",     70, 1130, 0.875, 0.884),
            ("BRAF",   100, 1123, 0.824, 0.865),
            ("EGFR",   275,  940, 0.893, 0.908),
            ("ERBB2",   78, 1145, 0.755, 0.801),
            ("KRAS",   184, 1025, 0.842, 0.859),
            ("METex14", 88, 1135, 0.841, 0.864),
        ]
    ],
    # Same paper, Table 4: the architectural contribution proper, isolated
    # against a vanilla transformer decoder on the identical data. Averages
    # only are reported per architecture, so we audit the average using the
    # median class counts across the six tasks.
    Claim(
        study="Brattoli et al. 2025",
        citation="arXiv:2508.02431, Table 4 (six-task average)",
        task="AsymTransDec vs vanilla TransDec (avg of 6)",
        n_pos=94,   # median of 70,100,275,78,184,88
        n_neg=1129,
        baseline="TransDec",
        baseline_auroc=0.849,
        proposed="AsymTransDec",
        proposed_auroc=0.863,
        reports_ci=False,
        source_note="primary text, arXiv HTML",
        warnings=["average over six tasks; audited at the median class counts"],
    ),
    # Same paper, Table 3: the tissue-stratification contribution, isolated.
    Claim(
        study="Brattoli et al. 2025",
        citation="arXiv:2508.02431, Table 3 (six-task average)",
        task="tissue stratification vs none (avg of 6)",
        n_pos=94,
        n_neg=1129,
        baseline="all patches, no stratification",
        baseline_auroc=0.855,
        proposed="all patches, stratified",
        proposed_auroc=0.863,
        reports_ci=False,
        source_note="primary text, arXiv HTML",
        warnings=["average over six tasks; audited at the median class counts"],
    ),
]


def stage1_inflation() -> tuple[float, list[tuple[str, float, float, float]]]:
    rows = []
    for gene, d in OURS.items():
        se_single = hanley_mcneil_se(d["auroc"], d["n_pos"], d["n_neg"])
        f = d["se_diff"] / se_single
        rows.append((gene, se_single, d["se_diff"], f))
    fs = sorted(r[3] for r in rows)
    mid = len(fs) // 2
    median_f = fs[mid] if len(fs) % 2 else (fs[mid - 1] + fs[mid]) / 2
    return median_f, rows


# --------------------------------------------------------------------------
# Evidence stream B. Ghaffari Laleh et al., Medical Image Analysis 79:102474
# (2022). Six pipelines, five mutation-prediction tasks, all pairwise
# comparisons tested with DeLong. Counts below are transcribed verbatim from
# the Results section, "Explainability of the performance differences".
# This study is a positive precedent, not an audit target: it reports 95% CIs
# by 1000x bootstrapping and a significance test for every comparison.
# --------------------------------------------------------------------------
LALEH = dict(
    citation="Ghaffari Laleh et al. 2022, Med. Image Anal. 79:102474",
    n_methods=6,
    n_tasks=5,
    # "significantly outperformed other methods in k/25"
    wins={"ResNet": 5, "ViT": 7, "EfficientNet": 7, "AttMIL": 4,
          "CLAM": 1, "classical MIL": 0},
    losses={"classical MIL": 15, "CLAM": 9, "ResNet": 0, "ViT": 0,
            "EfficientNet": 0, "AttMIL": 0},
)

# --------------------------------------------------------------------------
# Evidence stream C. Chen et al., PMLR 254:38-50 (2024), Supplementary
# Table 3: AUC reported for the SAME aggregator on the SAME public benchmark,
# as transcribed by each of several source papers. Verbatim from the primary
# text. Key -> {source paper: AUC}.
# --------------------------------------------------------------------------
CHEN_CROSS_SOURCE = {
    "CAMELYON16": {
        "AB-MIL":   {"DS-MIL": .865, "TransMIL": .876, "DTFD-MIL": .854,
                     "CWC-transformer": .876, "MHIM-MIL": .940, "WENO": .838},
        "MIL-RNN":  {"DS-MIL": .806, "TransMIL": .889, "DTFD-MIL": .875},
        "DS-MIL":   {"DS-MIL": .894, "TransMIL": .818, "DTFD-MIL": .899,
                     "CWC-transformer": .894, "MHIM-MIL": .946, "WENO": .840},
        "CLAM-SB":  {"TransMIL": .881, "DTFD-MIL": .871,
                     "CWC-transformer": .881, "MHIM-MIL": .947, "CAMIL": .933},
        "CLAM-MB":  {"TransMIL": .868, "DTFD-MIL": .878,
                     "CWC-transformer": .868, "MHIM-MIL": .947, "CAMIL": .938},
        "TransMIL": {"TransMIL": .931, "DTFD-MIL": .906,
                     "CWC-transformer": .931, "MHIM-MIL": .935, "CAMIL": .950},
        "DTFD-MIL": {"DTFD-MIL": .946, "MHIM-MIL": .952, "CAMIL": .941},
    },
    "TCGA-Lung": {
        "AB-MIL":   {"DS-MIL": .977, "DTFD-MIL": .941,
                     "CWC-transformer": .866, "MHIM-MIL": .914, "WENO": .949},
        "MIL-RNN":  {"DS-MIL": .964, "DTFD-MIL": .894, "WENO": .911},
        "DS-MIL":   {"DS-MIL": .981, "DTFD-MIL": .939,
                     "CWC-transformer": .960, "MHIM-MIL": .937, "WENO": .963},
        "CLAM-SB":  {"DTFD-MIL": .944, "CWC-transformer": .882,
                     "MHIM-MIL": .937},
        "CLAM-MB":  {"DTFD-MIL": .949, "CWC-transformer": .916,
                     "MHIM-MIL": .937},
        "TransMIL": {"DTFD-MIL": .949, "CWC-transformer": .936,
                     "MHIM-MIL": .925},
    },
}


def stream_b() -> None:
    n_pairs = LALEH["n_methods"] * (LALEH["n_methods"] - 1) // 2
    n_comparisons = n_pairs * LALEH["n_tasks"]
    total_wins = sum(LALEH["wins"].values())
    total_losses = sum(LALEH["losses"].values())
    print("=" * 78)
    print("LINEA B  Cuando alguien SI testea: Laleh et al. 2022, test de DeLong")
    print("=" * 78)
    print(f"  {LALEH['n_methods']} metodos x {LALEH['n_tasks']} tareas de mutacion")
    print(f"  pares unicos = {n_pairs}, comparaciones unicas = {n_comparisons}")
    print(f"  victorias significativas sumadas = {total_wins}")
    print(f"  derrotas significativas sumadas  = {total_losses}")
    assert total_wins == total_losses, "victorias y derrotas deben cuadrar"
    print("  (cuadran, luego hay exactamente ese numero de comparaciones significativas)")
    ns = n_comparisons - total_wins
    print(f"\n  NO significativas: {ns}/{n_comparisons} = {ns / n_comparisons:.0%}")
    print("  Dos tercios de las comparaciones entre arquitecturas no alcanzan")
    print("  significacion cuando se las somete a un test pareado.")


def stream_c() -> None:
    print()
    print("=" * 78)
    print("LINEA C  El MISMO metodo, el MISMO benchmark, distinto paper que lo reporta")
    print("=" * 78)
    print("  Chen et al. 2024, Tabla Suplementaria 3 (transcrita literalmente).")
    worst = None
    for dataset, methods in CHEN_CROSS_SOURCE.items():
        print(f"\n  {dataset}")
        print(f"    {'metodo':10s} {'n fuentes':>9s} {'min':>6s} {'max':>6s} {'rango':>7s}")
        for method, by_source in methods.items():
            vals = list(by_source.values())
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            print(f"    {method:10s} {len(vals):9d} {lo:6.3f} {hi:6.3f} {rng:7.3f}")
            if worst is None or rng > worst[2]:
                worst = (method, dataset, rng, lo, hi)
    m, d, rng, lo, hi = worst
    print(f"\n  Peor caso: {m} en {d} va de {lo:.3f} a {hi:.3f}, un rango de "
          f"{rng:.3f} ({rng * 100:.1f} puntos)")
    print("  segun QUE paper lo reimplemente. Las ganancias que esos mismos")
    print("  papers reivindican son de 1 a 3 puntos.")


def main() -> None:
    median_f, rows = stage1_inflation()

    print("=" * 78)
    print("ETAPA 1  Factor de inflacion f = SE(contraste pareado) / SE(Hanley-McNeil)")
    print("=" * 78)
    print(f"{'gen':7s} {'AUROC':>7s} {'n+':>5s} {'n-':>5s} "
          f"{'SE_HM':>8s} {'SE_par':>8s} {'f':>6s}")
    for gene, se_single, se_diff, f in rows:
        d = OURS[gene]
        print(f"{gene:7s} {d['auroc']:7.4f} {d['n_pos']:5d} {d['n_neg']:5d} "
              f"{se_single:8.4f} {se_diff:8.4f} {f:6.2f}")
    print(f"\nf mediano = {median_f:.2f}   rango = "
          f"[{min(r[3] for r in rows):.2f}, {max(r[3] for r in rows):.2f}]")
    print("f < 1: el contraste pareado es MENOS variable que un AUROC aislado,")
    print("porque ambos agregadores puntuan a los mismos pacientes y la")
    print("correlacion del pareado mas que compensa la varianza entre semillas.")
    print("Transportar f<1 es la eleccion caritativa: concede a cada estudio")
    print("auditado todo el beneficio del pareado que su diseno permitiria.")

    print()
    print("=" * 78)
    print("ETAPA 2  Auditoria de ganancias publicadas")
    print("=" * 78)
    hdr = (f"{'estudio / tarea':46s} {'n+':>5s} {'base':>6s} {'prop':>6s} "
           f"{'gan':>6s} {'MDE':>6s} {'gan/MDE':>8s}  veredicto")
    print(hdr)
    print("-" * len(hdr))
    n_below = 0
    for c in LITERATURE_ROWS:
        se_single = hanley_mcneil_se(c.proposed_auroc, c.n_pos, c.n_neg)
        se_diff = se_single * median_f
        mde = K_MDE * se_diff
        ratio = c.gain / mde
        below = ratio < 1.0
        n_below += below
        verdict = "por debajo del MDE" if below else "resoluble"
        label = f"{c.study.split()[0]} {c.task}"
        print(f"{label:46.46s} {c.n_pos:5d} {c.baseline_auroc:6.3f} "
              f"{c.proposed_auroc:6.3f} {c.gain:6.3f} {mde:6.3f} "
              f"{ratio:8.2f}  {verdict}")
    print("-" * len(hdr))
    print(f"{n_below} de {len(LITERATURE_ROWS)} afirmaciones caen por debajo del")
    print("efecto minimo detectable de su propio diseno.")
    n_ci = sum(c.reports_ci for c in LITERATURE_ROWS)
    print(f"{n_ci} de {len(LITERATURE_ROWS)} reportan intervalo de confianza.")

    stream_b()
    stream_c()

    print()
    print("SALVEDADES QUE DEBEN IR EN EL TEXTO:")
    print(" 1. f se calibra en NUESTRA cohorte, encoder y rango de AUROC (0.49-0.66).")
    print("    Los estudios auditados operan en 0.75-0.91, donde la varianza real")
    print("    es menor; Hanley-McNeil ya corrige por el nivel de AUROC, pero el")
    print("    componente de semilla podria no transportarse identico.")
    print(" 2. f incluye varianza entre semillas. Un diseno de una sola semilla no")
    print("    la mide, y por eso su incertidumbre reportada seria menor aunque su")
    print("    incertidumbre real no lo sea. Ese es justamente el punto.")
    print(" 3. Esto no afirma que las ganancias sean falsas. Afirma que los disenos")
    print("    publicados no pueden distinguirlas de cero, y que no acompanan la")
    print("    afirmacion de ninguna cuantificacion de incertidumbre.")


if __name__ == "__main__":
    main()
