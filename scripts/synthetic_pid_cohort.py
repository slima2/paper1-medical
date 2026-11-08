#!/usr/bin/env python3
"""Synthetic two-channel cohorts whose PID atoms are known by construction.

The regime reads in this paper are signatures, not measurements: we infer the
presence of an atom from the ordering of four trained conditions. That
inference assumes the map from atom configuration to signature is informative,
and the manuscript asserts it rather than demonstrating it. This script builds
the demonstration, by generating cohorts in which the atoms are set by fiat and
then running the identical adjudication over them.

The output is written in the exact on-disk layout the benchmark consumes

    <out>/labels.csv                       slide_id, patient_id, site, <gene>
    <out>/slides/<slide_id>/pattern_probs.npy    (T, 6)  float32, rows sum to 1
    <out>/slides/<slide_id>/pattern_labels.npy   (T,)    int64
    <out>/slides/<slide_id>/embeddings.npy       (T, D)  float32

so pattern_informed_abmil_benchmark_v2.py runs over it unmodified. Slide ids
follow the TCGA barcode shape because the benchmark derives patient and site
grouping by parsing it.

CONSTRUCTION
------------
Five independent latent Bernoulli factors are drawn per patient. Each is
visible in one channel, both, or neither:

    z_R     visible in BOTH channels        -> redundant information
    z_E     visible in the embedding only   -> unique to E
    z_P     visible in the vocabulary only  -> unique to P
    z_S1    visible in the vocabulary only  ] jointly determine Y under XOR,
    z_S2    visible in the embedding only   ] so neither channel alone informs

A configuration chooses which factors drive the label, which fixes the atom
content exactly. Channel disjointness is enforced by construction: the
embedding never encodes z_P or z_S1, and the pattern simplex never encodes
z_E or z_S2.

Signal enters only a `witness_rate` fraction of tiles, as in a real bag where
most tissue is uninformative.

Usage
-----
    # cheap calibration first: what AUROC does this effect size buy?
    python scripts/synthetic_pid_cohort.py --config R_UE --calibrate

    # then write a cohort
    python scripts/synthetic_pid_cohort.py --config R_UE --n-patients 444 \
        --seed 0 --out /path/to/synth/R_UE_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_PATTERNS = 6
GENE = "SYNTH"

# Which latent factors drive the label, per configuration. The atom content
# follows directly, because each factor's channel visibility is fixed above.
CONFIGS: dict[str, dict] = {
    "EMPTY": {
        "drivers": [],
        "atoms": {"R": 0, "U_P": 0, "U_E": 0, "S": 0},
        "expected_regime": "none (all conditions at chance)",
    },
    "R": {
        "drivers": ["z_R"],
        "atoms": {"R": 1, "U_P": 0, "U_E": 0, "S": 0},
        "expected_regime": "(i) pure redundancy: B2 ~ B3 ~ PI ~ FC, all above chance",
    },
    "UE": {
        "drivers": ["z_E"],
        "atoms": {"R": 0, "U_P": 0, "U_E": 1, "S": 0},
        "expected_regime": "(iii) out-of-ontology: B2 >> B3 ~ chance, PI ~ FC ~ B2",
    },
    "UP": {
        "drivers": ["z_P"],
        "atoms": {"R": 0, "U_P": 1, "U_E": 0, "S": 0},
        "expected_regime": "vocabulary-unique: B3 >> B2 ~ chance, PI ~ B3",
    },
    "R_UE": {
        # The configuration the manuscript assigns to TP53 and EGFR.
        "drivers": ["z_R", "z_E"],
        "atoms": {"R": 1, "U_P": 0, "U_E": 1, "S": 0},
        "expected_regime": "(i) as instantiated: B2 > B3 > chance, PI ~ FC ~ B2",
    },
    "UP_UE": {
        # Complementary uniques: the cell glioma 1p/19q is said to occupy.
        "drivers": ["z_P", "z_E"],
        "atoms": {"R": 0, "U_P": 1, "U_E": 1, "S": 0},
        "expected_regime": "(ii-a) additive complementarity: PI > max(B2, B3)",
    },
    "S": {
        # Y = z_S1 XOR z_S2, one bit per channel. The cell LUAD left empty.
        "drivers": ["z_S1", "z_S2"],
        "atoms": {"R": 0, "U_P": 0, "U_E": 0, "S": 1},
        "expected_regime": "(ii-b) synergy: B2 ~ B3 ~ PI ~ chance, FC > PI",
    },
}

# Each latent factor perturbs a disjoint slice of its channel(s). Pattern pairs
# are (positive, negative) indices into the six-pattern simplex; embedding
# directions are disjoint blocks of coordinates.
PATTERN_SLOTS = {"z_R": (0, 1), "z_P": (2, 3), "z_S1": (4, 5)}
EMBED_SLOTS = {"z_R": 0, "z_E": 1, "z_S2": 2}
# Coordinates per factor. The embedding signal is diluted across embed_dim in
# the bag mean whereas the vocabulary signal sits in 2 of 6 columns, so a block
# too small makes E artificially harder to read than P for reasons that belong
# to this construction rather than to the science. Calibrate before running.
EMBED_BLOCK = 64


def make_slide_id(patient_idx: int, site_idx: int) -> str:
    """TCGA-shaped id: the benchmark parses patient and site out of this."""
    return f"TCGA-{site_idx:02d}-{patient_idx:04d}-01Z-00-DX1"


def draw_latents(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        name: rng.integers(0, 2, size=n)
        for name in ("z_R", "z_E", "z_P", "z_S1", "z_S2")
    }


def labels_from(config: str, z: dict[str, np.ndarray],
                rng: np.random.Generator, label_noise: float) -> np.ndarray:
    drivers = CONFIGS[config]["drivers"]
    n = len(next(iter(z.values())))
    if not drivers:
        y = rng.integers(0, 2, size=n)
    elif config == "S":
        y = z["z_S1"] ^ z["z_S2"]
    elif len(drivers) == 1:
        y = z[drivers[0]].copy()
    else:
        # Two independent additive contributions. A logical OR would work but
        # puts prevalence at 0.75 and makes one channel nearly sufficient; the
        # majority rule below keeps prevalence balanced, leaves each channel
        # individually informative but insufficient, and stays strictly
        # additive so no synergy leaks into a configuration that declares none.
        votes = np.sum([2 * z[d] - 1 for d in drivers], axis=0)
        coin = rng.integers(0, 2, size=n)
        y = np.where(votes > 0, 1, np.where(votes < 0, 0, coin))
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y = np.where(flip, 1 - y, y)
    return y.astype(int)


def build_bag(
    rng: np.random.Generator,
    z_patient: dict[str, int],
    n_tiles: int,
    embed_dim: int,
    effect: float,
    witness_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One patient's two channels. Returns (pattern_probs, embeddings)."""
    witness = rng.random(n_tiles) < witness_rate

    # ---- vocabulary channel: a 6-simplex per tile -------------------------
    logits = rng.normal(0.0, 1.0, size=(n_tiles, N_PATTERNS))
    for factor, (pos, neg) in PATTERN_SLOTS.items():
        if z_patient[factor] == 1:
            logits[witness, pos] += effect
            logits[witness, neg] -= effect
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    # ---- embedding channel: disjoint coordinate blocks --------------------
    emb = rng.normal(0.0, 1.0, size=(n_tiles, embed_dim))
    for factor, slot in EMBED_SLOTS.items():
        if z_patient[factor] == 1:
            lo = slot * EMBED_BLOCK
            hi = min(lo + EMBED_BLOCK, embed_dim)
            emb[np.ix_(witness, np.arange(lo, hi))] += effect

    return probs.astype(np.float32), emb.astype(np.float32)


def calibrate(config: str, args: argparse.Namespace) -> None:
    """Cheap bag-mean linear read of each channel, before spending GPU time.

    This is deliberately weaker than the real aggregator: it bounds the signal
    from below and tells you whether the effect size lands the single-channel
    AUROCs near the operating point of the real cohort (~.55-.70).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    z = draw_latents(rng, args.n_patients)
    y = labels_from(config, z, rng, args.label_noise)

    feats_p, feats_e = [], []
    for i in range(args.n_patients):
        zp = {k: int(v[i]) for k, v in z.items()}
        n_tiles = int(rng.integers(args.min_tiles, args.max_tiles))
        probs, emb = build_bag(rng, zp, n_tiles, args.embed_dim,
                               args.effect, args.witness_rate)
        feats_p.append(probs.mean(axis=0))
        feats_e.append(emb.mean(axis=0))
    P = np.asarray(feats_p)
    E = np.asarray(feats_e)

    print(f"\nconfig {config}   n={args.n_patients}  effect={args.effect}"
          f"  witness={args.witness_rate}  noise={args.label_noise}")
    print(f"  atoms by construction: {CONFIGS[config]['atoms']}")
    print(f"  expected: {CONFIGS[config]['expected_regime']}")
    print(f"  prevalence: {y.mean():.3f}")
    for name, X in (("P (vocabulary)", P), ("E (embedding)", E),
                    ("concat", np.hstack([P, E]))):
        # Scale and regularise, or the 512 noise coordinates of E swamp the six
        # informative columns of P whenever the two are concatenated.
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.05))
        try:
            pred = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
            print(f"  bag-mean linear AUROC, {name:<16} {roc_auc_score(y, pred):.3f}")
        except ValueError as exc:
            print(f"  bag-mean linear AUROC, {name:<16} failed: {exc}")
    print("  NOTE: a linear read of the bag mean cannot express XOR, so config S")
    print("  is expected at chance here even though the synergy is present. That")
    print("  is the point of the config, and the Choquet head is what must find it.")


def write_cohort(config: str, args: argparse.Namespace) -> None:
    out = Path(args.out)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    z = draw_latents(rng, args.n_patients)
    y = labels_from(config, z, rng, args.label_noise)

    rows = []
    for i in range(args.n_patients):
        site = int(rng.integers(0, args.n_sites))
        slide_id = make_slide_id(i, site)
        zp = {k: int(v[i]) for k, v in z.items()}
        n_tiles = int(rng.integers(args.min_tiles, args.max_tiles))
        probs, emb = build_bag(rng, zp, n_tiles, args.embed_dim,
                               args.effect, args.witness_rate)

        slide_dir = out / "slides" / slide_id
        slide_dir.mkdir(exist_ok=True)
        np.save(slide_dir / "pattern_probs.npy", probs)
        np.save(slide_dir / "pattern_labels.npy",
                probs.argmax(axis=1).astype(np.int64))
        np.save(slide_dir / "embeddings.npy", emb)

        rows.append((slide_id, f"TCGA-{site:02d}-{i:04d}", f"{site:02d}",
                     int(y[i]), n_tiles))

    header = f"slide_id,patient_id,site,{GENE},n_tiles\n"
    body = "".join(f"{a},{b},{c},{d},{e}\n" for a, b, c, d, e in rows)
    (out / "labels.csv").write_text(header + body)

    manifest = {
        "config": config,
        "atoms_by_construction": CONFIGS[config]["atoms"],
        "drivers": CONFIGS[config]["drivers"],
        "expected_regime": CONFIGS[config]["expected_regime"],
        "n_patients": args.n_patients,
        "n_sites": args.n_sites,
        "embed_dim": args.embed_dim,
        "effect": args.effect,
        "witness_rate": args.witness_rate,
        "label_noise": args.label_noise,
        "tiles_per_bag": [args.min_tiles, args.max_tiles],
        "seed": args.seed,
        "prevalence": float(y.mean()),
        "channel_disjointness": {
            "embedding_encodes": sorted(EMBED_SLOTS),
            "vocabulary_encodes": sorted(PATTERN_SLOTS),
        },
    }
    (out / "ground_truth.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.n_patients} bags to {out}")
    print(f"  prevalence {y.mean():.3f}   atoms {CONFIGS[config]['atoms']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS),
                   help="which atom configuration to instantiate")
    p.add_argument("--out", help="output cohort directory (omit with --calibrate)")
    p.add_argument("--n-patients", type=int, default=444)
    p.add_argument("--n-sites", type=int, default=20)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--effect", type=float, default=0.6,
                   help="per-tile shift in units of channel noise SD")
    p.add_argument("--witness-rate", type=float, default=0.25,
                   help="fraction of tiles carrying the signal")
    p.add_argument("--label-noise", type=float, default=0.15)
    p.add_argument("--min-tiles", type=int, default=200)
    p.add_argument("--max-tiles", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrate", action="store_true",
                   help="report achievable AUROC without writing the cohort")
    args = p.parse_args()

    if args.calibrate:
        calibrate(args.config, args)
        return
    if not args.out:
        p.error("--out is required unless --calibrate is given")
    write_cohort(args.config, args)


if __name__ == "__main__":
    main()
