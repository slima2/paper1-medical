#!/usr/bin/env python3
"""How much of the vocabulary channel is already inside the embedding channel?

The regime taxonomy asks whether an expert vocabulary P carries information an
opaque encoder E does not. That question is only askable if P is not a function
of E. The provenance audit showed the published E is CTransPath fine-tuned
end-to-end on the six ANORAK growth patterns, which is the objective that also
produces P, so the two channels may be the penultimate representation and the
softmax of one network. If so, P is a deterministic function of E, the data
processing inequality forces unique-to-P to zero, and the finding that no LUAD
target shows complementarity is an identity rather than a result.

Rather than tracing which checkpoint produced which file, this measures the
quantity that actually matters: fit a probe from E to the per-tile pattern call
and report how well it recovers it. The probe is trained and tested on disjoint
slides, so the number is generalisation, not memorisation.

    ~1.00   P is a readout of E. Unique-to-P is zero by construction and
            regime (ii) was unreachable on this cohort.
    ~0.90   Two fits of the same task. Unique-to-P is small but not zero.
    ~0.60   Genuinely distinct channels, as an independent encoder would give.

Run the same probe on each encoder arm to turn the confound into a measured
axis:

    python scripts/probe_channel_dependence.py --data-dir .../data          # anorak-tuned
    python scripts/probe_channel_dependence.py --data-dir .../data_ctp_raw  # published
    python scripts/probe_channel_dependence.py --data-dir .../data_uni2h    # independent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_pairs(data_dir: Path, n_slides: int, tiles_per_slide: int,
               seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns pooled (E, pattern_call, slide_index)."""
    rng = np.random.default_rng(seed)
    slides = sorted(
        d for d in (data_dir / "slides").iterdir()
        if (d / "embeddings.npy").exists() and (d / "pattern_probs.npy").exists()
    )
    if not slides:
        raise SystemExit(f"no slides with both channels under {data_dir}")
    rng.shuffle(slides)
    slides = slides[:n_slides]

    xs, ys, gs = [], [], []
    for gi, sd in enumerate(slides):
        emb = np.load(sd / "embeddings.npy", mmap_mode="r")
        probs = np.load(sd / "pattern_probs.npy", mmap_mode="r")
        n = min(emb.shape[0], probs.shape[0])
        if n == 0:
            continue
        take = min(tiles_per_slide, n)
        idx = np.sort(rng.choice(n, take, replace=False))
        xs.append(np.asarray(emb[idx], dtype=np.float32))
        ys.append(np.asarray(probs[idx], dtype=np.float32).argmax(axis=1))
        gs.append(np.full(take, gi))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(gs)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n-slides", type=int, default=60)
    p.add_argument("--tiles-per-slide", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, y, g = load_pairs(args.data_dir, args.n_slides,
                         args.tiles_per_slide, args.seed)
    print(f"cohort: {args.data_dir}")
    print(f"  {X.shape[0]} tiles from {g.max() + 1} slides, "
          f"embedding is {X.shape[1]}-d")
    counts = np.bincount(y, minlength=6)
    print(f"  pattern call distribution: {counts.tolist()}")
    majority = counts.max() / counts.sum()
    print(f"  majority-class baseline:   {majority:.3f}")

    # Split by slide so the probe cannot memorise a slide's tiles.
    n_groups = int(g.max()) + 1
    cut = int(n_groups * 0.7)
    tr, te = g < cut, g >= cut

    probe = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0,
                                             multi_class="multinomial"))
    probe.fit(X[tr], y[tr])
    pred = probe.predict(X[te])
    acc = accuracy_score(y[te], pred)
    f1 = f1_score(y[te], pred, average="macro")

    print(f"\n  linear probe E -> pattern call, held-out slides")
    print(f"    accuracy  {acc:.3f}")
    print(f"    macro-F1  {f1:.3f}")

    # Read the macro-F1, not the accuracy. The pattern call is heavily
    # imbalanced, so accuracy sits near the majority baseline by default and
    # barely discriminates between a readout and an unrelated channel.
    mf1_majority = (2 * majority / (1 + majority)) / 6
    print(f"    (majority-class predictor would score macro-F1 "
          f"{mf1_majority:.3f}, so read the macro-F1 and not the accuracy)")

    print("\n" + "=" * 70)
    if f1 >= 0.95:
        print("READ: the vocabulary channel is a readout of the embedding.")
        print("Unique-to-P is zero for practical purposes; regime (ii) could")
        print("not have been observed whatever the biology.")
    elif f1 >= 0.70:
        print("READ: two fits of the same objective. Most of the vocabulary is")
        print("linearly present in the embedding, so unique-to-P is small but")
        print("not identically zero. A null complementarity result here is")
        print("heavily biased toward the null rather than strictly forced, and")
        print("must be reported as such rather than as a finding about tissue.")
    else:
        print("READ: the channels are distinct enough for the question to be")
        print("askable. A null complementarity result is informative here.")
    print("=" * 70)
    print("\nReport this number for every encoder arm. The contrast between")
    print("arms is what converts the confound into a stated property of the")
    print("triple (vocabulary, encoder, cohort) instead of a hidden one.")


if __name__ == "__main__":
    main()
