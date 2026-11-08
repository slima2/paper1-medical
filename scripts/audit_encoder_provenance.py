#!/usr/bin/env python3
"""What is actually inside the embedding channel?

The manuscript describes E as the frozen CTransPath encoder, published
checkpoint, and rests the redundancy reading on CTransPath being an encoder
that happens to represent the ANORAK vocabulary. extract_embeddings.py instead
builds CTransPath + a 512-d projection and then calls

    model.load_state_dict(best_fuzzyarcloss_v2_weights, strict=False)

on the whole Sequential. Because strict=False drops non-matching keys in
silence, the extent of the overwrite is not readable from the source: either
only the projection was loaded, leaving the backbone as published, or the
backbone was replaced by weights fine-tuned on the six growth patterns. The
first is a wording problem in the Methods. The second means the embedding
channel was trained on the vocabulary it is later found to make redundant,
which is a different and much larger problem.

This script answers which, without re-extracting anything.

    python scripts/audit_encoder_provenance.py \
        --model-ckpt /home/rapids/notebooks/slima/outputs/ablation_study_v16_optuna/best_fuzzyarcloss_v2.pth \
        --ctranspath-ckpt /home/rapids/notebooks/slima/models/ctranspath.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """Verbatim from extract_embeddings.py."""

    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        d1, d2 = embed_dim // 8, embed_dim // 4
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, d1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d1), nn.GELU(),
            nn.Conv2d(d1, d2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d2), nn.GELU(),
            nn.Conv2d(d2, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


def build(in_ch: int, embed: int):
    import timm
    m = timm.create_model("swin_tiny_patch4_window7_224",
                          pretrained=False, num_classes=0)
    m.patch_embed = ConvStem(in_chans=3, embed_dim=96)
    if in_ch != 3:
        old = m.patch_embed.proj[0]
        m.patch_embed.proj[0] = nn.Conv2d(in_ch, old.out_channels,
                                          old.kernel_size, old.stride,
                                          old.padding, bias=False)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = m
            self.out_features = m.num_features

        def forward(self, x):
            return self.model(x)

    proj = nn.Sequential(nn.Linear(m.num_features, embed),
                         nn.LayerNorm(embed), nn.GELU(), nn.Dropout(0.0))
    return nn.Sequential(Backbone(), proj)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-ckpt", type=Path, required=True)
    p.add_argument("--ctranspath-ckpt", type=Path)
    args = p.parse_args()

    ckpt = torch.load(str(args.model_ckpt), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    print(f"checkpoint: {args.model_ckpt}")
    print(f"top-level keys: {list(ckpt)[:12] if isinstance(ckpt, dict) else type(ckpt)}")
    print("\nconfig recorded in the checkpoint:")
    for k in sorted(cfg):
        print(f"  {k} = {cfg[k]}")

    embed = int(cfg.get("EMBED_DIM", 512))
    in_ch = 4 if cfg.get("USE_MASK_AS_CHANNEL", True) else 3

    raw = ckpt.get("model", ckpt.get("model_state", {}))
    raw = {k.replace("module.", ""): v for k, v in raw.items()}
    print(f"\n{len(raw)} tensors in the checkpoint state dict")
    print("first 8 keys:")
    for k in list(raw)[:8]:
        print(f"  {k}  {tuple(raw[k].shape)}")

    model = build(in_ch, embed)
    target = model.state_dict()

    matched = {k: v for k, v in raw.items()
               if k in target and v.shape == target[k].shape}
    backbone_hits = [k for k in matched if k.startswith("0.")]
    proj_hits = [k for k in matched if k.startswith("1.")]

    print(f"\nmodel expects {len(target)} tensors")
    print(f"matched      {len(matched)}")
    print(f"  backbone (prefix '0.')   {len(backbone_hits)}")
    print(f"  projection (prefix '1.') {len(proj_hits)}")

    print("\n" + "=" * 70)
    if backbone_hits:
        print("VERDICT: the backbone WAS overwritten by the fine-tuned weights.")
        print("E is CTransPath fine-tuned on the ANORAK growth patterns, not the")
        print("published frozen encoder. Every Methods statement calling it frozen")
        print("or 'as published' is wrong, and the redundancy reading needs the")
        print("circularity stated: E was trained to make P decodable.")
    elif proj_hits:
        print("VERDICT: only the projection loaded; the backbone stayed as published.")
        print("E is frozen CTransPath followed by a 512-d head trained on ANORAK.")
        print("Weaker than the above but still not 'the frozen encoder': the")
        print("projection is pattern-supervised and that must be stated.")
    else:
        print("VERDICT: nothing matched. strict=False swallowed the entire load and")
        print("E is the randomly initialised projection over frozen CTransPath.")
        print("That would make the embedding channel a random 512-d compression,")
        print("which is its own serious problem and would explain weak AUROCs.")
    print("=" * 70)

    if args.ctranspath_ckpt and args.ctranspath_ckpt.exists():
        pub = torch.load(str(args.ctranspath_ckpt), map_location="cpu",
                         weights_only=False)
        pub = pub.get("model", pub.get("state_dict", pub))
        shared = [k for k in raw if k.replace("0.model.", "") in pub]
        if shared:
            k = shared[0]
            a = raw[k]
            b = pub[k.replace("0.model.", "")]
            if a.shape == b.shape:
                delta = (a.float() - b.float()).abs().mean().item()
                print(f"\nmean |fine-tuned - published| on {k}: {delta:.6f}")
                print("  ~0 means the backbone is unchanged from the published"
                      " checkpoint.")


if __name__ == "__main__":
    main()
