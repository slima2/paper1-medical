"""Real 12-lead ECG panel from PTB-XL, plotted in the clinical 3x4 + rhythm-strip
format on a standard ECG grid (25 mm/s, 10 mm/mV).

This is the electrocardiographic counterpart of the real LUAD histopathology
panels: it shows an actual record streamed from PhysioNet PTB-XL (CC-BY 4.0),
the raw 100 Hz signal the encoder E ingests in run_ecg_domainC.py, together with
the age/sex the framework recovers from exactly this signal.

Output (in this folder):
    ecg_ptbxl_example.png

Requires internet on first run (streams one record + the metadata CSV from
PhysioNet); both are cached under figures/_ptbxl_cache/ afterwards.
"""
import ast
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_ptbxl_cache"
CACHE.mkdir(exist_ok=True)
PN_ROOT = "ptb-xl/1.0.3"
DB_URL = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"

# 3x4 clinical layout: (row, col) -> lead. Columns are consecutive 2.5 s windows.
GRID = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]
LEAD_LABEL = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}   # clinical display form
COL_SECONDS = 2.5           # each column shows a 2.5 s window
MM_PER_S = 25.0             # standard paper speed
MM_PER_MV = 10.0           # standard gain
LANE_MM = 40.0             # vertical space per lead row (mm)


def load_metadata() -> pd.DataFrame:
    local = CACHE / "ptbxl_database.csv"
    if not local.exists():
        print("downloading ptbxl_database.csv ...")
        pd.read_csv(DB_URL).to_csv(local, index=False)
    return pd.read_csv(local)


def pick_record(df: pd.DataFrame) -> pd.Series:
    """A clean, illustrative NORMAL record: normal ECG, sinus rhythm, high quality."""
    df = df.copy()
    df["scp"] = df["scp_codes"].apply(ast.literal_eval)
    def is_norm(d):
        return d.get("NORM", 0) >= 80 and ("SR" in d)      # normal ECG + sinus rhythm
    cand = df[df["scp"].apply(is_norm)]
    # prefer records without noted artefacts (baseline_drift/static_noise/burst empty)
    for col in ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems"):
        if col in cand.columns:
            cand = cand[cand[col].isna()]
    cand = cand[cand["age"].notna() & (cand["age"] <= 90)]
    cand = cand.sort_values("ecg_id")
    return cand.iloc[0]


def load_signal(filename_lr: str) -> np.ndarray:
    """Stream the 100 Hz record; return [n_samples, 12] in mV, lead order = header."""
    rel = filename_lr.replace("\\", "/")            # e.g. records100/00000/00001_lr
    pn_dir = f"{PN_ROOT}/{os.path.dirname(rel)}"
    rec_name = os.path.basename(rel)
    sig, fields = wfdb.rdsamp(rec_name, pn_dir=pn_dir)
    return sig, fields["sig_name"], fields["fs"]


def draw_grid(ax, x_mm, y_mm):
    """Standard ECG grid: 1 mm minor (light) and 5 mm major (darker)."""
    minor, major = "#f6d4d4", "#e79a9a"
    for x in np.arange(0, x_mm + 1e-6, 1.0):
        ax.plot([x, x], [0, y_mm], color=minor, lw=0.3, zorder=1)
    for y in np.arange(0, y_mm + 1e-6, 1.0):
        ax.plot([0, x_mm], [y, y], color=minor, lw=0.3, zorder=1)
    for x in np.arange(0, x_mm + 1e-6, 5.0):
        ax.plot([x, x], [0, y_mm], color=major, lw=0.6, zorder=1)
    for y in np.arange(0, y_mm + 1e-6, 5.0):
        ax.plot([0, x_mm], [y, y], color=major, lw=0.6, zorder=1)


def main():
    df = load_metadata()
    rec = pick_record(df)
    ecg_id = int(rec["ecg_id"])
    age = int(rec["age"]); sex = "female" if int(rec["sex"]) == 1 else "male"
    report = str(rec.get("report", "")).strip()
    print(f"chosen record ecg_id={ecg_id}  age={age}  sex={sex}")
    print(f"report: {report}")

    sig, leads, fs = load_signal(rec["filename_lr"])
    lead_idx = {ln.upper(): i for i, ln in enumerate(leads)}
    n = sig.shape[0]
    t = np.arange(n) / fs                                    # seconds, 0..10

    ncol = len(GRID[0]); nrow = len(GRID)
    total_x_mm = ncol * COL_SECONDS * MM_PER_S               # 250 mm
    rhythm_h = LANE_MM
    total_y_mm = nrow * LANE_MM + rhythm_h + 6

    fig_w_in = total_x_mm / 25.4 + 0.6
    fig_h_in = total_y_mm / 25.4 + 0.6
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(-3, total_x_mm + 3)
    ax.set_ylim(-3, total_y_mm + 3)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_grid(ax, total_x_mm, total_y_mm)

    def plot_segment(lead, t0, t1, x0_mm, baseline_mm, lw=0.9):
        i = lead_idx[lead]
        m = (t >= t0) & (t < t1 + 1e-9)
        x = x0_mm + (t[m] - t0) * MM_PER_S
        y = baseline_mm + sig[m, i] * MM_PER_MV
        ax.plot(x, y, color="#111827", lw=lw, zorder=3)
        ax.text(x0_mm + 1.5, baseline_mm + 12, LEAD_LABEL.get(lead, lead),
                fontsize=9, fontweight="bold", color="#111827", zorder=4)

    # 3x4 leads (each a 2.5 s window), rows from top
    for r, row in enumerate(GRID):
        baseline = total_y_mm - (r + 0.5) * LANE_MM
        for c, lead in enumerate(row):
            t0 = c * COL_SECONDS
            x0 = c * COL_SECONDS * MM_PER_S
            plot_segment(lead, t0, t0 + COL_SECONDS, x0, baseline)
            # column separator ticks (thin)
            if c > 0:
                ax.plot([x0, x0], [baseline - 15, baseline + 15],
                        color="#9ca3af", lw=0.5, zorder=2)

    # rhythm strip: full 10 s of lead II
    rb = rhythm_h * 0.5
    plot_segment("II", 0, 10, 0, rb, lw=0.9)
    ax.text(1.5, rb + 15, "II  (rhythm strip, 10 s)", fontsize=9,
            fontweight="bold", color="#111827")

    # calibration pulse (1 mV = 10 mm) at far left of the top row
    top_base = total_y_mm - 0.5 * LANE_MM
    cx = -2.0
    ax.plot([cx, cx, cx + 5, cx + 5, cx + 10],
            [top_base, top_base + 10, top_base + 10, top_base, top_base],
            color="#111827", lw=1.0, zorder=3)

    ax.text(total_x_mm, -1.5, "25 mm/s   10 mm/mV   100 Hz",
            fontsize=8.5, ha="right", va="top", color="#374151")

    out = HERE / "ecg_ptbxl_example.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    # emit caption-ready metadata
    meta = CACHE / "chosen_record.txt"
    meta.write_text(f"ecg_id={ecg_id}\nage={age}\nsex={sex}\nreport={report}\n"
                    f"filename_lr={rec['filename_lr']}\n", encoding="utf-8")
    print(f"metadata -> {meta}")


if __name__ == "__main__":
    main()
