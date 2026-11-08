import os, re, csv
R = os.environ.get("SLIDES", "/home/rapids/notebooks/slima/data/slides")
ok = [d for d in os.listdir(R)
      if os.path.exists(os.path.join(R, d, "embeddings.npy"))
      and os.path.exists(os.path.join(R, d, "pattern_probs.npy"))]
bc = lambda s: re.search(r"TCGA-\w{2}-\w{4}-\d\d\w?(?:-\w+)*", s).group(0)
st = lambda s: re.search(r"TCGA-\w{2}-\w{4}-(\d\d)", s).group(1)
pt = lambda s: s[:12]
idx = {bc(d): d for d in ok}
coh = [l.strip() for l in open("luad_cohort_slides.txt") if l.strip()]
pos = [idx[bc(s)] for s in coh if bc(s) in idx]
neg = [d for d in ok if st(d) == "11"]
print("cohorte %d -> resueltos %d | normales %d" % (len(coh), len(pos), len(neg)))
print("pacientes: tumor %d, normal %d, ambos %d" % (
    len({pt(d) for d in pos}), len({pt(d) for d in neg}),
    len({pt(d) for d in pos} & {pt(d) for d in neg})))
os.makedirs("data_tumor_control", exist_ok=True)
if not os.path.exists("data_tumor_control/slides"):
    try: os.symlink(R, "data_tumor_control/slides")
    except OSError as e: print("enlace manual necesario: ln -s %s data_tumor_control/slides" % R)
with open("data_tumor_control/labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["slide_id", "TUMOR"])
    for d in pos: w.writerow([d, 1])
    for d in neg: w.writerow([d, 0])
open("data_tumor_control/tumor_control_slides.txt", "w").write("\n".join(pos + neg) + "\n")
print("escrito data_tumor_control/: %d filas (%d tumor / %d normal)" % (
    len(pos) + len(neg), len(pos), len(neg)))
