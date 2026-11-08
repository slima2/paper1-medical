import glob, re, sys, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "results_controls"
F = sorted(x.replace("\\", "/") for x in glob.glob(R + "/*/oof/oof_*.csv"))
print(len(F), "ficheros OOF")
c0 = list(pd.read_csv(F[0], nrows=3).columns)
print("columnas:", c0)
def pick(h, ex):
    for k in h:
        for c in c0:
            if c not in ex and k == c.lower(): return c
    for k in h:
        for c in c0:
            if c not in ex and k in c.lower(): return c
L = pick(["label", "y_true", "true", "target", "gt", "y"], [])
P = pick(["prob", "score", "y_pred", "pred", "logit"], [L])
I = pick(["patient", "case", "submitter", "slide", "id"], [L, P])
print("id=%s  etiqueta=%s  prob=%s\n" % (I, L, P))
rx = re.compile(r"oof_(.+?)_([A-Z0-9]+)_fold\d+\.csv$")
D = {}
for f in F:
    m = rx.search(f.split("/")[-1])
    if not m: continue
    pr, sd = f.split("/")[-3].split("_seed")
    d = pd.read_csv(f)[[x for x in (I, L, P) if x]]
    d.columns = (["id", "y", "p"] if I else ["y", "p"])
    D.setdefault((pr, int(sd), m.group(1).split("_")[0], m.group(2)), []).append(d)
def auc(d):
    r = d.p.rank(); pos = d.y == 1; a, b = pos.sum(), (~pos).sum()
    return (r[pos].sum() - a * (a + 1) / 2) / (a * b)
rows = []
for k, v in D.items():
    d = pd.concat(v)
    if I: d = d.groupby("id", as_index=False).agg({"y": "max", "p": "mean"})
    rows.append(dict(proto=k[0], seed=k[1], cond=k[2], target=k[3],
                     n=len(d), npos=int(d.y.sum()), auroc=auc(d)))
r = pd.DataFrame(rows)
CIT = {("baseline2", "TP53SHUF"): .515, ("baseline3", "TP53SHUF"): .482,
       ("baseline2", "SEXF"): .588, ("baseline2", "TP53"): .647}
for (pr, cd, tg), g in r.groupby(["proto", "cond", "target"]):
    per = "  ".join("s%d=%.3f" % (x.seed, x.auroc)
                    for x in g.sort_values("seed").itertuples())
    m, s = g.auroc.mean(), g.auroc.std()
    c = CIT.get((cd, tg)); t = ""
    if c and pr == "patient":
        t = "   paper %.3f -> %+.3f %s" % (
            c, m - c, "OK" if abs(m - c) < .005 else "CAMBIA")
    print("%-8s %-10s %-9s n=%d n+=%d | %s | %.3f +-%.3f%s"
          % (pr, cd, tg, g.n.iloc[0], g.npos.iloc[0], per, m, s, t))
sx = r[(r.proto == "patient") & (r.cond == "baseline2")
       & (r.target == "SEXF")].auroc.mean()
print("\nControl sexo B2 paciente = %.3f | EGFR = 0.582 | EGFR-control = %+.3f"
      % (sx, .582 - sx))
print("EGFR POR DEBAJO del control -> el texto de Resultados se mantiene"
      if .582 < sx else
      "ATENCION: EGFR POR ENCIMA -> hay que invertir la frase y revisar el 0.59")
r.to_csv("controls_recomputed.csv", index=False)
print("escrito controls_recomputed.csv")
