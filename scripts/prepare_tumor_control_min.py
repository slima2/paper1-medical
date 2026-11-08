import sys, os, re, glob, math, collections
inv, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
TUM = {"01", "02", "05", "06", "07"}
NOR = {"10", "11", "12", "14"}
BC = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z]?")
def P(s):
    m = BC.search(s)
    return (m.group(1), m.group(2)) if m else None
ids = [l.strip() for l in open(inv) if l.strip()]
print(len(ids), "portaobjetos en el inventario\n")
by = collections.defaultdict(list)
for s in ids:
    p = P(s)
    by[p[1] if p else "??"].append(s)
for c in sorted(by):
    arm = "tumor" if c in TUM else ("normal" if c in NOR else "?")
    print("   tipo %s  %-7s %5d" % (c, arm, len(by[c])))
tum = [s for c in TUM for s in by.get(c, [])]
nor = [s for c in NOR for s in by.get(c, [])]
print("\n   tumor %d   normal %d" % (len(tum), len(nor)))
if not nor:
    sys.exit("\nPARADA: este inventario no trae normales, ya viene filtrado.\n"
             "Apunta --inventory a la lista previa al filtro (687) o al\n"
             "manifiesto del GDC.")
have = set()
for f in glob.glob(os.path.join(data, "**", "*"), recursive=True):
    if os.path.splitext(f)[1].lower() in (".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5"):
        have.add(os.path.splitext(os.path.basename(f))[0])
def cov(L):
    o = []
    for s in L:
        if s in have:
            o.append(s)
        else:
            h = [x for x in have if s in x or x in s]
            if h:
                o.append(h[0])
    return o
t2, n2 = cov(tum), cov(nor)
print("\ncaracteristicas: %d ficheros | tumor %d/%d | normal %d/%d"
      % (len(have), len(t2), len(tum), len(n2), len(nor)))
if len(n2) < max(10, 0.5 * len(nor)):
    print("   AVISO: faltan caracteristicas para la mayoria de los normales.")
    print("   Este control exige extraerlas primero; ya no es una tarde de GPU.")
pt = {P(s)[0] for s in t2}
pn = {P(s)[0] for s in n2}
both = pt & pn
print("\npacientes: solo tumor %d | solo normal %d | AMBOS %d"
      % (len(pt - pn), len(pn - pt), len(both)))
if both:
    print("   Agregar a paciente asignaria a esos %d una etiqueta que no" % len(both))
    print("   existe. Puntuar a NIVEL DE PORTAOBJETOS, manteniendo el")
    print("   agrupamiento por paciente en los pliegues para evitar fuga.")
def se(a, p, n):
    q1 = a / (2 - a); q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (p - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (p * n))
if t2 and n2:
    print("\npotencia a nivel de portaobjetos (n+=%d, n-=%d):" % (len(t2), len(n2)))
    for a in (.90, .95, .99):
        h = 1.96 * se(a, len(t2), len(n2))
        print("   si el AUROC real es %.2f -> IC95 +-%.3f, cota inferior %.3f" % (a, h, a - h))
    print("   sirve como control positivo solo si esa cota inferior supera")
    print("   claramente 0.67, que es el techo del brazo principal.")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "labels.csv"), "w") as f:
    f.write("slide_id,patient_id,TUMOR\n")
    for s in t2 + n2:
        p = P(s)
        f.write("%s,%s,%d\n" % (s, p[0], 1 if p[1] in TUM else 0))
open(os.path.join(out, "tumor_control_slides.txt"), "w").write("\n".join(t2 + n2) + "\n")
print("\nescrito %s/labels.csv (%d filas) y tumor_control_slides.txt"
      % (out, len(t2) + len(n2)))
