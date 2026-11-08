"""Count the Communications Medicine abstract, excluding the keyword list that follows it.

texcount reports the whole file, so it folds the twelve keywords into the
total and reads 176 against a 150-word limit. Only the abstract environment
counts. Run from paper1_medical/.
"""
import re

src = open("sections/01_abstract_CM.tex", encoding="utf-8").read()
src = re.sub(r"(?m)^\s*%.*$", "", src)
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
if not m:
    raise SystemExit("no encuentro el entorno abstract")
body = m.group(1)
body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)          # macros -> space
body = re.sub(r"[{}$\\~]", " ", body)
words = [w for w in re.split(r"\s+", body) if re.search(r"[A-Za-z0-9]", w)]
print(f"Resumen: {len(words)} palabras   (rango Communications Medicine 150-250)")
print("OK" if 150 <= len(words) <= 250 else "FUERA DE RANGO")
print()
print(" ".join(words))
