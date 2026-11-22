"""Score the Google Form export against the answer key.

Usage:  python3 user_study/analyze.py  path/to/responses.csv
Form layout expected: one "multiple choice grid" per question titled "Q1" ... "Q16",
rows = the three dimension questions, columns = A / B / 差不多.
"""
import csv, re, sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMS = [("Image quality", ("清晰", "自然", "quality")),
        ("Prompt fidelity", ("修改要求", "符合", "prompt", "instruction")),
        ("Structure preservation", ("主体", "保留", "structure", "preserv"))]

key = {int(r["question"]): r for r in csv.DictReader(open(HERE / "answer_key_DO_NOT_SHARE.csv"))}
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
counts = {d: {"ours": 0, "baseline": 0, "tie": 0} for d, _ in DIMS}
for col in rows[0]:
    m = re.search(r"Q\s*0*(\d+)", col)
    br = re.search(r"\[(.*)\]", col)
    if not m or not br or int(m.group(1)) not in key:
        continue
    dim = next((d for d, kws in DIMS if any(k.lower() in br.group(1).lower() for k in kws)), None)
    if dim is None:
        continue
    k = key[int(m.group(1))]
    for r in rows:
        a = r[col].strip()
        if a in ("A", "B"):
            counts[dim]["ours" if k[a] == "improvement1" else "baseline"] += 1
        elif a:
            counts[dim]["tie"] += 1

def sign_test(w, l):  # two-sided exact binomial test, ties excluded
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

print(f"Raters: {len(rows)}   Questions: {len(key)}\n")
print(f"| Criterion | Ours | MasaCtrl | Tie | p (sign test) |\n|---|---|---|---|---|")
for d, _ in DIMS:
    c = counts[d]; n = sum(c.values()) or 1
    print(f"| {d} | {100*c['ours']/n:.1f}% | {100*c['baseline']/n:.1f}% | {100*c['tie']/n:.1f}% | {sign_test(c['ours'], c['baseline']):.3g} |")
