"""Aggregate results/metrics_long.csv into tables for the report.

Writes
  results/summary.csv            mean of every metric per (method, scope)
  results/summary.md             readable markdown version
  paper/tables/baseline.tex      LaTeX table, drop-in for Overleaf

Scales follow PIE-Bench Table 1: Structure x1e3, LPIPS x1e3, MSE x1e4, SSIM x1e2.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# column order, display name, scale, better direction
COLS = [
    ("structure_distance", r"Struct.$\downarrow$", 1e3, "min"),
    ("psnr_unedit_part", r"PSNR$\uparrow$", 1, "max"),
    ("lpips_unedit_part", r"LPIPS$\downarrow$", 1e3, "min"),
    ("mse_unedit_part", r"MSE$\downarrow$", 1e4, "min"),
    ("ssim_unedit_part", r"SSIM$\uparrow$", 1e2, "max"),
    ("clip_similarity_target_image", r"CLIP-W$\uparrow$", 1, "max"),
    ("clip_similarity_target_image_edit_part", r"CLIP-E$\uparrow$", 1, "max"),
]

# Published DDIM+MasaCtrl numbers on all 700 PIE-Bench images (Ju et al., ICLR 2024, Table 1).
PUBLISHED = {
    "DDIM+MasaCtrl (PIE-Bench paper)": [28.38, 22.17, 106.62, 86.97, 79.67, 23.96, 21.16],
    "DirectInv.+MasaCtrl (PIE-Bench paper)": [24.70, 22.64, 87.94, 81.09, 81.33, 24.38, 21.35],
}

PRETTY = {
    "fixed_seed": "SD fixed seed (no control)",
    "ddim+masactrl": "DDIM + MasaCtrl",
    "directinversion+masactrl": "DirectInv. + MasaCtrl",
}

CATEGORY_NAMES = {"5": "change pose"}


def runtime_table(output_path: Path, methods):
    rows = {}
    for m in methods:
        f = output_path / m / "runtime.jsonl"
        if f.exists():
            s = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
            rows[m] = s["seconds"].mean()
    return rows


def to_latex(table: pd.DataFrame, caption: str, label: str, published: bool) -> str:
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
             " & ".join(["Method"] + [c[1] for c in COLS]) + r" \\", r"\midrule"]
    best = {}
    for key, _, _, d in COLS:
        if key in table.columns:
            best[key] = table[key].min() if d == "min" else table[key].max()
    for method, row in table.iterrows():
        cells = [PRETTY.get(method, method).replace("_", r"\_")]
        for key, _, _, _ in COLS:
            v = row.get(key)
            if v is None or pd.isna(v):
                cells.append("--")
            else:
                s = f"{v:.2f}"
                cells.append(r"\textbf{" + s + "}" if abs(v - best[key]) < 1e-9 else s)
        lines.append(" & ".join(cells) + r" \\")
    if published:
        lines.append(r"\midrule")
        for name, vals in PUBLISHED.items():
            lines.append(" & ".join([r"\textit{" + name + "}"] + [f"{v:.2f}" for v in vals]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_csv", default="results/metrics_long.csv")
    p.add_argument("--output_path", default="outputs")
    p.add_argument("--tex", default=str(ROOT / "paper" / "tables" / "baseline.tex"))
    args = p.parse_args()

    df = pd.read_csv(args.result_csv, dtype={"category": str})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    scale = {k: s for k, _, s, _ in COLS}
    df["value"] = df.apply(lambda r: r["value"] * scale.get(r["metric"], 1), axis=1)

    scopes = {"all": df}
    for c in sorted(df["category"].unique()):
        scopes[f"cat{c}"] = df[df["category"] == c]

    out_rows, md = [], []
    tables = {}
    for scope, d in scopes.items():
        t = d.pivot_table(index="method", columns="metric", values="value", aggfunc="mean")
        t = t[[c for c, *_ in COLS if c in t.columns]]
        order = [m for m in PRETTY if m in t.index] + sorted(m for m in t.index if m not in PRETTY)
        t = t.loc[order]
        n = d.drop_duplicates(["id"]).shape[0]
        tables[scope] = (t, n)
        for m, row in t.iterrows():
            out_rows.append({"scope": scope, "n_images": n, "method": m, **row.to_dict()})
        md.append(f"### {scope}  (n = {n} images)\n\n" + t.round(2).to_markdown() + "\n")

    rt = runtime_table(Path(args.output_path), df["method"].unique())
    if rt:
        md.append("### mean runtime per image (s)\n\n" + pd.Series(rt, name="seconds").round(1).to_markdown() + "\n")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_csv("results/summary.csv", index=False)
    Path("results/summary.md").write_text("\n".join(md))

    # LaTeX: whole benchmark (+ published reference rows when all 10 categories are present)
    t_all, n_all = tables["all"]
    full = n_all == 700  # published reference numbers only make sense on the full benchmark
    tex = to_latex(t_all, f"Baseline results on PIE-Bench ({n_all} images). "
                          r"Struct.\ $\times10^3$, LPIPS $\times10^3$, MSE $\times10^4$, SSIM $\times10^2$; "
                          r"PSNR/LPIPS/MSE/SSIM computed on the unedited region, CLIP-W/E on the whole/edited region.",
                   "tab:baseline", published=full)
    if "cat5" in tables:
        t5, n5 = tables["cat5"]
        tex += "\n" + to_latex(t5, f"Baseline results on the PIE-Bench \\emph{{change pose}} subset ({n5} images), "
                                    "the non-rigid setting MasaCtrl targets.", "tab:baseline_pose", published=False)
    Path(args.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tex).write_text(tex)
    print("\n".join(md))
    print("wrote results/summary.csv, results/summary.md,", args.tex)


if __name__ == "__main__":
    main()
