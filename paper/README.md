# Report (Overleaf)

Course requirement: **AAAI format, max 6 pages excluding references**, LaTeX preferred.

## Set up the Overleaf project (once, by one person)

1. Download the **AAAI-26 Author Kit** from aaai.org (Author Kit page) and take two files from it:
   `aaai2026.sty` and `aaai2026.bst`. (The Overleaf gallery also has the AAAI template — you can
   download those two files from there.)
2. Zip this `paper/` folder, then in Overleaf: **New Project → Upload Project** → select the zip.
3. Upload `aaai2026.sty` and `aaai2026.bst` into the project root (next to `main.tex`).
4. Menu → Settings: Compiler **pdfLaTeX**, Main document **main.tex**.
5. **Share** → add the 5 teammates as editors.

Do not add `\bibliographystyle` — the AAAI style file sets it. Avoid packages AAAI forbids
(e.g. `hyperref`, `geometry`, `fullpage`, changing margins/fonts).

## Who writes what

| file | content | owner |
|---|---|---|
| `sections/abstract.tex` | written last | |
| `sections/introduction.tex` | motivation, contributions | |
| `sections/related.tex` | editing, inversion, attention control | |
| `sections/background.tex` | precise MasaCtrl description (equations already in) | |
| `sections/method.tex` | our improvement | |
| `sections/experiments.tex` | setup (drafted), results, ablation | |
| `sections/conclusion.tex` | | |

`tables/baseline.tex` is **generated** by `python src/summarize.py` — after a new run, replace the
file in Overleaf (don't hand-edit numbers). Figures go in `figures/`.

Red `\todo{}` markers show what's missing; remove them before submission
(`\renewcommand{\todo}[1]{}` hides them all for a page-count check).
