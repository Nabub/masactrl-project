# AI6103 Group Project — Improving MasaCtrl

Course project for **AI6103 Deep Learning** (NTU). Topic option 2: *take a conference paper from 2019+ and
develop an improvement on top of its existing code base.*

Paper: **MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing**
(Cao et al., ICCV 2023) — [arXiv 2304.08465](https://arxiv.org/abs/2304.08465) ·
[official code](https://github.com/TencentARC/MasaCtrl)

Deliverables: report (AAAI format, ≤ 6 pages excl. references) + 5-min video. Report source is in `paper/`.

---

## Repository layout

```
masactrl-project/
├── src/
│   ├── run_baseline.py     # generate edits on PIE-Bench (or one image); add our method to METHODS here
│   ├── evaluate.py         # PIE-Bench metrics (structure dist., bg PSNR/LPIPS/MSE/SSIM, CLIP)
│   └── summarize.py        # -> results/summary.{csv,md} and paper/tables/baseline.tex
├── scripts/
│   ├── setup.sh            # clones official code at pinned commits into third_party/, pip install
│   └── run_baseline.sh     # smoke | pose | all
├── notebooks/baseline_colab.ipynb
├── paper/                  # Overleaf project (AAAI template)
├── results/                # small summary files (tracked)
├── third_party/            # (git-ignored) MasaCtrl, PnPInversion
├── data/                   # (git-ignored) PIE-Bench
└── outputs/                # (git-ignored) generated images
```

## Baseline protocol

| Item | Setting |
|---|---|
| Backbone | Stable Diffusion v1.4 (`CompVis/stable-diffusion-v1-4`), fp32 |
| Sampler | DDIM, 50 steps, CFG 7.5 |
| MasaCtrl | mutual self-attention from step **S = 4**, U-Net layer **L = 10** (paper default), source prompt = "" for real images |
| Benchmark | **PIE-Bench** (700 real images, 10 edit types; type 5 = *change pose*, the non-rigid case MasaCtrl targets) |
| Metrics | PIE-Bench's own code: structure distance (DINO), PSNR / LPIPS / MSE / SSIM on the unedited region, CLIP score (whole / edited region) |
| Seed | 1234, reset before every edit |

Methods run by default:

| key | what it is | why |
|---|---|---|
| `fixed_seed` | same inverted noise, target prompt, no attention control | lower bound ("Fixed Seed" column in the paper) |
| `ddim+masactrl` | official MasaCtrl | **the baseline we improve on**; full-benchmark numbers can be checked against PIE-Bench Table 1 (28.38 / 22.17 / 106.62 / 86.97 / 79.67 / 23.96 / 21.16) |
| `directinversion+masactrl` | MasaCtrl with a better inversion | stronger baseline — the course penalises "artificially lowered baselines" |

The MasaCtrl paper itself reports **no quantitative numbers**, which is why we evaluate on PIE-Bench.

## Quick start (Colab or GPU server, ≥ 16 GB VRAM)

```bash
git clone <this repo> && cd masactrl-project
bash scripts/setup.sh                 # pinned official code + deps
bash scripts/run_baseline.sh smoke    # 1 image -> outputs/smoke/corgi_grid.png
# get PIE-Bench (request form in the PnPInversion README) -> data/PIE-Bench/{mapping_file.json,annotation_images/}
bash scripts/run_baseline.sh pose     # 40 change-pose images, then metrics + LaTeX table
bash scripts/run_baseline.sh all      # all 700 images (A100 recommended; resumable)
```

Or open `notebooks/baseline_colab.ipynb` in Colab.

Useful flags: `--limit 5` (debug), `--start_step/--start_layer` (S, L ablation), `--methods ddim+masactrl`.

## Adding our improvement

1. Implement it as a new method on `BaselineEditor` in `src/run_baseline.py` (e.g. a subclass of
   `MutualSelfAttentionControl` in a new file `src/ours.py`).
2. Register it in `METHODS` (`"ours": BaselineEditor.ours`).
3. `METHODS="ddim+masactrl ours" bash scripts/run_baseline.sh pose` → same metrics, same table.

Keep everything else (seed, steps, CFG, S, L, model) identical so the comparison is fair.

## Team workflow

- `main` stays runnable; work on branches (`feat/<name>`), merge via PR.
- Don't commit images or checkpoints — results go to `results/summary.*`; pick figure examples into `paper/figures/`.
- Overleaf: see `paper/README.md`.
