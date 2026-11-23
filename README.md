# AI6103 Group Project — Improving MasaCtrl

Course project for **AI6103 Deep Learning** (NTU). Topic option 2: *take a conference paper from 2019+ and
develop an improvement on top of its existing code base.*

Paper: **MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing**
(Cao et al., ICCV 2023) — [arXiv 2304.08465](https://arxiv.org/abs/2304.08465) ·
[official code](https://github.com/TencentARC/MasaCtrl)

Our Group's Project Repository: [GitHub repository - masactrl-project](https://github.com/Nabub/masactrl-project)

Deliverables: report (AAAI format, ≤ 6 pages excl. references) + 5-min video. Report source is in `paper/`.

**Group Members (Name — Matriculation Number):**
- Yangyuan Li — G2605384E 
- Ziang Wang — G2608868E 
- Yiping Ding — G2604966A  
- Jinnan Yao — G2603383B  
- Yanan Bu — G2610370H  
- Jiayi Ma — G2610678A  

---

## Repository layout

```
masactrl-project/
├── src/
│   ├── run_baseline.py     # generate edits on PIE-Bench (or one image)
│   ├── yangyuan.py         # foreground attention masks and hybrid initialization
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
git clone https://github.com/Nabub/masactrl-project.git && cd masactrl-project
bash scripts/setup.sh                 # pinned official code + deps
bash scripts/run_baseline.sh smoke    # 1 image -> outputs/smoke/corgi_grid.png
# get PIE-Bench (request form in the PnPInversion README) -> data/PIE-Bench/{mapping_file.json,annotation_images/}
bash scripts/run_baseline.sh pose     # 40 change-pose images, then metrics + LaTeX table
bash scripts/run_baseline.sh all      # all 700 images (A100 recommended; resumable)
```

Or open `notebooks/baseline_colab.ipynb` in Colab.

Useful flags: `--limit 5` (debug), `--start_step/--start_layer` (S, L ablation), `--methods ddim+masactrl`.

## Yangyuan improvement

Yangyuan uses 16×16 cross-attention maps to locate the subject, restricts source feature
transfer to the foreground, and initializes the background with seeded noise
so it can follow the new prompt.

Run both DDIM + MasaCtrl and Yangyuan after setup.

```bash
# Corgi comparison → outputs/corgi/corgi_grid.png
python src/run_baseline.py --methods ddim+masactrl Yangyuan \
  --image third_party/MasaCtrl/gradio_app/images/corgi.jpg --image_resize nearest \
  --src_prompt 'a photo of a corgi lying down' \
  --tgt_prompt 'a photo of a running corgi on green grass, side view' \
  --steps 50 --guidance 3.0 --start_step 15 --start_layer 10 --seed 42 \
  --model_id "${MODEL_ID:-CompVis/stable-diffusion-v1-4}" --device "${DEVICE:-auto}" \
  --output_path outputs/corgi --rerun

# Two category 8 PIE-Bench edits → outputs/background-reproduction/
python src/run_baseline.py --methods ddim+masactrl Yangyuan \
  --data_path data/PIE-Bench --categories 8 --limit 2 \
  --subject_hints results/background_pilot_subject_hints.json \
  --steps 50 --guidance 7.5 --start_step 4 --start_layer 10 --seed 1234 \
  --model_id "${MODEL_ID:-CompVis/stable-diffusion-v1-4}" --device "${DEVICE:-auto}" \
  --output_path outputs/background-reproduction --rerun
```

[Pinned setup for reproducing the Apple Silicon pilot](results/README.md#apple-silicon-reproduction)
· [Score the background edits](results/README.md#score-the-background-pilot)
· [Original corgi run](results/README.md#original-corgi-comparison)

## Team workflow

- `main` stays runnable; work on branches (`feat/<name>`), merge via PR.
- Don't commit images or checkpoints — results go to `results/summary.*`; pick figure examples into `paper/figures/`.
- Overleaf: see `paper/README.md`.
