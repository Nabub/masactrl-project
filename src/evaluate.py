"""Evaluate edited images with the PIE-Bench metrics (Ju et al., ICLR 2024).

Uses PIE-Bench's own MetricsCalculator so that numbers are directly comparable
with Table 1 of the PIE-Bench paper (which includes DDIM+MasaCtrl).

Output: a long-format CSV  (id, category, method, metric, value)
        -> feed it to src/summarize.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PNP_ROOT = ROOT / "third_party" / "PnPInversion"
if not PNP_ROOT.exists():
    sys.exit(f"[error] {PNP_ROOT} not found - run `bash scripts/setup.sh` first.")
sys.path.insert(0, str(PNP_ROOT))

from evaluation.evaluate import calculate_metric, mask_decode  # noqa: E402
from evaluation.matrics_calculator import MetricsCalculator  # noqa: E402

DEFAULT_METRICS = [
    "structure_distance",          # DINO self-similarity distance, whole image   (lower better)
    "psnr_unedit_part",            # background preservation                       (higher better)
    "lpips_unedit_part",           #                                               (lower better)
    "mse_unedit_part",             #                                               (lower better)
    "ssim_unedit_part",            #                                               (higher better)
    "clip_similarity_target_image",            # CLIP(edited image, target prompt)   (higher better)
    "clip_similarity_target_image_edit_part",  # same, edited region only            (higher better)
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/PIE-Bench")
    p.add_argument("--output_path", default="outputs", help="root that contains one folder per method")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--categories", nargs="+", default=[str(i) for i in range(10)])
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--result_csv", default="results/metrics_long.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    data_path = Path(args.data_path)
    with open(data_path / "mapping_file.json") as f:
        mapping = json.load(f)

    calc = MetricsCalculator(args.device)
    Path(args.result_csv).parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with open(args.result_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "method", "metric", "value"])
        for key, item in mapping.items():
            cat = item["editing_type_id"]
            if cat not in args.categories:
                continue
            mask = mask_decode(item["mask"])[:, :, None].repeat(3, axis=2)
            src_prompt = item["original_prompt"].replace("[", "").replace("]", "")
            tgt_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
            src = Image.open(data_path / "annotation_images" / item["image_path"]).convert("RGB")
            for method in args.methods:
                tgt_path = Path(args.output_path) / method / Path(item["image_path"]).with_suffix(".png")
                if not tgt_path.exists():
                    missing += 1
                    continue
                tgt = Image.open(tgt_path).convert("RGB")
                for metric in args.metrics:
                    v = calculate_metric(calc, metric, src, tgt, mask, mask, src_prompt, tgt_prompt)
                    w.writerow([key, cat, method, metric, "nan" if isinstance(v, str) else float(v)])
            print(f"evaluated {key} (cat {cat})")
    if missing:
        print(f"[warn] {missing} (image, method) pairs had no output and were skipped")
    print("wrote", args.result_csv)


if __name__ == "__main__":
    main()
