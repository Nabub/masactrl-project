#!/usr/bin/env bash
# Full baseline pipeline: generate -> evaluate -> summarise (+ LaTeX table).
#
#   bash scripts/run_baseline.sh pose      # PIE-Bench cat 5 (change pose, 40 imgs)
#   bash scripts/run_baseline.sh all       # all 700 images (A100 / cluster GPU recommended; resumable)
# Rough cost (fp32, 3 methods): a few minutes per image on a T4, well under a minute on an A100.
# Re-running skips images that already exist, so a killed Colab session can simply be restarted.
#   bash scripts/run_baseline.sh smoke     # 1 image, no PIE-Bench needed (corgi example from the MasaCtrl repo)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-pose}
DATA=${DATA:-data/PIE-Bench}
OUT=${OUT:-outputs}
METHODS=${METHODS:-"fixed_seed ddim+masactrl directinversion+masactrl"}

case $MODE in
  smoke)
    python src/run_baseline.py --image third_party/MasaCtrl/gradio_app/images/corgi.jpg \
      --src_prompt "" --tgt_prompt "a photo of a running corgi" \
      --methods $METHODS --output_path "$OUT/smoke"
    exit 0 ;;
  pose) CATS="5" ;;
  all)  CATS="0 1 2 3 4 5 6 7 8 9" ;;
  *) echo "usage: $0 {smoke|pose|all}"; exit 1 ;;
esac

python src/run_baseline.py --data_path "$DATA" --output_path "$OUT" --methods $METHODS --categories $CATS
python src/evaluate.py     --data_path "$DATA" --output_path "$OUT" --methods $METHODS --categories $CATS \
                           --result_csv "results/metrics_long_${MODE}.csv"
python src/summarize.py    --result_csv "results/metrics_long_${MODE}.csv" --output_path "$OUT"
