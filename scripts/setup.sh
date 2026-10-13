#!/usr/bin/env bash
# Fetch the official code bases at pinned commits and install dependencies.
#   bash scripts/setup.sh            # clone + pip install
#   bash scripts/setup.sh --no-pip   # clone only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party"

clone_pinned () {  # url dir commit
  local url=$1 dir=$ROOT/third_party/$2 commit=$3
  if [ ! -d "$dir/.git" ]; then git clone -q "$url" "$dir"; fi
  git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  echo "[ok] $2 @ ${commit:0:7}"
}

# Official MasaCtrl (Cao et al., ICCV 2023) - reference implementation, kept for diffing/playground notebooks
clone_pinned https://github.com/TencentARC/MasaCtrl.git     MasaCtrl     d0ba16bb7c12125b509e374b000ee61065bb8149
# PIE-Bench / PnP Inversion (Ju et al., ICLR 2024) - contains MasaCtrl + DirectInversion hook + evaluation metrics
clone_pinned https://github.com/cure-lab/PnPInversion.git   PnPInversion 07f97f448150e2ca220bebd54c8f687c5c50c67a

if [ "${1:-}" != "--no-pip" ]; then
  pip install -q -r "$ROOT/requirements.txt"
fi
echo "[done] next: put PIE-Bench in data/PIE-Bench and run scripts/run_baseline.sh"
