# Results

`summary.csv` / `summary.md` are written here by `src/summarize.py`.

`background_pilot_subject_hints.json` supplies the foreground words for the two
category 8 cases (`811000000000` and `811000000001`). The background command in the
[project README](../README.md#yangyuan-improvement) writes the generated images
to `outputs/background-reproduction/`.
The first two cases in category 8 are from `1_artificial/1_animal`: an illustrated
sheep and a watercolor deer. Their target prompts preserve the illustration style.

The pilot used PIE-Bench v1. Its `mapping_file.json` had SHA-256
`534c58a7d58d009b41aa865fb05ae96f79e573017b2d02ab15bb9658e8c749dc`.
The images came from `image_editing_piebench_inputs.zip` in
`yanlinli/image-editing-evaluation-handoff`, archive SHA-256
`7ad1c345c7abf0306251b7f3fcd16b57ac6d50548930c120a01937c1d82c1cd8`.

## Apple Silicon reproduction

Use Python 3.9 from the repository root to prepare the pinned environment and model.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pilot.txt
bash scripts/setup.sh --no-pip

export MODEL_ID="$PWD/data/models/sd-v1-4"
huggingface-cli download CompVis/stable-diffusion-v1-4 \
  --revision 133a221b8aa7292a167afc5127cb63fb5005638b \
  --include model_index.json 'feature_extractor/*' 'tokenizer/*' \
    '*/config.json' '*/model.safetensors' '*/diffusion_pytorch_model.safetensors' \
  --local-dir "$MODEL_ID"
export DEVICE=mps
```

Run the [Yangyuan commands](../README.md#yangyuan-improvement) in the same terminal.

## Score the background pilot

Use the existing PIE-Bench evaluator after generating the two edits per method.

```bash
python src/evaluate.py --data_path data/PIE-Bench \
  --output_path outputs/background-reproduction --methods ddim+masactrl Yangyuan \
  --categories 8 --device cpu \
  --result_csv outputs/background-reproduction/metrics_long_background.csv
```

The evaluator reports skipped outputs for the other category 8 cases and writes
28 scores for the four generated images.

## Original corgi comparison

The [report figure](../paper/figures/corgi-comparison.png) uses the saved panels
from the original local MasaCtrl prototype. To regenerate them, use the prototype
checkout with its Python environment and cached model, then run these commands
from that checkout's root.

```bash
source .venv/bin/activate
python -m reproduction --device mps --offline \
  --config reproduction/configs/real_edit_corgi_original.json \
  --output-root outputs/corgi-reference
python -m reproduction --device mps --offline \
  --config reproduction/configs/real_edit_corgi_attention.json \
  --output-root outputs/corgi-reference
```

The saved run used PyTorch 2.2.2, Diffusers 0.15.0, Transformers 4.26.1, and
Pillow 9.5.0. The project README's corgi command uses the same nearest-neighbor
resize as the prototype. PIE-Bench runs use the benchmark image loader.
