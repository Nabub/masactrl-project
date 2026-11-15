"""Baseline runner for the MasaCtrl project (AI6103).

Runs the *official* MasaCtrl code (vendored by the PIE-Bench / PnPInversion
repository, which adds the DirectInversion hook) on PIE-Bench, or on a single
image for a quick sanity check.

Methods
-------
fixed_seed               SD v1.4, DDIM-inverted noise + target prompt, no attention control
                         (the "Fixed Seed" column in the MasaCtrl paper; a lower bound).
ddim+masactrl            Official MasaCtrl, DDIM inversion, S=4, L=10 (paper default).
directinversion+masactrl MasaCtrl on top of PnP/Direct Inversion (stronger baseline).
Yangyuan                MasaCtrl with attention masks and hybrid initialization.

Every method writes one 512x512 PNG per image to
    <output_path>/<method>/<image_path without extension>.png
plus a runtime log <output_path>/<method>/runtime.jsonl.

New methods (e.g. our improvement) should be added to METHODS below so they are
run and evaluated through exactly the same pipeline.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PNP_ROOT = ROOT / "third_party" / "PnPInversion"
if not PNP_ROOT.exists():
    sys.exit(f"[error] {PNP_ROOT} not found - run `bash scripts/setup.sh` first.")
sys.path.insert(0, str(PNP_ROOT))

from diffusers import DDIMScheduler  # noqa: E402
from models.masactrl.diffuser_utils import MasaCtrlPipeline  # noqa: E402
from models.masactrl.masactrl import MutualSelfAttentionControl  # noqa: E402
from models.masactrl.masactrl_utils import (  # noqa: E402
    AttentionBase,
    regiter_attention_editor_diffusers,
)
from models.p2p.inversion import DirectInversion  # noqa: E402
from utils.utils import load_512  # noqa: E402


# --------------------------------------------------------------------------- utils
def setup_seed(seed: int = 1234):
    """Same seeding as the PIE-Bench scripts, applied before every edit."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_to_tensor(image_np: np.ndarray, device) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3x512x512 in [-1, 1] (identical to MasaCtrl's load_image)."""
    t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    t = F.interpolate(t, (512, 512))
    return t.to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """3xHxW in [0, 1] -> PIL."""
    arr = (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- editor
class BaselineEditor:
    def __init__(self, model_id: str, device, num_steps: int = 50, guidance: float = 7.5,
                 start_step: int = 4, start_layer: int = 10, *, seed: int = 1234,
                 offline: bool = False, query_chunk_size: int = 128):
        self.device = device
        self.num_steps = num_steps
        self.guidance = guidance
        self.start_step = start_step
        self.start_layer = start_layer
        self.seed = seed
        self.query_chunk_size = query_chunk_size
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  clip_sample=False, set_alpha_to_one=False)
        self.model = MasaCtrlPipeline.from_pretrained(
            model_id, scheduler=scheduler, local_files_only=offline).to(device)
        if device.type == "mps":
            self.model.enable_vae_slicing()
        self.model.scheduler.set_timesteps(num_steps)
        self._ddim_cache = None  # (key, start_code) so fixed_seed & ddim+masactrl share one inversion

    def register_attention(self, editor):
        if self.device.type in ("mps", "cpu"):
            from yangyuan import register_chunked_baseline

            register_chunked_baseline(self.model, editor, self.query_chunk_size)
        else:
            regiter_attention_editor_diffusers(self.model, editor)

    # ---- inversions
    def ddim_invert(self, key, image_np):
        if self._ddim_cache is not None and self._ddim_cache[0] == key:
            return self._ddim_cache[1]
        source = image_to_tensor(image_np, self.device)
        # plain attention during inversion (a MasaCtrl editor from the previous image may still be hooked)
        self.register_attention(AttentionBase())
        # Real-image setting of the paper: source prompt is empty for inversion.
        start_code, _ = self.model.invert(source, "", guidance_scale=self.guidance,
                                          num_inference_steps=self.num_steps, return_intermediates=False)
        self._ddim_cache = (key, start_code)
        return start_code

    # ---- methods (each returns a PIL image of the edited result)
    def fixed_seed(self, key, image_np, prompt_src, prompt_tar):
        start_code = self.ddim_invert(key, image_np)
        self.register_attention(AttentionBase())
        out = self.model([prompt_tar], latents=start_code[-1:], num_inference_steps=self.num_steps,
                         guidance_scale=self.guidance)
        return tensor_to_pil(out[-1])

    def ddim_masactrl(self, key, image_np, prompt_src, prompt_tar):
        start_code = self.ddim_invert(key, image_np)
        prompts = ["", prompt_tar]
        self.register_attention(MutualSelfAttentionControl(
            self.start_step, self.start_layer, total_steps=self.num_steps))
        out = self.model(prompts, latents=start_code.expand(len(prompts), -1, -1, -1),
                         num_inference_steps=self.num_steps, guidance_scale=self.guidance)
        return tensor_to_pil(out[-1])

    def directinversion_masactrl(self, key, image_np, prompt_src, prompt_tar):
        prompts = ["", prompt_tar]
        inverter = DirectInversion(model=self.model, num_ddim_steps=self.num_steps)
        _, _, x_stars, noise_loss_list = inverter.invert(image_gt=image_np, prompt=prompts,
                                                         guidance_scale=self.guidance)
        x_t = x_stars[-1]
        self.register_attention(MutualSelfAttentionControl(
            self.start_step, self.start_layer, total_steps=self.num_steps))
        out = self.model(prompts, latents=x_t.expand(len(prompts), -1, -1, -1),
                         num_inference_steps=self.num_steps, guidance_scale=self.guidance,
                         noise_loss_list=noise_loss_list)
        return tensor_to_pil(out[-1])

    def yangyuan(self, key, image_np, prompt_src, prompt_tar, *, subject_hint=""):
        from yangyuan import improved_edit

        edited = improved_edit(
            self.model, image_to_tensor(image_np, self.device), prompt_src, prompt_tar,
            subject_hint=subject_hint, steps=self.num_steps, guidance=self.guidance,
            start_step=self.start_step, start_layer=self.start_layer, seed=self.seed,
            query_chunk_size=self.query_chunk_size,
        )
        return tensor_to_pil(edited)


METHODS = {
    "fixed_seed": BaselineEditor.fixed_seed,
    "ddim+masactrl": BaselineEditor.ddim_masactrl,
    "directinversion+masactrl": BaselineEditor.directinversion_masactrl,
    "Yangyuan": BaselineEditor.yangyuan,
}


# --------------------------------------------------------------------------- main
def iter_pie_bench(data_path: Path, categories, limit):
    with open(data_path / "mapping_file.json") as f:
        mapping = json.load(f)
    n = 0
    for key, item in mapping.items():
        if str(item["editing_type_id"]) not in categories:
            continue
        yield key, item
        n += 1
        if limit and n >= limit:
            break


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", default="data/PIE-Bench", help="folder with mapping_file.json and annotation_images/")
    p.add_argument("--output_path", default="outputs")
    p.add_argument("--methods", nargs="+", default=[m for m in METHODS if m != "Yangyuan"], choices=list(METHODS))
    p.add_argument("--categories", nargs="+", default=[str(i) for i in range(10)],
                   help="PIE-Bench editing_type_id; 5 = change pose (MasaCtrl's target case)")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (debugging)")
    p.add_argument("--subject_hints", type=Path,
                   help="JSON mapping from image IDs to Yangyuan foreground subject hints")
    p.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--start_step", type=int, default=4, help="S in the paper")
    p.add_argument("--start_layer", type=int, default=10, help="L in the paper")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--offline", action="store_true", help="load cached Stable Diffusion weights only")
    p.add_argument("--query_chunk_size", type=int, default=128, help="maximum queries per attention chunk")
    p.add_argument("--rerun", action="store_true", help="overwrite existing outputs")
    # single-image sanity check (no PIE-Bench needed)
    p.add_argument("--image", help="run one image instead of PIE-Bench")
    p.add_argument("--image_resize", choices=("pie-bench", "nearest"), default="pie-bench",
                   help="resize a single image with the benchmark or original prototype preprocessing")
    p.add_argument("--src_prompt", default="")
    p.add_argument("--tgt_prompt", default="a photo of a running corgi")
    args = p.parse_args()
    if args.image_resize == "nearest" and not args.image:
        p.error("--image_resize nearest requires --image")
    subject_hints = {}
    if args.subject_hints:
        try:
            subject_hints = json.loads(args.subject_hints.read_text())
        except (OSError, ValueError) as exc:
            p.error(f"Could not read subject hints: {exc}")
        if not isinstance(subject_hints, dict) or any(
            not isinstance(value, str) or not value.strip() for value in subject_hints.values()
        ):
            p.error("Subject hints must map image IDs to nonempty strings")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if (device_name == "cuda" and not torch.cuda.is_available() or
            device_name == "mps" and not torch.backends.mps.is_available()):
        p.error(f"{device_name} is unavailable")
    if args.limit < 0 or args.query_chunk_size < 1:
        p.error("limit must be nonnegative and query_chunk_size must be positive")
    device = torch.device(device_name)
    print(f"Generation device: {device}", flush=True)
    out_root = Path(args.output_path)

    if args.image:
        items = [("single", {"image_path": args.image, "original_prompt": args.src_prompt,
                             "editing_prompt": args.tgt_prompt, "editing_type_id": "-"})]
        img_root = Path(".")
    else:
        data_path = Path(args.data_path)
        items = list(iter_pie_bench(data_path, set(args.categories), args.limit))
        img_root = data_path / "annotation_images"
    if not items:
        p.error("no matching images")
    for _, item in items:
        if not (img_root / item["image_path"]).is_file():
            p.error(f"missing input image {img_root / item['image_path']}")
    editor = BaselineEditor(args.model_id, device, args.steps, args.guidance, args.start_step, args.start_layer,
                            seed=args.seed, offline=args.offline, query_chunk_size=args.query_chunk_size)
    print(f"{len(items)} images x {len(args.methods)} methods")

    for idx, (key, item) in enumerate(items):
        prompt_src = item["original_prompt"].replace("[", "").replace("]", "")
        prompt_tar = item["editing_prompt"].replace("[", "").replace("]", "")
        img_path = img_root / item["image_path"]
        if args.image_resize == "nearest":
            from torchvision.io import ImageReadMode, read_image

            pixels = read_image(str(img_path), mode=ImageReadMode.RGB).unsqueeze(0)
            resized = F.interpolate(pixels, (512, 512), mode="nearest")
            image_np = resized[0].permute(1, 2, 0).numpy()
        else:
            image_np = load_512(str(img_path))
        for method in args.methods:
            rel = Path(item["image_path"]).with_suffix(".png") if not args.image else Path(Path(args.image).stem + ".png")
            save_path = out_root / method / rel
            if save_path.exists() and not args.rerun:
                continue
            save_path.parent.mkdir(parents=True, exist_ok=True)
            setup_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            if method == "Yangyuan":
                edited = METHODS[method](editor, key, image_np, item["original_prompt"], item["editing_prompt"],
                                         subject_hint=subject_hints.get(key, item.get("blended_word", "")))
            else:
                edited = METHODS[method](editor, key, image_np, prompt_src, prompt_tar)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0
            edited.save(save_path)
            with open(out_root / method / "runtime.jsonl", "a") as f:
                f.write(json.dumps({"id": key, "category": item["editing_type_id"], "seconds": round(dt, 2)}) + "\n")
            print(f"[{idx + 1}/{len(items)}] {method:<26} {key}  {dt:5.1f}s  -> {save_path}", flush=True)
        if args.image:  # also save a side-by-side grid for quick viewing
            tiles = [Image.fromarray(image_np)] + [Image.open(out_root / m / f"{Path(args.image).stem}.png") for m in args.methods]
            grid = Image.new("RGB", (512 * len(tiles), 512))
            for i, t in enumerate(tiles):
                grid.paste(t, (512 * i, 0))
            grid.save(out_root / f"{Path(args.image).stem}_grid.png")
            print("grid:", out_root / f"{Path(args.image).stem}_grid.png", "(source | " + " | ".join(args.methods) + ")")


if __name__ == "__main__":
    main()
