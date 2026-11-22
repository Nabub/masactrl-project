"""Build blinded A/B comparison images for the human evaluation (PIE-Bench cat 8)."""
import csv, json, random, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "user_study"
RES = ROOT / "results/baseline+improvement1"
METHODS = {"baseline": RES / "baseline", "improvement1": RES / "improvement1"}
SEED, PER_GROUP = 2026, 2          # 2 images from each of the 8 sub-groups -> 16 questions

def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()

mapping = json.loads((ROOT / "mapping_file.json").read_text())
items = {k: v for k, v in mapping.items() if str(v["editing_type_id"]) == "8"}
groups = {}
for k, v in sorted(items.items()):
    groups.setdefault(str(Path(v["image_path"]).parent), []).append(k)
rng = random.Random(SEED)
chosen = []
for g in sorted(groups):
    chosen += rng.sample(groups[g], PER_GROUP)
rng.shuffle(chosen)
flips = [True, False] * (len(chosen) // 2) + [True] * (len(chosen) % 2)
rng.shuffle(flips)   # balance: ours shown as A in exactly half of the questions

S, PAD, HEAD = 512, 16, 70
rows = []
for qi, key in enumerate(chosen, 1):
    v = items[key]
    rel = Path(v["image_path"])
    src = Image.open(ROOT / rel).convert("RGB").resize((S, S))
    outs = {m: Image.open(p / rel.with_suffix(".png")).convert("RGB") for m, p in METHODS.items()}
    order = ["improvement1", "baseline"] if flips[qi - 1] else ["baseline", "improvement1"]  # order[0]=A
    tgt = v["editing_prompt"].replace("[", "").replace("]", "")
    instr = v.get("editing_instruction", "")
    text = f"Edit instruction: {instr}\nTarget description: {tgt}"
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, 95)
    W = 3 * S + 4 * PAD
    H = HEAD + S + PAD + 34 * len(lines) + 2 * PAD
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 18), f"Question {qi}", fill="black", font=font(30, True))
    for i, (label, im) in enumerate([("Original", src), ("A", outs[order[0]]), ("B", outs[order[1]])]):
        x = PAD + i * (S + PAD)
        canvas.paste(im, (x, HEAD))
        d.rectangle([x, HEAD, x + 120, HEAD + 44], fill="black")
        d.text((x + 12, HEAD + 6), label if label != "Original" else "Orig.", fill="white", font=font(28, True))
    y = HEAD + S + PAD
    for ln in lines:
        d.text((PAD, y), ln, fill="black", font=font(26))
        y += 34
    name = f"Q{qi:02d}.jpg"
    canvas.save(OUT / "images" / name, quality=90)
    rows.append({"question": qi, "image_file": name, "pie_id": key, "image_path": str(rel),
                 "A": order[0], "B": order[1], "instruction": instr, "target_prompt": tgt})

with open(OUT / "answer_key_DO_NOT_SHARE.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "questions written;", sum(r["A"] == "improvement1" for r in rows), "have ours as A")
