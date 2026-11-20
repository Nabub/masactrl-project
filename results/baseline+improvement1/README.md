# PIE-Bench category 8 (change background), 80 images

Code: github.com/Nabub/masactrl-project, commit 24c205c, unmodified.
`baseline/` = method `ddim+masactrl`, `improvement1/` = method `Yangyuan` (folder renamed only).
Settings: SD v1.4 fp32, 50 DDIM steps, guidance 7.5, start_step 15, start_layer 10, seed 1234, run on NTU TC2.
Each PNG is 512x512, same relative path as the PIE-Bench image.

Subject hints for improvement1 (`subject_hints_category8.json`) were made automatically from the
source prompt (`src/make_subject_hints.py`: last noun before the first preposition/verb), not by hand.
Known misses: 823000000006 -> "wooden", 814000000007 -> "level". Some scene images have no clear subject.

| Method | Structure x1e3 ↓ | PSNR ↑ | LPIPS x1e3 ↓ | MSE x1e4 ↓ | SSIM x1e2 ↑ | CLIP whole ↑ | CLIP edited ↑ |
|---|---|---|---|---|---|---|---|
| baseline | 33.88 | 21.68 | 87.70 | 94.30 | 82.61 | 24.33 | 22.09 |
| improvement1 | 135.95 | 16.78 | 198.53 | 324.99 | 73.20 | 28.18 | 25.97 |

PSNR/LPIPS/MSE/SSIM are on the unedited part (the subject) and cover 70 images; the other 10 have no
unedited part in the PIE-Bench mask. Per-image scores: `metrics_long_category8.csv`.
