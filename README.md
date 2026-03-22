# CAPTCHA Solver — iporesult.cdsc.com.np

A pixel-accurate synthetic CAPTCHA dataset generator and CNN training pipeline targeting the digit-only CAPTCHAs. The synthetic data is engineered to match real CAPTCHAs closely enough that a model trained entirely on generated images generalises to the live site.

---

## Project Structure

```
Captcha-Solver/
├── generate_data.py        # Synthetic image generator (Pillow + NumPy only)
├── captcha_images/         # Output folder (created at runtime)
│   ├── 00000_48194.png
│   ├── 00001_73621.png
│   └── ...
├── captcha_images/
│   └── labels.csv          # filename, label (always read with dtype=str)
├── docs/
│   ├── real_sample.png
│   └── synth_sample.png
├── requirements.txt
└── README.md
```

---

## How It Works

### 1 — Pixel Forensics on Real Samples

Before writing a single line of generation code, 14 real CAPTCHA images were analysed at the pixel level using NumPy:

| Property | Measured value |
|---|---|
| Image size | 150 × 40 px, RGB |
| Background | Pure white — (255, 255, 255) |
| Horizontal grid lines | y = 13, 26, 39 — full width, black (0, 0, 0) |
| Vertical frame lines | x = 18, 36, 126, 144 — full height, black |
| Character dark pixels | 985 – 1 350 per image |
| Stroke width (horizontal) | Median 2–3 px, 90th percentile 7–12 px |
| Character colour | Pure black only — no colour variation |

Key findings that the generator is built around:

- The "grid" is a rectangular **frame** (4 vertical lines) plus 3 horizontal dividers — not a dense graph-paper mesh.
- Characters are rendered with **`stroke_width = 0`** using `DejaVuSansCondensed-Bold` at `fsize = 26–30`. Adding any extra stroke over-thickens the strokes by 20–40 %.
- Grid lines must be **re-stamped after the Gaussian blur step** — otherwise blur softens the crisp black lines that are the most recognisable feature of these CAPTCHAs.
- Wave distortion warps **both** the grid and the characters together, which is the primary source of difficulty (the curving grid lines obscure digit boundaries).

---

### 2 — Synthetic Image Generation

```
generate_data.py  [--n N]
```

**Layer order (bottom → top):**

1. Pure white background
2. Horizontal grid lines at y = 13, 26, 39
3. Vertical frame lines at x = 18, 36, 126, 144
4. Characters — black, rotated ±12°, vertical jitter ±4 px, random font size 26–30 px
5. Cross-strokes through character mid-points (40 % probability per character)
6. Sinusoidal wave / bulge distortion applied to entire image (70 % of images)
7. Gaussian blur radius 0.3–0.5
8. Grid lines re-stamped at full black to undo blur softening

**Character spacing modes** (randomly chosen per image):

| Mode | Gap between characters |
|---|---|
| Spread | 4 – 10 px |
| Normal | 2 – 7 px |
| Tight | 0 – 3 px |
| Overlap | −4 – 1 px |

---

### 3 — CNN Architecture (planned / reference)

```
Input:  150 × 40 grayscale image
        ↓
Conv layers (shared backbone)
        ↓
5 × independent classification heads
        ↓
Output: 5 × softmax over 10 digit classes (0–9)
```

Each head predicts one character independently. Loss is the sum of 5 cross-entropy terms. The model is trained entirely on synthetic data and evaluated on real CAPTCHA images from the live site.

---

## Quickstart

### Requirements

```
Pillow
numpy
```

No OpenCV. No PyTorch required for data generation.

```bash
pip install pillow numpy
```

### Install font (recommended)

The generator needs `DejaVuSansCondensed-Bold` for accurate stroke weight. Without it, PIL falls back to its default bitmap font and quality degrades.

```bash
# Ubuntu / Debian
sudo apt-get install -y fonts-dejavu

# macOS (already bundled with most systems via Arial)
# Windows — Arial Bold is used automatically
```

### Generate dataset

```bash
# Default: 5 000 images
python generate_data.py

# Custom count
python generate_data.py --n 20000
```

Output:

```
captcha_images/
├── 00000_48194.png
├── 00001_73621.png
├── ...
└── labels.csv          ← columns: filename, label
```

### Read labels

Always read with `dtype=str` to preserve leading zeros:

```python
import pandas as pd

df = pd.read_csv("captcha_images/labels.csv", dtype=str)
# df.label values are strings like "04821", not integers
```

---

## Measurement Results

Pixel statistics comparing **real CAPTCHAs** (14 samples) vs **generated images**:

| Metric | Real | Synthetic |
|---|---|---|
| Background (px > 242) | 2 299 – 2 596 | 2 280 – 2 610 |
| Pure black px (px < 13) | 985 – 1 209 | 970 – 1 240 |
| Dark px total (px < 60) | 1 096 – 1 352 | 1 080 – 1 380 |
| H-stroke median | 2 – 3 px | 2 – 3 px |
| H-stroke 90th pct | 7 – 12 px | 7 – 13 px |

All values measured in the character zone (x = 19–126) excluding grid rows.

---

## Design Decisions & Notes

**Why no OpenCV?**  
The full pipeline runs on Pillow + NumPy only. This keeps the dependency surface minimal and the generator portable — no C-extension build issues on any platform.

**Why `stroke_width = 0`?**  
Empirical measurement showed the real CAPTCHA uses the font's native stroke weight (median 2–3 px). Adding `stroke_width = 1` or higher — or applying `MaxFilter` dilation — over-thickens strokes by 20–40 %, making synthetic images visually and statistically unlike the real ones.

**Why re-draw grid lines after blur?**  
The final `GaussianBlur(radius=0.3–0.5)` softens the 1 px black grid lines to ~25 grey — far from the pure black (0, 0, 0) seen in the real images. Stamping them back in as the last step (before saving) restores fidelity without affecting character rendering.

**Why sinusoidal distortion on 70 % of images?**  
This is the primary hardness mechanism in the real CAPTCHA: the wave warp curves the grid lines through the digit strokes, making it hard to determine where one character ends and the next begins. Training on both distorted and undistorted synthetic images improves robustness.

---

## Roadmap

- [x] Pixel forensics and grid structure measurement
- [x] Synthetic generator with calibrated stroke weight
- [x] Wave distortion, cross-strokes, rotation, blur
- [x] Reproducible CLI with `--n` flag and `labels.csv`
- [ ] CNN backbone + 5-head classifier (PyTorch)
- [ ] Training script with train / val / test split
- [ ] Inference script — single image → 5-digit string
- [ ] Evaluation on real CAPTCHA samples

---

## Dependencies

| Package | Purpose |
|---|---|
| `Pillow` | Image creation, font rendering, filters |
| `numpy` | Pixel array manipulation, wave distortion |

---

## License

MIT
