"""Figures for the GPU report. Uses only the numbers recorded in docs/dl-extension.md (no data needed)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = ROOT / "report/figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})

# ---- pipeline
fig, ax = plt.subplots(figsize=(7.1, 1.35))
ax.axis("off")
steps = ["Grayscale\n2048 × 2048", "Normalize by\ndisk mean / std", "Two U-Nets\nResNet34 + 50\n+ flip TTA",
         "Average\nprobability\nmaps", "Hysteresis,\nlimb erosion,\nfragment grouping", "Components\n→ COCO RLE\nCSV"]
for i, text in enumerate(steps):
    x = i / 6 + 0.005
    ax.add_patch(FancyBboxPatch((x, 0.12), 0.14, 0.76, boxstyle="round,pad=.009", facecolor="#e9f3fa", edgecolor="#2878a4"))
    ax.text(x + 0.07, 0.5, text, ha="center", va="center", fontsize=7)
    if i < 5:
        ax.annotate("", xy=(x + 0.16, 0.5), xytext=(x + 0.142, 0.5), arrowprops=dict(arrowstyle="->", color="#2878a4"))
fig.tight_layout()
fig.savefig(out / "gpu_pipeline.pdf")
plt.close(fig)

# ---- validation PQ with 95% date-cluster bootstrap intervals (docs/dl-extension.md, v2 table)
rows = [  # label, PQ, low, high
    ("Classical baseline", 0.1796, 0.1602, 0.1992),
    ("U-Net ResNet34 (v1)", 0.4111, 0.3905, 0.4308),
    ("U-Net ResNet34, 768 px, EMA", 0.4208, 0.4006, 0.4405),
    ("U-Net ResNet50, 768 px, EMA", 0.4295, 0.4088, 0.4489),
    ("Ensemble R34 + R50 (selected)", 0.4271, 0.4053, 0.4466),
]
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for i, (label, pq, lo, hi) in enumerate(rows[::-1]):
    color = "#2878a4" if "selected" in label else "#8aa1b1"
    ax.barh(i, pq, color=color, height=0.6)
    ax.plot([lo, hi], [i, i], color="black", lw=1)
    ax.text(0.01, i, label, va="center", fontsize=6.5, color="white" if pq > 0.25 else "black")
ax.set_yticks([])
ax.set_xlim(0, 0.5)
ax.set_xlabel("Validation micro PQ (95% interval)")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(out / "gpu_validation.pdf")
plt.close(fig)
print("saved", out / "gpu_pipeline.pdf", out / "gpu_validation.pdf")
