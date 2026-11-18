"""Put an input image and its value maps side by side, one panel per field.

Each map is drawn over a faded copy of the input image, and one colour bar on the right gives the scale.

Usage: python3 compose_figure.py RESULT_DIR FIELD [FIELD ...] --out figure.png
RESULT_DIR is a folder written by `example.py --save_images` (input.png and <field>_gray.png).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("result_dir")
parser.add_argument("fields", nargs="+")
parser.add_argument("--out", required=True)
parser.add_argument("--title", default="")
parser.add_argument("--alpha", type=float, default=0.6, help="Opacity of the value map over the image")
args = parser.parse_args()

# Fields whose 0-1 value stands for a physical scale
SCALES = {"mass": "mass (0 = 10 g, 1 = 100 kg, log)"}

image = Image.open(os.path.join(args.result_dir, "input.png")).convert("RGB")
# Faded grey background, so the colours of the map are not mixed with the colours of the photo
background = np.asarray(image.convert("L"), dtype=np.float32) / 255 * 0.6 + 0.4
n = len(args.fields) + 1
fig, axs = plt.subplots(1, n, figsize=(3.2 * n + 0.6, 3.2 * image.height / image.width + 0.6))
axs[0].imshow(image)
axs[0].set_title("input")
for ax, field in zip(axs[1:], args.fields):
    value = np.asarray(Image.open(os.path.join(args.result_dir, f"{field}_gray.png")), dtype=np.float32) / 255
    ax.imshow(background, cmap="gray", vmin=0, vmax=1)
    mappable = ax.imshow(value, cmap="turbo", vmin=0, vmax=1, alpha=args.alpha, interpolation="bilinear")
    ax.set_title(SCALES.get(field, field))
for ax in axs:
    ax.axis("off")
if args.title:
    fig.suptitle(args.title)
fig.tight_layout(rect=(0, 0, 1 - 0.5 / fig.get_figwidth(), 1))
cax = fig.add_axes((1 - 0.42 / fig.get_figwidth(), 0.12, 0.12 / fig.get_figwidth(), 0.72))
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, 1)), cax=cax)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
fig.savefig(args.out, dpi=130, bbox_inches="tight", pad_inches=0.1)
