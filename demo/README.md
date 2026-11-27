---
title: ROAM VLM Terrain Maps
emoji: 🏔️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Upload a photo, get VLM risk and steppable maps
models:
  - Mendel192/san
---

# ROAM vlm_nav demo

Upload a photo and see what the vision-language model in **ROAM** makes of it: it names the terrain
and gives each kind a risk, steppable and preferred-area value, and SAN open-vocabulary segmentation
turns those labels into per-pixel maps. Click on the result to plan a path with A*.

- Paper: *Bridging vision-language reasoning and learned control for legged navigation in the wild*,
  Science Robotics (2026), [doi:10.1126/scirobotics.aee1549](https://doi.org/10.1126/scirobotics.aee1549)
- arXiv: [arxiv.org/abs/2611.25215](https://arxiv.org/abs/2611.25215)
- Project page: [roam-hiking.github.io](https://roam-hiking.github.io/)
- Code: [github.com/leggedrobotics/roam](https://github.com/leggedrobotics/roam)

The examples use cached labels and need no key. Each visitor also gets a few photos of their own per
day on the demo's Gemini key. After that, or when the shared quota runs out, they can paste their own
[Gemini API key](https://aistudio.google.com/apikey) (or an OpenAI key with a `gpt-*` model).
A pasted key is used for that one request and not stored.

## Space settings

| Setting | What it does |
|---|---|
| Secret `SHARED_GEMINI_API_KEY` | The demo's own key. Use a free-tier AI Studio key with **no billing attached**: it cannot cost anything, and when its quota is used up the demo asks visitors for their own key for 15 minutes. Leave it unset to always ask for a key. |
| Variable `FREE_PHOTOS_PER_DAY` | Photos per visitor per day on the shared key (default 5). A visitor is their Hugging Face account if logged in, else their IP address. The counts live in memory, so a Space restart resets them. |

## Run it elsewhere

- **Colab:** open [`demo/vlm_nav_demo.ipynb`](https://colab.research.google.com/github/leggedrobotics/roam/blob/main/demo/vlm_nav_demo.ipynb)
  and run all cells (GPU runtime).
- **Locally:** from a clone of the roam repository, `python demo/app.py` (installs SAN and detectron2
  on first start).

This folder is the Space itself: the files here are pushed to the Hugging Face Space as they are.
