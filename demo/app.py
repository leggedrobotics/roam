"""ROAM vlm_nav demo: upload an image, get the VLM's labels and the per-pixel risk / steppable / preferred maps.

Runs as a Hugging Face Space (ZeroGPU), in Colab (demo/vlm_nav_demo.ipynb) or locally:

    python demo/app.py            # from the repository root

The VLM call (Gemini or OpenAI) runs on CPU; only the SAN segmentation uses the GPU.
"""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_repo_root():
    """The roam checkout: this file's parent when run from the repo, else a fresh clone (Space)."""
    parent = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(parent, "vlm_nav", "config")):
        return parent
    root = os.path.join(HERE, "roam")
    if not os.path.isdir(root):
        run(["git", "clone", "--depth", "1", "https://github.com/leggedrobotics/roam.git", root])
    return root


def ensure_san():
    """detectron2 must build against the installed torch, so it cannot sit in requirements.txt.

    It is built without its CUDA ops (SAN does not use them), which takes a few minutes instead of ten.
    SAN goes in without its old pins (numpy 1.22, Pillow 9); its pip package also leaves out configs/,
    so those come from a clone of the SAN repo.
    """
    if importlib.util.find_spec("detectron2") is None:
        print("Building detectron2 (first start only, a few minutes)...", flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", FORCE_CUDA="0")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                        "git+https://github.com/facebookresearch/detectron2.git"], check=True, env=env)
    if importlib.util.find_spec("san") is None:
        run([sys.executable, "-m", "pip", "install", "--no-deps", "git+https://github.com/mktk1117/SAN.git"])
    san_src = os.environ.get("SAN_SRC", os.path.join(HERE, "SAN"))
    if not os.path.isdir(os.path.join(san_src, "configs")):
        run(["git", "clone", "--depth", "1", "https://github.com/mktk1117/SAN.git", san_src])
    return os.path.join(san_src, "configs", "san_clip_vit_res4_coco.yaml")


ROOT = find_repo_root()
sys.path.insert(0, os.path.join(ROOT, "vlm_nav"))
SAN_CONFIG = ensure_san()

try:
    import spaces  # Hugging Face ZeroGPU

    gpu = spaces.GPU(duration=30)
except ImportError:  # Colab or local: a plain GPU (or CPU)

    def gpu(fn):
        return fn


import gradio as gr
import matplotlib
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from vlm_nav.planning.image_planner import build_cost_map, path_cost, plan_path, smooth_path
from vlm_nav.vlm_clients import create_client
from vlm_nav.vlm_seg.prompt_generator import create_prompt_generator_from_yaml
from vlm_nav.vlm_seg.semantic_label_processor import SegmentationParameters, SemanticLabelProcessor

PROMPTS = os.path.join(ROOT, "vlm_nav", "config", "prompts.yaml")
EXAMPLES = os.path.join(ROOT, "vlm_nav", "example_data")
TASKS = {
    "nature_hiking": "Navigation: risk, steppable, preferred area (the paper's setup)",
    "terrain_properties": "Navigation: friction, softness, steppable",
    "tabletop_manipulation": "Manipulation: graspable, grasp preference, avoid, friction, fragility",
    "object_interaction": "Manipulation: movable, pushable, pullable, mass",
}
DEFAULT_MODEL = "gemini-4-hydrogen"
MAX_SIZE = (640, 480)

# Memory is off: every upload is a new scene, unlike the robot's image stream.
generators = {task: create_prompt_generator_from_yaml(PROMPTS, task, False) for task in TASKS}

seg_cfg = SegmentationParameters(
    segmentation_model="SAN",
    model_config_file=SAN_CONFIG,
    model_weight_path=hf_hub_download("Mendel192/san", "san_vit_b_16.pth"),
)
processor = SemanticLabelProcessor(seg_cfg)


def load_image(image):
    image = image.convert("RGB")
    image.thumbnail(MAX_SIZE)
    return image


class QuotaError(Exception):
    """The API key hit its rate limit or daily quota (HTTP 429)."""


class NeedKey(Exception):
    """No VLM call possible without the visitor's own key; shown next to the Analyze button."""


GET_KEY = "Paste your own [Gemini API key](https://aistudio.google.com/apikey) (free) to keep going, or try one of the examples."


def is_quota_error(e):
    text = str(e)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower()


def ask_vlm(image, task, model, api_key, attempts=3):
    """Ask the VLM for labels with the given key (None: the provider's environment variable)."""
    generator = generators[task]
    try:
        client = create_client(model, api_key=api_key)
    except Exception as e:
        raise gr.Error(f"Could not create the VLM client: {e}")
    prompt, schema = generator.get_prompt(), generator.response_schema()
    for attempt in range(attempts):
        try:
            report = client._ask([image], prompt, schema)
            break
        except Exception as e:
            print(f"[{model}] request failed (attempt {attempt + 1}/{attempts}): {e}", flush=True)
            if is_quota_error(e):
                raise QuotaError(str(e))
            if attempt + 1 == attempts:
                raise gr.Error("The VLM call failed. Check the API key and the model name.")
            time.sleep(2**attempt)
    response = generator.report_to_dict(report)
    return generator.parse_json_response(response), response


class SharedKey:
    """The demo's own Gemini key (Space secret SHARED_GEMINI_API_KEY), so visitors can try a photo without one.

    Use a free-tier AI Studio key with no billing attached: it cannot cost anything, it just returns
    429 when its quota is used up, and the demo then asks visitors for their own key for a while.
    Each visitor (Hugging Face user, else IP address) gets `per_visitor` photos per day.
    """

    def __init__(self, key, per_visitor, pause_s=15 * 60):
        self.key, self.per_visitor, self.pause_s = key, per_visitor, pause_s
        self.uses = defaultdict(list)  # visitor -> timestamps of successful calls
        self.paused_until = 0.0
        self.lock = threading.Lock()

    def left(self, visitor):
        now = time.time()
        with self.lock:
            self.uses[visitor] = [t for t in self.uses[visitor] if now - t < 24 * 3600]
            return self.per_visitor - len(self.uses[visitor])

    def ask(self, image, task, visitor):
        if time.time() < self.paused_until:
            raise NeedKey("The demo's shared quota is used up for now. " + GET_KEY)
        if self.left(visitor) <= 0:
            raise NeedKey(f"You have used today's {photos(self.per_visitor)} on the demo's key. " + GET_KEY)
        try:
            result = ask_vlm(image, task, DEFAULT_MODEL, self.key)
        except QuotaError:
            self.paused_until = time.time() + self.pause_s
            raise NeedKey("The demo's shared quota is used up for now. " + GET_KEY)
        with self.lock:
            self.uses[visitor].append(time.time())
        return result


# A separate name from GEMINI_API_KEY, which is your own key when you run this locally or in Colab (no limit).
shared = SharedKey(os.environ["SHARED_GEMINI_API_KEY"], int(os.environ.get("FREE_PHOTOS_PER_DAY", "5"))) \
    if os.environ.get("SHARED_GEMINI_API_KEY") else None


def visitor_id(request):
    """Logged-in Hugging Face user, else the client IP (Spaces sit behind a proxy)."""
    if request is None:
        return "local"
    if getattr(request, "username", None):
        return "user:" + request.username
    forwarded = request.headers.get("x-forwarded-for", "")
    return "ip:" + (forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown"))


def photos(n):
    return f"{n} free photo" + ("" if n == 1 else "s")


def show_quota(request: gr.Request):
    return quota_note(visitor_id(request))


def quota_note(visitor):
    if shared is None:
        return ""
    left = max(shared.left(visitor), 0)
    return f"Free photos left today on the demo's key: **{left} of {shared.per_visitor}**."


@gpu
def segment(image, labels, task):
    layers, names = processor.process(image, labels, generators[task].get_default_values())
    return {name: layers[i].float().cpu().numpy() for i, name in enumerate(names)}


def overlay(image, layer, alpha=0.55):
    rgb = np.asarray(image, np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](np.clip(layer, 0, 1))[..., :3]
    return Image.fromarray(np.uint8(255 * ((1 - alpha) * rgb + alpha * heat)))


def label_table(labels):
    fields = sorted({k for v in labels.values() for k in v})
    return [[name] + [values.get(f, "") for f in fields] for name, values in labels.items()], ["label"] + fields


def request_labels(image, task, model, api_key, visitor):
    if api_key:
        try:
            return ask_vlm(image, task, model, api_key)
        except QuotaError:
            raise NeedKey("Your API key hit its rate limit or quota. Wait a minute and try again.")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return ask_vlm(image, task, model, None)  # your own key (local, Colab): no limit
    if shared is not None:
        return shared.ask(image, task, visitor)
    raise NeedKey("This needs a VLM. " + GET_KEY)


def analyze(image, task, model, api_key, labels_text, request: gr.Request = None):
    visitor = visitor_id(request)
    api_key = api_key.strip()
    if image is None:
        raise gr.Error("Upload an image first.")
    image = load_image(image)
    if labels_text.strip():
        try:
            labels = json.loads(labels_text)
        except json.JSONDecodeError as e:
            raise gr.Error(f"The labels are not valid JSON: {e}")
        response = None
    else:
        try:
            labels, response = request_labels(image, task, model or DEFAULT_MODEL, api_key, visitor)
        except NeedKey as e:
            # Keep the previous results; say why next to the button.
            return (gr.update(),) * 6 + (f"⚠️ {e}",)
    rows, headers = label_table(labels)
    maps = segment(image, dict(labels), task)
    gallery = [(overlay(image, layer), name) for name, layer in maps.items()]
    state = {"image": image, "maps": maps}
    return (
        gallery,
        gr.update(value=rows, headers=headers),
        json.dumps(labels, indent=2),
        json.dumps(response, indent=2) if response else "(labels given, no VLM call)",
        state,
        draw_plan(state, None),
        quota_note(visitor),
    )


def draw_plan(state, goal):
    """Risk overlay plus the A* path from the bottom centre (the robot's feet) to the clicked goal."""
    if not state or "risk" not in state["maps"]:
        return None
    import matplotlib.pyplot as plt

    image, maps = state["image"], state["maps"]
    risk, preferred, steppable = maps["risk"], maps.get("preferred_area"), maps.get("steppable")
    cost = build_cost_map(risk, preferred, preferred_weight=1000.0, steppable=steppable, inflate_radius=10)
    h, w = risk.shape
    start = (h - 1, w // 2)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(image)
    blocked = np.isinf(cost)
    ax.imshow(np.ma.masked_where(blocked, risk), cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.35)
    ax.imshow(np.ma.masked_where(~blocked, blocked), cmap="gray", vmin=0, vmax=1, alpha=0.45)
    if preferred is not None:
        ax.contour(preferred, levels=[0.5], colors="royalblue", linewidths=1.5)
    ax.plot(start[1], start[0], "o", color="cyan", markersize=10)
    text = "Click a goal on this image"
    if goal is not None:
        ax.plot(goal[1], goal[0], "*", color="magenta", markersize=16)
        path = plan_path(cost, start, goal)
        if path is None:
            text = "No path: the goal is not reachable on steppable ground"
        else:
            smooth = smooth_path(path)
            ax.plot(smooth[:, 1], smooth[:, 0], "-", color="magenta", linewidth=3)
            text = f"A* path: {len(path)} px, cost {path_cost(cost, path):.0f}"
    ax.text(8, 20, text, color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.6, lw=0))
    ax.axis("off")
    fig.canvas.draw()
    out = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


def on_click(state, evt: gr.SelectData):
    col, row = evt.index
    return draw_plan(state, (int(row), int(col)))


def example_labels(name, task):
    path = os.path.join(EXAMPLES, "labels", f"{name}_{task}.json")
    with open(path) as f:
        return json.dumps(json.load(f), indent=2)


EXAMPLE_ROWS = []
for name, task in [
    ("hiking_lake", "nature_hiking"),
    ("snow_stairs_zurich", "nature_hiking"),
    ("stepping_stones_stream", "terrain_properties"),
    ("messy_desk", "tabletop_manipulation"),
]:
    if os.path.exists(os.path.join(EXAMPLES, "labels", f"{name}_{task}.json")):
        EXAMPLE_ROWS.append([os.path.join(EXAMPLES, f"{name}.jpg"), task, example_labels(name, task)])

INTRO = """
# ROAM: what a vision-language model sees on the trail

Upload a photo. A vision-language model (VLM) names the terrain in it and gives each kind a value
(for hiking: **risk**, **steppable**, **preferred area**). Open-vocabulary segmentation (SAN) then turns
those labels into per-pixel maps, like the ones our legged robot used to hike in the Alps.

[Paper (Science Robotics)](https://doi.org/10.1126/scirobotics.aee1549) ·
[arXiv](https://arxiv.org/abs/2611.25215) · [Project page](https://roam-hiking.github.io/) ·
[Code](https://github.com/leggedrobotics/roam)

{KEY_NOTE}
"""

if shared is not None:
    KEY_NOTE = (f"The examples always work (their labels are cached). You get {photos(shared.per_visitor)} of your own "
                "per day on the demo's Gemini key. After that, or when the shared quota runs out, paste your own "
                "[Gemini API key](https://aistudio.google.com/apikey) (free tier works).")
else:
    KEY_NOTE = ("The examples run without a key (their labels are cached). For your own photo, paste a "
                "[Gemini API key](https://aistudio.google.com/apikey) (free tier works).")
KEY_NOTE += " A key you paste is used for that request only and never stored."

with gr.Blocks(title="ROAM vlm_nav demo") as demo:
    gr.Markdown(INTRO.format(KEY_NOTE=KEY_NOTE))
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Image", height=360)
            task_in = gr.Dropdown(list(TASKS), value="nature_hiking", label="Task")
            task_help = gr.Markdown(TASKS["nature_hiking"])
            key_in = gr.Textbox(label="Your Gemini or OpenAI API key (optional)", type="password", placeholder="Not needed for the examples or your free photos")
            with gr.Accordion("Advanced", open=False):
                model_in = gr.Textbox(DEFAULT_MODEL, label="VLM model (gemini-* or gpt-*)")
                labels_in = gr.Code(
                    language="json",
                    label="Labels (optional). Leave empty to ask the VLM, or edit them and run again to re-segment",
                )
            run_btn = gr.Button("Analyze", variant="primary")
            quota_out = gr.Markdown()
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Per-pixel maps (blue = 0, red = 1)", columns=3, height=300)
            table = gr.Dataframe(headers=["label"], label="VLM labels", interactive=False, wrap=True)
            with gr.Tab("Plan a path"):
                gr.Markdown("Navigation tasks only. Click a goal: A* plans from the bottom centre, avoiding risk "
                            "and blocked (grey) ground and preferring the blue-outlined area.")
                plan_img = gr.Image(label="Click a goal", interactive=False, height=360)
            with gr.Tab("Raw VLM response"):
                response_out = gr.Code(language="json", label="Response")

    gr.Examples(EXAMPLE_ROWS, inputs=[image_in, task_in, labels_in], label="Examples (no key needed)")

    task_in.change(lambda t: TASKS[t], task_in, task_help)
    # A new image or task needs new labels. (.upload/.input fire on user actions only, not on examples.)
    image_in.upload(lambda: "", None, labels_in)
    task_in.input(lambda: "", None, labels_in)
    run_btn.click(
        analyze,
        [image_in, task_in, model_in, key_in, labels_in],
        [gallery, table, labels_in, response_out, state, plan_img, quota_out],
    )
    demo.load(show_quota, None, quota_out)
    plan_img.select(on_click, state, plan_img)

if __name__ == "__main__":
    # Colab needs a public link to show the UI.
    demo.queue(max_size=20).launch(share="COLAB_RELEASE_TAG" in os.environ)
