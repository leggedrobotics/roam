"""Click a goal on the image and plan a path to it with A* over ROAM's risk map.

The start is the bottom centre of the image (the robot's feet). Left click sets the goal, right click moves
the start. The cost is 1 + risk_weight * risk, blocked outside the preferred area, and scaled by monocular
depth so that far-away pixels cost as much ground as they cover.

  # Ask the VLM and segment live:
  python3 click_plan_demo.py example_data/hiking_lake.jpg --seg_model api
  # Reuse maps saved by `example.py --save_images` (no API call):
  python3 click_plan_demo.py example_data/hiking_lake.jpg --maps_dir result/hiking_lake
  # Without a display, plan once and save the figure:
  python3 click_plan_demo.py example_data/hiking_lake.jpg --maps_dir result/hiking_lake --goal 200,320 --save plan.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vlm_nav.planning import build_cost_map, path_cost, plan_path, smooth_path


def parse_pixel(text):
    """'row,col' -> (row, col)."""
    row, col = (int(v) for v in text.split(","))
    return row, col


def resize_map(layer, size):
    """Resize an (H, W) float map to a PIL (width, height) size."""
    if layer.shape[::-1] == tuple(size):
        return layer
    return np.asarray(Image.fromarray(layer.astype(np.float32)).resize(size, Image.BILINEAR))


def load_maps(maps_dir, size):
    def load(name):
        return np.asarray(Image.open(os.path.join(maps_dir, f"{name}_gray.png")).convert("L"), np.float32) / 255.0

    return resize_map(load("risk"), size), resize_map(load("preferred_area"), size)


def segment_maps(image, args):
    from vlm_nav import VLMNavAgent

    agent = VLMNavAgent(args.config_path, args.task, segmentation_model=args.seg_model, vlm_model=args.vlm_model)
    print("Asking the VLM to analyze the image...")
    labels = agent.analyze(image)
    if labels is None:
        raise SystemExit("The VLM call failed.")
    for label, values in labels.items():
        print(f"- {label}: {values}")
    seg_return = agent.segment(image, labels)
    if seg_return is None:
        raise SystemExit("Segmentation failed.")
    layers, names = seg_return
    maps = {name: resize_map(layers[i].cpu().numpy(), image.size) for i, name in enumerate(names)}
    return maps["risk"], maps["preferred_area"]


class ClickPlanner:
    def __init__(self, image, risk, cost, start):
        self.image, self.risk, self.cost = image, risk, cost
        self.start, self.goal = start, None
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.draw()

    def on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        pixel = (int(round(event.ydata)), int(round(event.xdata)))
        if event.button == 3:
            self.start = pixel
        else:
            self.goal = pixel
        self.draw()

    def draw(self):
        ax = self.ax
        ax.clear()
        ax.imshow(self.image)
        blocked = np.isinf(self.cost)
        ax.imshow(np.ma.masked_where(blocked, self.risk), cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.35)
        ax.imshow(np.ma.masked_where(~blocked, blocked), cmap="gray", vmin=0, vmax=1, alpha=0.45)
        ax.plot(self.start[1], self.start[0], "o", color="cyan", markersize=10, label="start (right click)")
        title = "Left click a goal"
        if self.goal is not None:
            ax.plot(self.goal[1], self.goal[0], "*", color="magenta", markersize=16, label="goal (left click)")
            path = plan_path(self.cost, self.start, self.goal)
            if path is None:
                title = "No path: the goal is not reachable on preferred ground"
            else:
                smooth = smooth_path(path)
                ax.plot(smooth[:, 1], smooth[:, 0], "-", color="magenta", linewidth=3)
                title = f"A* path: {len(path)} px, cost {path_cost(self.cost, path):.0f}"
        ax.set_title(f"{title}  (green = low risk, grey = outside preferred area)")
        ax.legend(loc="upper right")
        ax.axis("off")
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_path", type=str, help="The camera image")
    parser.add_argument("--maps_dir", type=str, help="Folder with risk_gray.png and preferred_area_gray.png")
    parser.add_argument("-c", "--config_path", type=str, default="config/prompts.yaml")
    parser.add_argument("-t", "--task", type=str, default="nature_hiking")
    parser.add_argument("-s", "--seg_model", type=str, default="SAN", help="SAN (in-process) or api (server)")
    parser.add_argument("--vlm_model", type=str, default="gpt-6.1-sol", help="OpenAI (gpt-*) or Gemini (gemini-*)")
    parser.add_argument("--no_depth", action="store_true", help="Plan on pixels without depth scaling")
    parser.add_argument("--risk_weight", type=float, default=100.0, help="Cost of a fully risky pixel")
    parser.add_argument("--start", type=parse_pixel, help="Start pixel 'row,col' (default: bottom centre)")
    parser.add_argument("--goal", type=parse_pixel, help="Goal pixel 'row,col', to plan without clicking")
    parser.add_argument("--save", type=str, help="Save the figure here instead of opening a window")
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    image.thumbnail((640, 640))
    if args.maps_dir:
        risk, preferred = load_maps(args.maps_dir, image.size)
    else:
        risk, preferred = segment_maps(image, args)

    depth = None
    if not args.no_depth:
        from vlm_nav.models.depth_model import DepthEstimator

        print("Estimating depth...")
        depth = DepthEstimator()(image)

    cost = build_cost_map(risk, preferred, depth, risk_weight=args.risk_weight)
    start = args.start or (image.size[1] - 1, image.size[0] // 2)
    planner = ClickPlanner(np.asarray(image), risk, cost, start)
    if args.goal is not None:
        planner.goal = args.goal
        planner.draw()
    if args.save:
        planner.fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
