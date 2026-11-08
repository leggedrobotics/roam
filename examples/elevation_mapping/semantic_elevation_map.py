"""Fuse ROAM's risk, steppable and preferred-area images into an elevation map with elevation_mapping_cupy.

Replays a 30 s clip from the Seealpsee hike (lidar point clouds, camera poses and the semantic images the
robot computed on board), runs elevation_mapping_cupy's GPU core directly (no ROS), and shows the fused
layers top-down. Click on the map to plan a path from the robot with A* over the fused risk map.

  python3 semantic_elevation_map.py                     # replay ROAM's on-board output (no API key)
  python3 semantic_elevation_map.py --source segment    # re-segment the images with the recorded VLM labels
  python3 semantic_elevation_map.py --source vlm        # ask the VLM again, then segment
  python3 semantic_elevation_map.py --goal 6,-3 --save map.png   # no display: plan to (x, y) in metres

Needs a CUDA GPU, cupy and elevation_mapping_cupy with semantic layers (see README.md next to this file).
"""

import argparse
import inspect
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vlm_nav.planning import build_cost_map, plan_path, smooth_path

CHANNELS = ["risk", "steppable", "preferred_area"]
# Fused alongside the ROAM channels so that the map knows which cells a camera has actually seen.
# The exponential fusion starts from zero, so without it an unseen cell would read as "no risk".
OBSERVED = "roam_observed"
UNKNOWN_RISK = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))


def load_clip(data_dir):
    clip = dict(np.load(os.path.join(data_dir, "clip.npz")))
    clip["cloud_xyz"] = clip["cloud_xyz"].astype(np.float32) * 0.01  # stored in centimetres
    with open(os.path.join(data_dir, "labels.json")) as f:
        clip["labels"] = json.load(f)
    return clip


def frame_image(data_dir, i):
    return Image.open(os.path.join(data_dir, "images", f"{i:04d}.jpg")).convert("RGB")


def semantic_frames(clip, data_dir, source, vlm_model, seg_model):
    """(F, H, W, 3) float images in CHANNELS order, from the recording or from vlm_nav."""
    if source == "recorded":
        return clip["sem"].astype(np.float32) / 255.0

    from vlm_nav import VLMNavAgent

    config = os.path.join(HERE, "..", "..", "vlm_nav", "config", "prompts.yaml")
    agent = VLMNavAgent(config, "nature_hiking", segmentation_model=seg_model, vlm_model=vlm_model, resize=False)
    if source == "segment":
        labels = clip["labels"]["label_estimation"]
    else:
        labels = agent.analyze(frame_image(data_dir, 0))
        if labels is None:
            raise SystemExit("The VLM call failed.")
    print("Labels:", json.dumps(labels, indent=1))

    height, width = clip["sem"].shape[1:3]
    frames = []
    for i in range(len(clip["sem"])):
        layers, names = agent.segment(frame_image(data_dir, i), dict(labels))
        layers = layers.cpu().numpy()
        frames.append(
            np.stack(
                [np.asarray(Image.fromarray(layers[names.index(c)]).resize((width, height))) for c in CHANNELS], -1
            )
        )
        print(f"segmented {i + 1}/{len(clip['sem'])}", end="\r")
    print()
    return np.stack(frames)


def make_map(resolution, map_length, weight_file):
    import cupy as cp

    if not hasattr(cp, "bool8"):  # main (ROS 1) still uses cp.bool8, which NumPy 2 era CuPy removed
        cp.bool8 = cp.bool_
    from elevation_mapping_cupy.elevation_mapping import ElevationMap
    from elevation_mapping_cupy.parameter import Parameter

    param = Parameter(
        use_chainer=False,
        weight_file=weight_file,
        plugin_config_file=os.path.join(HERE, "plugin_config.yaml"),
        resolution=resolution,
        map_length=map_length,
        subscriber_cfg={"roam": {"data_type": "image", "channels": CHANNELS + [OBSERVED]}},
        additional_layers=[],
        fusion_algorithms=["image_exponential"],
        image_channel_fusions={"default": "exponential"},
        # Lidar settings close to the ANYmal hiking setup.
        # ROS 2 branches scale point noise with range^2 (main with height^2); 0.002 keeps lidar returns
        # out to ~20 m on both, where a larger factor drops everything past ~5 m on ROS 2.
        sensor_noise_factor=0.002,
        min_valid_distance=0.5,
        max_height_range=1.5,
        ramped_height_range_a=10.0,
        max_ray_length=10.0,
        enable_drift_compensation=False,
        enable_overlap_clearance=False,
    )
    param.load_weights(weight_file)
    param.update()
    return ElevationMap(param)


def default_weight_file():
    import elevation_mapping_cupy

    package_dir = os.path.dirname(elevation_mapping_cupy.__file__)
    for candidate in [
        os.path.join(package_dir, "..", "config", "core", "weights.dat"),  # source checkout (ROS 2 branches)
        os.path.join(package_dir, "..", "..", "config", "core", "weights.dat"),  # source checkout (main, ROS 1)
        os.path.join(package_dir, "..", "..", "..", "..", "share", "elevation_mapping_cupy", "config", "core", "weights.dat"),
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise SystemExit("Could not find elevation_mapping_cupy's config/core/weights.dat; pass --weight_file.")


def replay(elevation_map, clip, sem):
    """Feed point clouds and semantic images to the map in time order."""
    import cupy as cp

    events = [(t, "cloud", i) for i, t in enumerate(clip["cloud_stamps"])]
    events += [(t, "image", i) for i, t in enumerate(clip["sem_stamps"])]
    height, width = sem.shape[1:3]
    ones = np.ones((height, width), np.float32)
    takes_stamp = "source_stamp_ns" in inspect.signature(elevation_map.input_pointcloud).parameters
    for _, kind, i in sorted(events):
        if kind == "cloud":
            T_base = clip["cloud_T_odom_base"][i]
            elevation_map.move_to(T_base[:3, 3], T_base[:3, :3])
            points = clip["cloud_xyz"][clip["cloud_splits"][i] : clip["cloud_splits"][i + 1]]
            T = cp.asarray(clip["cloud_T_odom_lidar"][i], dtype=cp.float32)  # main (ROS 1) does not convert
            args = [cp.asarray(points), ["x", "y", "z"], T[:3, :3], T[:3, 3], 0.0, 0.0]
            if takes_stamp:  # ROS 2 branches only count a cloud as an observation when it carries its stamp
                args.append(int(clip["cloud_stamps"][i] * 1e9))
            elevation_map.input_pointcloud(*args)
            elevation_map.update_normal(elevation_map.elevation_map[0])
        else:
            # input_image takes the map-to-camera transform (world to camera extrinsics).
            T = np.linalg.inv(clip["sem_T_odom_cam"][i])
            layers = [sem[i, ..., k] for k in range(len(CHANNELS))] + [ones]
            elevation_map.input_image(
                layers, CHANNELS + [OBSERVED], T[:3, :3], T[:3, 3], clip["K"], clip["D"],
                str(clip["distortion_model"]), height, width,
            )


def read_layers(elevation_map):
    n = elevation_map.cell_n - 2
    out = {}
    for name in ["elevation"] + CHANNELS + [OBSERVED]:
        data = np.zeros((n, n), np.float32)
        elevation_map.get_map_with_name_ref(name, data)
        out[name] = data
    for name in CHANNELS:  # cells no camera has seen are unknown, not zero
        out[name][out[OBSERVED] < 0.5] = np.nan
    return out


class MapView:
    """Top-down layers in grid_map orientation: +x (forward at the start) up, +y left."""

    def __init__(self, layers, center, resolution, robot_xy, trajectory, image):
        self.layers, self.center, self.resolution = layers, center, resolution
        self.robot_xy, self.trajectory, self.goal = robot_xy, trajectory, None
        n = layers["elevation"].shape[0]
        half = n * resolution / 2
        # grid_map convention: row i -> x = cx + half - (i + 0.5) * res, col j -> y = cy + half - (j + 0.5) * res
        self.extent = [center[1] + half, center[1] - half, center[0] - half, center[0] + half]
        # Cells a camera has seen use the fused values. Unseen cells with elevation count as unknown (risk 0.5);
        # unseen cells without elevation are blocked, except where the robot walked (the lidar's blind spot).
        seen = np.isfinite(layers["risk"])
        walked = self.walked_mask(trajectory, radius=0.5)
        risk = np.where(seen, layers["risk"], np.where(walked, 0.0, UNKNOWN_RISK))
        preferred = np.where(seen, layers["preferred_area"], 1.0)
        self.cost = build_cost_map(risk, preferred)
        self.cost[~seen & ~walked & ~np.isfinite(layers["elevation"])] = np.inf
        self.fig, axs = plt.subplots(1, 5, figsize=(21, 4.8))
        self.axs = axs
        axs[0].imshow(image)
        axs[0].set_title("last camera frame")
        axs[0].axis("off")
        cmaps = {"elevation": "terrain", "risk": "RdYlGn_r", "steppable": "RdYlGn", "preferred_area": "RdYlGn"}
        for ax, name in zip(axs[1:5], ["elevation"] + CHANNELS):
            kw = {} if name == "elevation" else {"vmin": 0, "vmax": 1}
            im = ax.imshow(layers[name], cmap=cmaps[name], extent=self.extent, **kw)
            self.fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"fused {name}")
        for ax in axs[1:]:
            ax.plot(trajectory[:, 1], trajectory[:, 0], "-", color="k", linewidth=1)
            ax.plot(robot_xy[1], robot_xy[0], "o", color="cyan", markeredgecolor="k")
            ax.set_xlabel("y [m]")
        axs[1].set_ylabel("x [m]")
        self.path_lines = []
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.tight_layout()

    def walked_mask(self, trajectory, radius):
        n = self.layers["elevation"].shape[0]
        rows, cols = np.mgrid[0:n, 0:n]
        xy = self.to_xy(np.stack([rows.ravel(), cols.ravel()], 1))
        mask = np.zeros(n * n, bool)
        for x, y in trajectory[:, :2]:
            mask |= (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2 < radius**2
        return mask.reshape(n, n)

    def to_cell(self, xy):
        n = self.cost.shape[0]
        half = n * self.resolution / 2
        i = int((self.center[0] + half - xy[0]) / self.resolution)
        j = int((self.center[1] + half - xy[1]) / self.resolution)
        return int(np.clip(i, 0, n - 1)), int(np.clip(j, 0, n - 1))

    def to_xy(self, cells):
        half = self.layers["elevation"].shape[0] * self.resolution / 2
        x = self.center[0] + half - (cells[:, 0] + 0.5) * self.resolution
        y = self.center[1] + half - (cells[:, 1] + 0.5) * self.resolution
        return np.stack([x, y], 1)

    def on_click(self, event):
        if event.inaxes not in self.axs[1:] or event.xdata is None:
            return
        self.plan((event.ydata, event.xdata))

    def plan(self, goal_xy):
        for line in self.path_lines:
            line.remove()
        self.path_lines = []
        path = plan_path(self.cost, self.to_cell(self.robot_xy), self.to_cell(goal_xy))
        title = "no path: the goal is cut off by non-preferred ground" if path is None else f"A* path: {len(path) * self.resolution:.1f} m"
        for ax in self.axs[1:]:
            self.path_lines += ax.plot(goal_xy[1], goal_xy[0], "*", color="magenta", markersize=14)
            if path is not None:
                xy = self.to_xy(smooth_path(path))
                self.path_lines += ax.plot(xy[:, 1], xy[:, 0], "-", color="magenta", linewidth=2.5)
        self.fig.suptitle(f"Click a goal on any map. {title}")
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "data", "seealpsee_trail"))
    parser.add_argument("--source", choices=["recorded", "segment", "vlm"], default="recorded")
    parser.add_argument("--vlm_model", default="gpt-6.1-sol")
    parser.add_argument("--seg_model", default="SAN", help="SAN (in-process) or api (server)")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--map_length", type=float, default=30.0)
    parser.add_argument("--weight_file", help="elevation_mapping_cupy config/core/weights.dat")
    parser.add_argument("--goal", help="Plan to 'x,y' (metres, odom frame) without clicking")
    parser.add_argument("--save", help="Save the figure here instead of opening a window")
    args = parser.parse_args()

    clip = load_clip(args.data)
    sem = semantic_frames(clip, args.data, args.source, args.vlm_model, args.seg_model)
    elevation_map = make_map(args.resolution, args.map_length, args.weight_file or default_weight_file())
    replay(elevation_map, clip, sem)
    layers = read_layers(elevation_map)

    robot_xy = clip["robot_xyz"][-1, :2]
    view = MapView(
        layers, elevation_map.center.get(), args.resolution, robot_xy, clip["robot_xyz"],
        frame_image(args.data, len(clip["sem"]) - 1),
    )
    if args.goal:
        view.plan(tuple(float(v) for v in args.goal.split(",")))
    if args.save:
        view.fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
