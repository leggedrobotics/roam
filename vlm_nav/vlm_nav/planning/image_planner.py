"""A* path planning in image space over ROAM's risk and preferred-area maps.

All maps are (H, W) float arrays aligned with the camera image. Pixels are (row, col).
"""

from typing import Optional, Tuple

import numpy as np
import pyastar2d

Pixel = Tuple[int, int]


def build_cost_map(
    risk: np.ndarray,
    preferred: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    risk_weight: float = 100.0,
    preferred_threshold: float = 0.5,
    max_depth_scale: float = 50.0,
) -> np.ndarray:
    """Per-pixel step cost for A*: 1 + risk_weight * risk, blocked (inf) outside the preferred area.

    With a depth map, the cost is also scaled by depth relative to the nearest ground, because a pixel far
    away covers more ground than one at the robot's feet. Paths then trade risk against walking distance
    instead of pixel count. The cost never drops below 1, which keeps the A* heuristic admissible.
    """
    cost = 1.0 + risk_weight * np.clip(risk, 0.0, 1.0)
    if depth is not None:
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            near = np.percentile(depth[valid], 5)
            scale = np.clip(np.where(valid, depth, near) / near, 1.0, max_depth_scale)
            cost = cost * scale
    cost = cost.astype(np.float32)
    if preferred is not None:
        cost[preferred < preferred_threshold] = np.inf
    return cost


def nearest_free_pixel(cost: np.ndarray, pixel: Pixel) -> Optional[Pixel]:
    """The pixel itself if it is traversable, otherwise the closest traversable one (None if there is none)."""
    row, col = pixel
    if np.isfinite(cost[row, col]):
        return pixel
    free = np.argwhere(np.isfinite(cost))
    if len(free) == 0:
        return None
    nearest = free[np.argmin(((free - np.array(pixel)) ** 2).sum(axis=1))]
    return int(nearest[0]), int(nearest[1])


def smooth_path(path: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving-average smoothing that keeps both end points."""
    if len(path) < 3:
        return path.astype(float)
    window = min(window, len(path))
    padded = np.pad(path.astype(float), ((window // 2, window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)
    smoothed[0], smoothed[-1] = path[0], path[-1]
    return smoothed[: len(path)]


def plan_path(cost: np.ndarray, start: Pixel, goal: Pixel, snap: bool = True) -> Optional[np.ndarray]:
    """Lowest-cost 8-connected path from start to goal as an (N, 2) array of (row, col), or None.

    With snap, a start or goal on blocked ground moves to the nearest traversable pixel first.
    """
    if snap:
        start, goal = nearest_free_pixel(cost, start), nearest_free_pixel(cost, goal)
        if start is None or goal is None:
            return None
    return pyastar2d.astar_path(cost, tuple(start), tuple(goal), allow_diagonal=True)


def path_cost(cost: np.ndarray, path: np.ndarray) -> float:
    """Sum of step costs along a path (the quantity A* minimizes, up to diagonal weighting)."""
    return float(cost[path[:, 0], path[:, 1]].sum())
