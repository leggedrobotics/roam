import os

import numpy as np
from PIL import Image as PILImage
import torch
import matplotlib.pyplot as plt
import rospkg


def rotate_image(image, angle):
    if angle < 0:
        angle = angle + 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("Invalid angle: %d" % angle)
    if angle == 0:
        return image
    k = angle // 90
    if isinstance(image, PILImage.Image):
        return image.transpose({1: PILImage.ROTATE_90, 2: PILImage.ROTATE_180, 3: PILImage.ROTATE_270}[k])
    elif isinstance(image, torch.Tensor):
        return torch.rot90(image, k, [1, 2])
    elif isinstance(image, np.ndarray):
        return np.rot90(image, k, [1, 2])
    raise TypeError("Invalid image type: %s" % type(image))


def resize_image(image, max_size=(480, 480)):
    orig_size = image.size
    image.thumbnail(max_size)
    resize_ratio = image.size[0] / orig_size[0]
    return image, resize_ratio


def colorize_labels(risk_map):
    """Colorize each channel of a [C, H, W] map (blue = 1, red = 0) and tile them side by side."""
    if not (isinstance(risk_map, np.ndarray) and risk_map.ndim == 3):
        raise ValueError("risk_map should be a numpy array with shape [C, H, W]")
    colorized_images = []
    for img in risk_map:
        colorized_image = plt.get_cmap("jet")(1.0 - img)[:, :, :3]
        colorized_images.append((colorized_image * 255).astype(np.uint8))
    return np.concatenate(colorized_images, 1)


def resolve_config_path(path):
    """Resolve vlm_nav_config_path.

    Accepts an absolute path, a '$(find <pkg>)/...' path, or a path relative to the vlm_nav
    source folder next to this package (roam/vlm_nav).
    """
    rospack = rospkg.RosPack()
    if "$(find " in path:
        package_name = path.split("$(find ")[1].split(")")[0].strip()
        relative_path = path.split(")", 1)[1].lstrip("/")
        return os.path.join(rospack.get_path(package_name), relative_path)
    if os.path.isabs(path):
        return path
    # realpath, so a package symlinked into the catkin workspace still finds roam/vlm_nav.
    package_dir = os.path.realpath(rospack.get_path("vlm_nav_ros1"))
    return os.path.join(os.path.dirname(package_dir), "vlm_nav", path)


def format_vlm_response_string(response, labels):
    response_str = "Response: {}\nPrompt: {}\n".format(response.get("response", "N/A"), response.get("prompt", "N/A"))
    labels_str = ""
    for key, value in labels.items():
        labels_str += "{}:".format(key)
        for sub_key, sub_value in value.items():
            labels_str += "  {}: {}".format(sub_key, sub_value)
        labels_str += "\n"
    return response_str, labels_str
