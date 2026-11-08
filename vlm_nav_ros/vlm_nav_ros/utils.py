import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
import os
from ament_index_python.packages import get_package_share_directory
import rclpy

def color_map(N=256, normalized=False):
    """Create a color map for the class labels.

    Args:
        N (int):
        normalized (bool):
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N + 1, 3), dtype=dtype)
    for i in range(N + 1):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    return cmap[1:]


def rotate_image(image, angle):
    if angle < 0:
        angle = angle + 360
    if isinstance(image, PIL.Image.Image):
        if angle == 0:
            return image
        elif angle == 90:
            return image.transpose(PIL.Image.ROTATE_90)
        elif angle == 180:
            return image.transpose(PIL.Image.ROTATE_180)
        elif angle == 270:
            return image.transpose(PIL.Image.ROTATE_270)
        else:
            raise ValueError("Invalid angle: %d" % angle)
    elif isinstance(image, torch.Tensor):
        if angle == 0:
            return image
        elif angle == 90:
            return torch.rot90(image, 1, [1, 2])
        elif angle == 180:
            return torch.rot90(image, 2, [1, 2])
        elif angle == 270:
            return torch.rot90(image, 3, [1, 2])
        else:
            raise ValueError("Invalid angle: %d" % angle)
    elif isinstance(image, np.ndarray):
        if angle == 0:
            return image
        elif angle == 90:
            return np.rot90(image, 1, [1, 2])
        elif angle == 180:
            return np.rot90(image, 2, [1, 2])
        elif angle == 270:
            return np.rot90(image, 3, [1, 2])
        else:
            raise ValueError("Invalid angle: %d" % angle)
    else:
        raise TypeError("Invalid image type: %s" % type(image))


def resize_image(image, max_size=(480, 480)):
    orig_size = image.size
    image.thumbnail(max_size)
    resize_ratio = image.size[0] / orig_size[0]
    return image, resize_ratio


def colorize_semantics(sem_seg, vocab):
    """
    This function colorizes the semantic segmentation map.

    Args:
        sem_seg (torch.Tensor): The semantic segmentation map.
        vocab (list): The vocabulary used for semantic segmentation.

    Returns:
        np.ndarray: The colorized semantic segmentation map.
    """
    if isinstance(sem_seg, np.ndarray):
        sem_seg = torch.from_numpy(sem_seg)
    seg_map = sem_seg.argmax(dim=0).cpu().numpy()  # (H, W)
    cmap = color_map(N=len(sem_seg) + 1)
    colorized_image = cmap[seg_map].astype(np.uint8)
    return colorized_image


def colorize_labels(risk_map):
    """
    This function colorizes the risk map.

    Args:
        risk_map (torch.Tensor): The risk map.

    Returns:
        np.ndarray: The colorized risk map.
    """
    colorized_images = []
    if isinstance(risk_map, np.ndarray) and risk_map.ndim == 3:
        risk_map = risk_map  # Convert from [H, W, 1] to [H, W]
    else:
        raise ValueError("risk_map should be a numpy array with shape [H, W, 1]")

    for i in range(risk_map.shape[0]):
        img = risk_map[i]
        colorized_image = plt.get_cmap("jet")(1.0 - img)[
            :, :, :3
        ]  # Invert risk_map to make small values blue and large values red
        colorized_image = (colorized_image * 255).astype(np.uint8)
        colorized_images.append(colorized_image)
    colorized_image = np.concatenate(colorized_images, 1)
    return colorized_image

def get_path_using_ros2pkg(vlm_nav_config_path: str) -> str:
    """
    Resolves a ROS2-style package path in the given config path.
    Example: '$(find vlm_nav_ros)/config/prompts.yaml' -> '/absolute/path/to/vlm_nav_ros/config/prompts.yaml'
    """
    if "$(find " in vlm_nav_config_path:
        try:
            # Extract package name
            package_name = vlm_nav_config_path.split("$(find ")[1].split(")")[0].strip()
            # Get package share directory
            package_path = get_package_share_directory(package_name)
            # Get relative path after ')'
            relative_path = vlm_nav_config_path.split(")")[1].lstrip("/")
            # Construct full path
            vlm_nav_config_path = os.path.join(package_path, relative_path)
        except Exception as e:
            # In ROS2, you can use rclpy.logging
            node = rclpy.get_logger() if rclpy.ok() else None
            if node:
                node.error(f"Could not resolve vlm_nav_config_path: {e}")
            else:
                print(f"[ERROR] Could not resolve vlm_nav_config_path: {e}")
    return vlm_nav_config_path


def format_vlm_response_string(response, labels):
    """
    This function creates a string from the VLM response.

    Args:
        response (dict): The VLM response.
        labels (dict): The estimated labels.

    Returns:
        str: The string representation of the VLM response.
    """
    response_str = "Response: {}".format(response.get("response", "N/A"))
    prompt_str = "Prompt: {}".format(response.get("prompt", "N/A"))


    response_str = "{}\n{}\n".format(response_str, prompt_str)

    labels_str = ""
    for key, value in labels.items():
        labels_str += "{}:".format(key)
        for sub_key, sub_value in value.items():
            labels_str += "  {}: {}".format(sub_key, sub_value)
        labels_str += "\n"

    return response_str, labels_str
