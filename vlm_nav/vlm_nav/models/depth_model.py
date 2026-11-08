"""Monocular depth with Depth Anything V2 through Hugging Face transformers (pip install vlm_nav[depth])."""

import numpy as np
import torch
from PIL import Image


class DepthEstimator:
    """Relative depth for an RGB image: larger values are farther away. Units are arbitrary."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # The relative models predict inverse depth (disparity), at the image size after post-processing.
        disparity = self.processor.post_process_depth_estimation(outputs, target_sizes=[image.size[::-1]])[0]
        disparity = disparity["predicted_depth"].float().cpu().numpy()
        disparity = disparity / max(float(disparity.max()), 1e-6)
        return 1.0 / np.clip(disparity, 1e-3, None)
