#!/usr/bin/env python3
import os
import numpy as np
import torch
import PIL
from matplotlib import pyplot as plt
import math
import io
from typing import List, Tuple
from dataclasses import dataclass

from vlm_nav.models.san_models import resolve_model


ABSOLUTE_FILE_PATH = __file__
ABSOLUTE_DIR_PATH = os.path.dirname(ABSOLUTE_FILE_PATH)


def softmax(x, dim=None, temp=1.0):
    x_temp = x / temp
    e_x = torch.exp(x_temp - torch.max(x_temp, dim=dim, keepdim=True).values)
    return e_x / e_x.sum(dim=dim, keepdim=True)


def generate_vocab_image(normalized, vocab):
    num_classes = normalized.shape[0]
    cols = max(3, num_classes)
    rows = math.ceil(num_classes / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle("Vocabulary and Corresponding Heatmaps")

    for i, label in enumerate(vocab):
        ax = axs.flat[i]
        heatmap = normalized[i].cpu().numpy()
        im = ax.imshow(heatmap, cmap="jet", interpolation="nearest")
        ax.title.set_text(label)
        fig.colorbar(im, ax=ax)

    for i in range(num_classes, rows * cols):
        axs.flat[i].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


@dataclass
class SegmentationParameters:
    vocab_query_topic: str = "/vocab_query"
    segmentation_model: str = "SAN"
    model_config_file: str = "../configs/san_clip_vit_res4_coco.yaml"
    model_weight_path: str = os.path.join(ABSOLUTE_DIR_PATH, "../saved_models/san_vit_b_16.pth")
    other_category_weight: float = 0.3
    softmax_temperature: float = 0.3
    half_precision: bool = False


class SemanticLabelProcessor:
    def __init__(self, cfg: SegmentationParameters):
        self.cfg = cfg
        self.semantic_model = resolve_model(cfg.segmentation_model, cfg)

    def segment_image(self, image: PIL.Image, vocab_query: list) -> torch.Tensor:
        """
        This function segments the input image by identifying the semantic elements based on the provided vocabulary query.
        It processes the image through the semantic segmentation model and returns the segmentation results along with the vocabulary.

        Args:
            image (PIL.Image): The input image to segment.
            vocab_query (list): The vocabulary terms to query in the segmentation.

        Returns:
            Tuple[torch.Tensor, List[str]]: A tuple containing the segmented image tensor and the list of vocabulary terms identified in the image.
        """
        # Run the semantic segmentation model
        vocab_query = list(vocab_query)
        if "other" not in vocab_query:
            vocab_query.append("other")
        results = self.semantic_model["model"](
            image=image,
            vocabulary=vocab_query,
            augment_vocabulary=False,
        )
        sem_seg = results["result"]
        vocab = results["vocabulary"]

        if isinstance(sem_seg, np.ndarray):
            sem_seg = torch.from_numpy(sem_seg.copy())
        return sem_seg, vocab

    def process(
        self,
        image: PIL.Image,
        labels: dict,
        default_values: dict,
        output_debug_info: bool = False,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor, List[str]]:
        """
        This function segments the input image by identifying the semantic elements based on the provided vocabulary query.
        It processes the image through the semantic segmentation model and returns the segmentation results along with the vocabulary.

        Args:
            image (PIL.Image): The input image to segment.
            semantic_traversabilities (dict): The semantic traversability labels to query in the segmentation.

        Returns:
            Tuple[torch.Tensor, List[str]]: A tuple containing the segmented image tensor and the list of vocabulary terms identified in the image.
        """
        vocab_query = labels.keys()
        sem_seg, vocab = self.segment_image(image, vocab_query)
        if sem_seg.dtype == torch.float16:
            sem_seg = sem_seg.to(torch.float32)
        layers, layer_names, normalized = self.calculate_output_fields(sem_seg, vocab, labels, default_values)
        if output_debug_info:
            debug_info = {"sem_seg": sem_seg, "normalized": normalized, "vocab": vocab}
            return layers, layer_names, debug_info
        else:
            return layers, layer_names

    def calculate_output_fields(
        self, sem_seg: torch.Tensor, vocab: list, labels: dict, default_values: dict
    ) -> torch.Tensor:
        """
        This function converts the semantic segmentation into a traversability map based on the provided semantic traversability labels.

        Args:
            sem_seg (torch.Tensor): The semantic segmentation tensor.
            vocab (list): The vocabulary terms identified in the image.
            semantic_traversabilities (dict): The semantic traversability labels to query in the segmentation.

        Returns:
            torch.Tensor: The traversability map.
        """
        normalized = self.normalize_semantic(sem_seg, vocab)

        first_key = next(iter(labels))
        layer_names = list(labels[first_key].keys())

        if "other" in vocab:
            filtered_initial_values = {k: v for k, v in default_values.items() if k in layer_names}
            labels["other"] = filtered_initial_values

        layers = []
        for layer_name in layer_names:
            layer = self.calculate_layer(normalized, vocab, layer_name, labels, default_values)
            layers.append(layer)
        result = torch.stack(layers, dim=0)
        return result, layer_names, normalized

    def normalize_semantic(self, sem_seg: torch.Tensor, vocab: list) -> torch.Tensor:
        """
        Normalize the semantic segmentation tensor.

        This function applies a normalization process to the semantic segmentation tensor
        to ensure that the values are scaled appropriately for further processing. It includes
        an optional weighting for the 'other' category and applies a softmax function with
        temperature scaling to the entire tensor.

        Args:
            sem_seg (torch.Tensor): The semantic segmentation tensor before normalization.
            vocab (list): The list of vocabulary terms corresponding to the segmentation classes.

        Returns:
            torch.Tensor: The normalized semantic segmentation tensor.
        """
        # If the 'other' category is present in the vocabulary, apply a special weight to it
        # This helps to adjust the influence of the 'other' category in the segmentation
        if "other" in vocab:
            other_idx = vocab.index("other")
            sem_seg[other_idx] *= self.cfg.other_category_weight
        # Apply a filter to reduce the difference between large and small values
        # Apply tanh filter only for large values in the semantic segmentation tensor
        sem_seg = torch.tanh(sem_seg / 2.0) * 2.0
        normalized = softmax(sem_seg, dim=0, temp=self.cfg.softmax_temperature)

        normalized = torch.clamp(normalized, 0, 1)
        return normalized

    def can_convert_to_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def calculate_layer(
        self, normalized: torch.Tensor, vocab: List[str], layer_name: str, vocab_values: dict, default_values: dict
    ) -> torch.Tensor:
        # Filter out entries in vocab_values that cannot be converted to float for the current layer_name

        vocab_values = {
            k: v for k, v in vocab_values.items() if k in vocab and self.can_convert_to_float(v[layer_name])
        }
        vocab_filtered = [k for k in vocab if k in vocab_values]
        layer_indices = [vocab.index(k) for k in vocab_filtered]
        normalized = torch.index_select(normalized, 0, torch.tensor(layer_indices, device=normalized.device))

        values = torch.tensor(
            [
                (
                    float(vocab_values[k][layer_name])
                    if isinstance(vocab_values[k][layer_name], (int, float, bool, str))
                    else default_values[layer_name]
                )
                for k in vocab_filtered
            ]
        ).to(normalized.device)
        layer = normalized * values.view(-1, 1, 1)
        layer = torch.sum(layer, dim=0)
        return layer
