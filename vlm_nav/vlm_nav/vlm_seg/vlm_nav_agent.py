import json
import os
import numpy as np
from PIL import Image
from typing import Dict, Union, Tuple, Literal, List

from vlm_nav.vlm_clients import create_client

from .prompt_generator import ImageLabelPromptGenerator, create_prompt_generator_from_yaml
from .semantic_label_processor import SemanticLabelProcessor, SegmentationParameters


class VLMNavAgent:
    def __init__(
        self,
        prompt_cfg_path: str,
        prompt_setup_key: str,
        resize: bool = True,
        vlm_model: str = "gemini-4-hydrogen",
        segmentation_model: Literal["SAN", "api"] = "SAN",
        resize_shape: Tuple[int, int] = (640, 480),
        use_memory: bool = True,
        half_precision: bool = False,
        **vlm_kwargs,
    ):
        """vlm_kwargs go to the VLM client (see vlm_nav.vlm_clients.create_client)."""
        self.prompt_generator = create_prompt_generator_from_yaml(prompt_cfg_path, prompt_setup_key, use_memory)
        self.vlm_model = vlm_model
        self.vlm_kwargs = vlm_kwargs
        self._vlm_client = None
        self.response_schema = self.prompt_generator.response_schema()
        semantic_cfg = SegmentationParameters(segmentation_model=segmentation_model, half_precision=half_precision)
        self.semantic_label_processor = SemanticLabelProcessor(semantic_cfg)
        self.resize = resize
        self.resize_shape = resize_shape

    @property
    def vlm_client(self):
        # Created on first use, so segmenting with given labels needs no API key.
        if self._vlm_client is None:
            self._vlm_client = create_client(self.vlm_model, **self.vlm_kwargs)
        return self._vlm_client

    def convert_image_to_pil_format(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Unknown type: {type(image)}")
        if self.resize:
            image.thumbnail(self.resize_shape)
        return image

    def reset_memory(self):
        self.prompt_generator.reset()

    def analyze(
        self,
        image: Union[str, np.ndarray, Image.Image, List[Union[str, np.ndarray, Image.Image]]],
        return_response: bool = False,
    ) -> Union[Dict, Tuple[Dict, Dict], None]:
        images = image if isinstance(image, list) else [image]
        images = [self.convert_image_to_pil_format(img) for img in images]
        prompt = self.prompt_generator.get_prompt()
        report = self.vlm_client.ask(images, prompt, self.response_schema)
        if report is None:
            print("No response from VLM")
            return (None, None) if return_response else None
        response = self.prompt_generator.report_to_dict(report)
        response_return = {"prompt": prompt, "response": json.dumps(response, indent=2)}
        self.prompt_generator.update(response)
        labels = self.prompt_generator.parse_json_response(response)
        if return_response:
            return labels, response_return
        else:
            return labels

    def segment(
        self,
        image: Union[str, np.ndarray, Image.Image],
        labels: dict,
        output_debug_info: bool = False,
    ) -> np.ndarray:
        image = self.convert_image_to_pil_format(image)
        default_values = self.prompt_generator.get_default_values()
        try:
            return self.semantic_label_processor.process(
                image, labels, default_values, output_debug_info=output_debug_info
            )
        except Exception as e:
            print(e)
            return

    def analyze_and_segment(
        self,
        image: Union[str, np.ndarray, Image.Image],
        output_debug_info: bool = False,
    ) -> Union[Dict, Tuple[Dict, Dict], None]:
        labels, response = self.analyze(image, return_response=True)
        if labels is None:
            return
        seg_return = self.segment(image, labels, output_debug_info=True)
        if seg_return is None:
            return
        seg_image, _, seg_debug_info = seg_return
        if output_debug_info:
            debug_info = {"labels": labels, "response": response, "segmentation": seg_debug_info}
            return seg_image, debug_info
        else:
            return seg_image
