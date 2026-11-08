import os
import numpy as np
import json
from typing import Dict, List, Type, Union

from pydantic import BaseModel, Field, create_model


class Memory(object):
    def __init__(self, memory_length: int = 2):
        self.memory_length = memory_length
        self.memory = []

    def add(self, text: str):
        self.memory.append(text)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

    def reset(self):
        self.memory = []

    def get_string(self):
        memory_strings = []
        for i, memory in enumerate(reversed(self.memory)):
            age_label = f"t - {len(self.memory) - 1 - i}" if i < len(self.memory) - 1 else "t"
            memory_strings.append(f"{age_label}: {memory}")
        str_memory = ""
        for memory in memory_strings:
            str_memory += memory + "\n"
        return str_memory

    def get_merged_memory(self):
        """Assuming memory is json."""
        merged_memory = {}
        try:
            for memory in reversed(self.memory):
                d = json.loads(memory)
                for key, value in d.items():
                    if isinstance(value, dict):
                        if key not in merged_memory:
                            merged_memory[key] = value
                        else:
                            for k, v in value.items():
                                try:
                                    if k not in merged_memory[key]:
                                        merged_memory[key][k] = float(v)
                                    else:
                                        merged_memory[key][k] = (float(v) + float(merged_memory[key][k])) / 2
                                except ValueError:
                                    print(f"Value {v} is not a float for label {key}.")
                    elif isinstance(value, float) or isinstance(value, int):
                        if key not in merged_memory:
                            merged_memory[key] = float(value)
                        else:
                            merged_memory[key] = (float(value) + float(merged_memory[key])) / 2
        except json.JSONDecodeError:
            print("Memory is not json.")
            return self.get_string()
        return json.dumps(merged_memory, indent=4)


class ImageLabelPromptGenerator(object):
    def __init__(
        self,
        task_prompt: str,
        robot_prompt: str,
        system_prompt: str,
        use_memory: bool = True,
        memory_length: int = 3,
    ):
        self.task_prompt = task_prompt
        self.robot_prompt = robot_prompt
        self.system_prompt = system_prompt

        # Output fields information is stored in a dictionary with the output_json_key as the key
        self.output_fields = {}
        self.use_memory = use_memory
        if use_memory:
            self.previous_estimate = Memory(memory_length=memory_length)
            self.memory = Memory(memory_length=memory_length)

    def __str__(self):
        """
        Returns a formatted string representation of the ImageLabelPromptGenerator instance.
        """
        prompts_info = (
            f"Task Prompt: {self.task_prompt}\n"
            f"Robot Prompt: {self.robot_prompt}\n"
            f"System Prompt: {self.system_prompt}\n"
            f"Use Memory: {self.use_memory}\n"
        )
        if self.use_memory:
            prompts_info += f"Memory: {self.memory}\n"
        prompts_info += "Output Fields:\n"
        for key, value in self.output_fields.items():
            prompts_info += f"  {key}: {value['prompt']} (Default: {value['default']})\n"
        return prompts_info

    def add_output_field(
        self, prompt: str, output_json_key: str, output_json_value: str, default_value: Union[float, bool]
    ):
        """
        Adds a new output field to the generator's configuration with a default value.

        This method allows for the addition of a new field that will be included in the JSON output.
        It associates a prompt with a specific key and a descriptive value, along with a default value.

        Args:
            prompt (str): The prompt text that corresponds to the output field.
            output_json_key (str): The key under which the output value will be stored in the JSON.
            output_json_value (str): The descriptive value to be associated with the key in the JSON output.
            default_value (Union[float, bool]): The default value for the output field.

        Raises:
            ValueError: If the output_json_key already exists in the output_fields dictionary.
        """
        if output_json_key not in self.output_fields:
            self.output_fields[output_json_key] = {
                "prompt": prompt,
                "value": output_json_value,
                "default": default_value,
            }
        else:
            raise ValueError(f"Output field '{output_json_key}' already exists.")

    def get_default_values(self) -> Dict:
        """
        Returns the default values for the output fields.

        This method returns a dictionary with the default values for each output field that has been added to the
        generator's configuration.

        Returns:
            Dict: A dictionary with the default values for the output fields.
        """
        default_values = {}
        for key, value in self.output_fields.items():
            default_values[key] = value["default"]
        return default_values

    def update(self, response_json: Dict):
        """
        Updates the internal memory state with the response from the analysis.

        This method takes the 'memory' field from the provided response dictionary and updates the internal memory state of the generator. This is typically used to maintain context or state across multiple analysis cycles.

        Args:
            response (Dict): A dictionary containing the 'memory' field with the text to be used for updating the generator's memory state.
        """
        if self.use_memory:
            if "memory" in response_json:
                self.memory.add(response_json["memory"])
            else:
                print("[WARNING] Memory not found in response.")
            if "label_estimation" in response_json:
                self.previous_estimate.add(json.dumps(response_json["label_estimation"], indent=4))

    def reset(self):
        """
        Resets the internal memory state of the generator.

        This method resets the internal memory state of the generator to an empty string. This is typically used when starting a new analysis cycle.
        """
        if self.use_memory:
            self.memory.reset()
            self.previous_estimate.reset()

    def get_prompt(self) -> str:
        """
        Constructs the full prompt to be used for analysis.

        This method combines various components of the prompt including task, robot, system, output fields, JSON structure, and memory prompts into a single string that can be used to guide the analysis process.

        Returns:
            str: The complete prompt string for analysis.
        """
        prompt = self.task_prompt + "\n"
        prompt += self.robot_prompt + "\n"
        prompt += self._output_fields_prompt() + "\n"
        prompt += self.system_prompt + "\n"
        if self.use_memory:
            prompt += self._generate_memory_prompt() + "\n"
        prompt += self._generate_json_prompt() + "\n"
        return prompt

    def response_schema(self) -> Type[BaseModel]:
        """Pydantic schema the VLM must answer in: free-text analysis, one entry per terrain label, and memory.

        The fields are ordered so the model writes its analysis before committing to values.
        """
        label_fields = {"label": (str, Field(description="Terrain element as a short snake_case label"))}
        for key, value in self.output_fields.items():
            label_fields[key] = (float, Field(description=value["value"]))
        TerrainLabel = create_model("TerrainLabel", **label_fields)
        return create_model(
            "TerrainReport",
            analysis=(str, Field(description="Your analysis of the scene")),
            labels=(List[TerrainLabel], Field(description="Every terrain element with its assigned values")),
            memory=(str, Field(description="Memory text useful to keep consistency")),
        )

    def report_to_dict(self, report: BaseModel) -> Dict:
        """Convert a TerrainReport into {"analysis", "memory", "label_estimation": {label: {field: value}}}."""
        label_estimation = {}
        for entry in report.labels:
            values = entry.model_dump()
            label_estimation[values.pop("label")] = values
        return {"analysis": report.analysis, "memory": report.memory, "label_estimation": label_estimation}

    def parse_json_response(self, response: Dict) -> Dict:
        """Parse image segmentation labels from the response."""
        if "label_estimation" in response:
            return response["label_estimation"]
        else:
            raise ValueError(f"Response does not contain 'label_estimation' field.")

    def _output_fields_prompt(self) -> str:
        prompt = "First output analysis. Then,\n"
        for value in self.output_fields.values():
            prompt += f"    {value['prompt']}\n"
        return prompt

    def _generate_json_prompt(self) -> str:
        # The structure itself is enforced by response_schema().
        fields = ", ".join(self.output_fields.keys())
        return (
            "**Conclude with a report.** It should categorize each terrain element (snake_case label) "
            f"with its assigned values ({fields}), and include a memory text useful to keep consistency."
        )

    def _generate_memory_prompt(self) -> str:
        if self.use_memory:
            memory_prompt = "Previous Estimate:\n" + "{previous_estimate}\n" + "Memory:\n" + "{memory}"
            return memory_prompt.format(
                previous_estimate=self.previous_estimate.get_merged_memory(), memory=self.memory.get_string()
            )
        return


def create_prompt_generator_from_yaml(yaml_file_path: str, setup_key: str, use_memory: bool = True):
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True

    def load_yaml_file(file_path: str):
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return yaml.load(file)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    prompts = load_yaml_file(yaml_file_path)

    # Check if the setup key exists in the prompts
    if setup_key not in prompts["setups"]:
        raise KeyError(f"The setup key '{setup_key}' does not exist in the provided YAML configuration.")
    setup = prompts["setups"][setup_key]

    # Check if the task, robot, and system keys exist in the prompts
    task_key = setup["task"]
    robot_key = setup["robot"]
    system_key = setup["system"]
    if task_key not in prompts["task"]:
        raise KeyError(f"Task key '{task_key}' not found in prompts.")
    if robot_key not in prompts["robot"]:
        raise KeyError(f"Robot key '{robot_key}' not found in prompts.")
    if system_key not in prompts["system"]:
        raise KeyError(f"System key '{system_key}' not found in prompts.")

    task_prompt = prompts["task"][task_key]
    robot_prompt = prompts["robot"][robot_key]
    system_prompt = prompts["system"][system_key]

    prompt_generator = ImageLabelPromptGenerator(
        task_prompt=task_prompt,
        robot_prompt=robot_prompt,
        system_prompt=system_prompt,
        use_memory=use_memory,
    )

    for field in setup["output_fields"]:
        output_field_data = prompts["output_fields"].get(field)
        if output_field_data is None:
            raise KeyError(f"Output field '{field}' not found in prompts.")
        prompt = output_field_data.get("prompt")
        value = output_field_data.get("value")
        default = output_field_data.get("default")
        if prompt is None or value is None or default is None:
            raise ValueError(f"Output field '{field}' is missing 'prompt', 'value', or 'default' keys.")
        prompt_generator.add_output_field(prompt, field, value, default)

    return prompt_generator
