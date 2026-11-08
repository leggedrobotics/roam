#!/usr/bin/env python3
import sys
import numpy as np
import torch
import PIL
import copy
from matplotlib import pyplot as plt
import threading
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_srvs.srv import SetBool
from cv_bridge import CvBridge
from rclpy.parameter import Parameter

from vlm_nav import VLMNavAgent
from vlm_nav_ros.utils import (
    color_map,
    rotate_image,
    resize_image,
    colorize_semantics,
    colorize_labels,
    format_vlm_response_string,
)
import os


class SensorProcessor:
    def __init__(self, node, image_topic, camera_info_topic, max_size, rotate, output_topic_prefix):
        self.node = node
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.max_size = max_size
        self.rotate = rotate
        self.output_topic_prefix = output_topic_prefix

        self.image_lock = threading.Lock()
        self.camera_info_lock = threading.Lock()
        self.pub_lock = threading.Lock()

        self.camera_info = None
        self.image = None
        self.header = None
        self.resize_ratio = None
        self.is_compressed = False

        self.bridge = CvBridge()

        self.init_subs()
        self.init_pubs()

    def init_subs(self):
        if "compressed" in self.image_topic:
            self.image_sub = self.node.create_subscription(
                CompressedImage, self.image_topic, self.image_callback, 1
            )
            self.is_compressed = True
        else:
            self.image_sub = self.node.create_subscription(
                Image, self.image_topic, self.image_callback, 1
            )
        self.camera_info_sub = self.node.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 1
        )

    def init_pubs(self):
        sem_image_topic = self.output_topic_prefix + "_sem"
        sem_viz_topic = self.output_topic_prefix + "_sem_viz"
        sem_camera_info_topic = self.camera_info_topic + "_sem"

        self.sem_image_pub = self.node.create_publisher(Image, sem_image_topic, 1)
        self.sem_viz_pub = self.node.create_publisher(Image, sem_viz_topic, 1)
        self.sem_camera_info_pub = self.node.create_publisher(CameraInfo, sem_camera_info_topic, 1)

    def get_image(self):
        with self.image_lock:
            return copy.deepcopy(self.image)

    def get_image_and_header(self):
        with self.image_lock:
            return copy.deepcopy(self.image), copy.deepcopy(self.header)

    def publish_images(self, sem_seg, sem_viz, vocab, header):
        with self.pub_lock:
            self.publish_image(self.sem_image_pub, sem_seg, header, encoding="passthrough", rotate_back=True)
            self.publish_image(self.sem_viz_pub, sem_viz, header, encoding="bgr8")
            if self.camera_info is not None:
                self.camera_info.header = header
                self.sem_camera_info_pub.publish(self.camera_info)

    def publish_image(self, pub, image, header, encoding="passthrough", rotate_back=False):
        if rotate_back and self.rotate:
            image = rotate_image(image, -self.rotate)
        if encoding == "passthrough":
            image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        if image.shape[2] > 10:
            self.node.get_logger().warn(
                f"Image in shape of [{image.shape[0]} x {image.shape[1]} x {image.shape[2]}] has too many channels to publish. Will cause great delay."
            )

        is_compressed = pub.msg_type == CompressedImage
        if is_compressed:
            image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        else:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        image_msg.header = header
        pub.publish(image_msg)

    def image_callback(self, data):
        if not data.data:
            print("Received empty image message")
            return

        try:
            if self.is_compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "rgb8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            image = PILImage.fromarray(cv_image)
            image = self.process_image(image)
            with self.image_lock:
                self.image = image
                self.header = data.header
        except Exception as e:
            self.node.get_logger().error(f"Could not convert image message to cv image: {e}")

    def camera_info_callback(self, data):
        try:
            camera_info_msg = self.process_camera_info(data)
            with self.camera_info_lock:
                self.camera_info = camera_info_msg
        except Exception as e:
            self.node.get_logger().error(f"Could not process camera info: {e}")

    def process_image(self, cv_image):
        if self.rotate:
            cv_image = rotate_image(cv_image, self.rotate)
        cv_image, self.resize_ratio = resize_image(cv_image, self.max_size)
        # Size of the published semantic image, which is rotated back to the camera's orientation.
        width, height = cv_image.size
        self.output_size = (height, width) if self.rotate % 180 == 90 else (width, height)
        return cv_image

    def process_camera_info(self, camera_info_msg: CameraInfo):
        if self.resize_ratio is None:
            return
        camera_info_msg = self.resize_camera_info_matrices(camera_info_msg, self.resize_ratio)
        return camera_info_msg

    def resize_camera_info_matrices(self, camera_info_msg, resize_ratio):
        # Take the size from the resized image: int(ratio * size) can be one pixel off PIL's rounding.
        camera_info_msg.width, camera_info_msg.height = self.output_size
        K = np.array(camera_info_msg.k).reshape((3, 3))
        P = np.array(camera_info_msg.p).reshape((3, 4))
        K[:2, :3] *= resize_ratio
        P[:2, :3] *= resize_ratio
        camera_info_msg.k = K[:3, :3].flatten().tolist()
        camera_info_msg.p = P.flatten().tolist()
        return camera_info_msg


class VLMNavNode(Node):
    def __init__(self):
        super().__init__('vlm_nav_node')

        self.labels = None

        self.declare_parameter("vlm_nav_config_path", "config/prompts.yaml")
        self.declare_parameter("vlm_nav_config_key", "nature_hiking")
        self.declare_parameter("seg_model", "SAN")
        self.declare_parameter("vlm_model", "gpt-6.1-sol")
        self.declare_parameter("half_precision", True)
        self.declare_parameter("max_threads", 5)
        self.declare_parameter("segment_duration", 0.5)
        self.declare_parameter("vlm_sensor_names", ["zed_camera_front"])
        self.declare_parameter("vlm_duration", 10.0)
        self.declare_parameter("label_mapping", None)  # should be dict but cannot be defined as default

        self.sensor_configs = self._load_sensor_configs()

        if os.path.isabs(self.get_parameter("vlm_nav_config_path").get_parameter_value().string_value):
            vlm_nav_config_path = self.get_parameter("vlm_nav_config_path").get_parameter_value().string_value
        else:
            vlm_nav_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../vlm_nav/",self.get_parameter("vlm_nav_config_path").get_parameter_value().string_value)
        vlm_nav_config_key = self.get_parameter("vlm_nav_config_key").get_parameter_value().string_value
        seg_model = self.get_parameter("seg_model").get_parameter_value().string_value
        vlm_model = self.get_parameter("vlm_model").get_parameter_value().string_value
        half_precision = self.get_parameter("half_precision").get_parameter_value().bool_value

        print(f"Using VLM nav config path: {vlm_nav_config_path}")
        print(f"Using VLM: {vlm_model}")
        print(f"Received parameters: {self.get_parameters(['vlm_nav_config_path', 'vlm_nav_config_key', 'seg_model', 'vlm_model', 'half_precision', 'max_threads', 'segment_duration', 'vlm_sensor_names', 'vlm_duration'])}")

        self.max_threads = self.get_parameter("max_threads").value
        self.segment_duration = self.get_parameter("segment_duration").value
        self.vlm_sensor_names = self.get_parameter("vlm_sensor_names").value
        self.vlm_duration = self.get_parameter("vlm_duration").value
        self.label_mapping = self.get_parameter("label_mapping").value
        if self.label_mapping is None:
            self.label_mapping = {}

        self.vlm_nav_agent = VLMNavAgent(
            vlm_nav_config_path,
            vlm_nav_config_key,
            segmentation_model=seg_model,
            vlm_model=vlm_model,
            half_precision=half_precision,
        )

        current_time = self.get_clock().now()
        self.last_image_time = rclpy.time.Time(seconds=self._to_sec(current_time) - 1000.0)
        self.last_thread_time = rclpy.time.Time(seconds=self._to_sec(current_time) - 1000.0)
        self.last_segment_time = rclpy.time.Time(seconds=self._to_sec(current_time) - 1000.0)

        self.lock = threading.Lock()
        self.seg_lock = threading.Lock()
        self.running_threads = 0

        self.enable_vlm = True
        self.register_sub_pub()
        self.cv_bridge = CvBridge()

        for sensor_name in self.sensor_processors.keys():
            self.create_timer(
                self.segment_duration,
                lambda sensor_name=sensor_name: self.run_segmentation(sensor_name)
            )

        self.vlm_timer = self.create_timer(0.1, self.ask_vlm)
        self.setup_vlm_service()

    def _to_sec(self, t) -> float:
        """Convert rclpy.time.Time or builtin_interfaces.msg.Time to float seconds."""
        if hasattr(t, "nanoseconds"):  # rclpy.time.Time
            return t.nanoseconds / 1e9
        elif hasattr(t, "sec") and hasattr(t, "nanosec"):  # builtin_interfaces.msg.Time
            return float(t.sec) + float(t.nanosec) / 1e9
        else:
            raise TypeError(f"Unsupported time object type: {type(t)}")

    def _load_sensor_configs(self):
        vlm_sensor_names = self.get_parameter_or("vlm_sensor_names", Parameter("vlm_sensor_names", Parameter.Type.STRING_ARRAY, ["hdr_camera_front"])).value
        for sensor_name in vlm_sensor_names:
            self.declare_parameter(f"sensor_configs.{sensor_name}.image_topic", Parameter.Type.STRING)
            self.declare_parameter(f"sensor_configs.{sensor_name}.camera_info_topic", Parameter.Type.STRING)
            self.declare_parameter(f"sensor_configs.{sensor_name}.output_topic_prefix", Parameter.Type.STRING)
            self.declare_parameter(f"sensor_configs.{sensor_name}.max_size", [640, 480])
            self.declare_parameter(f"sensor_configs.{sensor_name}.rotate", 0)

        sensor_config_params = self.get_parameters_by_prefix("sensor_configs")
        sensor_configs = {}
        for param_name, param in sensor_config_params.items():
            path_parts = param_name.split('.')
            sensor_name = path_parts[0]
            config_key = path_parts[1]

            if sensor_name not in sensor_configs:
                sensor_configs[sensor_name] = {}
            sensor_configs[sensor_name][config_key] = param.value

            assert sensor_configs[sensor_name][config_key] is not None, f"Sensor config {sensor_name} {config_key} is None"

        return sensor_configs

    def start_stop_vlm_service(self, request, response):
        if request.data:
            self.enable_vlm = True
            response.success = True
            response.message = "VLM asking started"
        else:
            self.enable_vlm = False
            response.success = True
            response.message = "VLM asking stopped"
        return response

    def setup_vlm_service(self):
        self.vlm_service = self.create_service(SetBool, "~/start_stop_vlm", self.start_stop_vlm_service)

    def register_sub_pub(self):
        self.sensor_processors = {}
        for sensor_name, config in self.sensor_configs.items():
            if "max_size" not in config:
                config["max_size"] = [640, 480]
            if "rotate" not in config:
                config["rotate"] = 0
            sensor_processor = SensorProcessor(
                node=self,
                image_topic=config["image_topic"],
                camera_info_topic=config["camera_info_topic"],
                max_size=config["max_size"],
                rotate=config["rotate"],
                output_topic_prefix=config["output_topic_prefix"],
            )
            self.sensor_processors[sensor_name] = sensor_processor

        self.response_pub = self.create_publisher(String, "~/response", 1)
        self.label_pub = self.create_publisher(String, "~/labels", 1)
        self.duration_pub = self.create_publisher(Float64, "~/duration", 1)

    def run_segmentation(self, sensor_name):
        image, header = self.sensor_processors[sensor_name].get_image_and_header()
        with self.lock:
            labels = self.labels

        current_time = self.get_clock().now()
        current_sec = self._to_sec(current_time)
        last_sec = self._to_sec(self.last_segment_time)
        if (current_sec - last_sec) > 10.0:
            if labels is None and image is None:
                self.get_logger().warn(f"Both image and labels are missing for {sensor_name}", throttle_duration_sec=5)
                return
            elif labels is None:
                self.get_logger().warn(f"Segmentation labels are missing for {sensor_name}", throttle_duration_sec=5)
                return
            elif image is None:
                self.get_logger().warn(f"Image is missing for {sensor_name}", throttle_duration_sec=5)
                return

        if labels is None or image is None:
            return

        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with self.seg_lock:
            seg_return = self.vlm_nav_agent.segment(copy.deepcopy(np.array(image)), labels)
        if seg_return is None:
            return
        seg_image, labels = seg_return

        self.last_segment_time = header.stamp

        if isinstance(seg_image, torch.Tensor):
            seg_image = seg_image.cpu().numpy()
        sem_viz_np = colorize_labels(seg_image)

        sem_viz_np = np.concatenate([bgr_image, sem_viz_np], 1)
        self.sensor_processors[sensor_name].publish_images(seg_image, sem_viz_np, labels, header)

    def get_image_for_vlm(self):
        images = []
        headers = []
        for sensor_name in self.vlm_sensor_names:
            if sensor_name not in self.sensor_processors:
                self.get_logger().warn(f"Sensor {sensor_name} is not registered.")
                continue
            image, header = self.sensor_processors[sensor_name].get_image_and_header()
            if image is None:
                self.get_logger().warn(f"Image from sensor {sensor_name} is missing.", throttle_duration_sec=5)
                continue
            images.append(image)
            headers.append(header)
        return images, headers

    def ask_vlm(self):
        current_time = self.get_clock().now()
        current_sec = self._to_sec(current_time)
        last_sec = self._to_sec(self.last_thread_time)
        if self.running_threads < self.max_threads and (current_sec - last_sec) > self.vlm_duration:
            self.last_thread_time = current_time
            threading.Thread(target=self._ask_vlm_thread).start()
        elif self.running_threads >= self.max_threads:
            self.get_logger().warn(f"Maximum number of threads ({self.max_threads}) already running.", throttle_duration_sec=5)

    def _ask_vlm_thread(self):
        if not self.enable_vlm:
            self.get_logger().warn("VLM asking is disabled.", throttle_duration_sec=5)
            return
        current_time = self.get_clock().now()
        current_sec = self._to_sec(current_time)
        last_sec = self._to_sec(self.last_image_time)
        if (current_sec - last_sec) > self.vlm_duration:
            start_time = self.get_clock().now()
            images, headers = self.get_image_for_vlm()
            if images is not None and len(images) > 0:
                if self.running_threads == 0:
                    self.get_logger().info("Asking VLM to analyze the image...")
                with self.lock:
                    self.running_threads += 1
                labels, response = self.vlm_nav_agent.analyze(images, return_response=True)
                with self.lock:
                    self.running_threads -= 1
                if labels is None:
                    self.get_logger().warn("No response from VLM")
                    return
                ask_time = headers[0].stamp

                # expand labels if they are present in the label mapping
                for expanded_label, additional_labels in self.label_mapping.items():
                    if expanded_label in labels:
                        if isinstance(additional_labels, str):
                            additional_labels = [additional_labels]
                        elif isinstance(additional_labels, list):
                            for additional_label in additional_labels:
                                labels[additional_label] = labels[expanded_label]
                        else:
                            raise ValueError(f"Invalid type for additional labels: {type(additional_labels)}")

                with self.lock:
                    self.labels = labels
                    self.response = response

                end_time = self.get_clock().now()
                end_sec = self._to_sec(end_time)
                start_sec = self._to_sec(start_time)
                duration = end_sec - start_sec
                self.get_logger().info(
                    f"Got VLM response in {duration:.6f} sec. There's {self.running_threads} running threads"
                )
                self.last_image_time = ask_time
                self.publish_response(duration)

    def publish_response(self, duration: float):
        with self.lock:
            response = self.response
            labels = self.labels
        if response is not None and labels is not None:
            response_str, label_table_str = format_vlm_response_string(response, labels)
            response_msg = String()
            response_msg.data = response_str
            label_msg = String()
            label_msg.data = label_table_str

            self.response_pub.publish(response_msg)
            self.label_pub.publish(label_msg)
            self.duration_pub.publish(Float64(data=duration))


def main(args=None):
    rclpy.init(args=args)
    node = VLMNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
