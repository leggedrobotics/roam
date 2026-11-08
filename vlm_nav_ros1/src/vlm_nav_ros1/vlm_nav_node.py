#!/usr/bin/env python3
"""ROS 1 version of vlm_nav_ros/vlm_nav_node.py: same parameters, topics and behaviour."""

import copy
import threading

import cv2  # import before cv_bridge
import numpy as np
from PIL import Image as PILImage
import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float64, String
from std_srvs.srv import SetBool, SetBoolResponse

from vlm_nav import VLMNavAgent
from vlm_nav_ros1.utils import (
    colorize_labels,
    format_vlm_response_string,
    resize_image,
    resolve_config_path,
    rotate_image,
)


class SensorProcessor:
    def __init__(self, image_topic, camera_info_topic, max_size, rotate, output_topic_prefix):
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.max_size = tuple(max_size)
        self.rotate = rotate
        self.output_topic_prefix = output_topic_prefix

        self.image_lock = threading.Lock()
        self.camera_info_lock = threading.Lock()
        self.pub_lock = threading.Lock()

        self.camera_info = None
        self.image = None
        self.header = None
        self.resize_ratio = None
        self.is_compressed = "compressed" in image_topic

        self.bridge = CvBridge()
        self.init_subs()
        self.init_pubs()

    def init_subs(self):
        image_type = CompressedImage if self.is_compressed else Image
        self.image_sub = rospy.Subscriber(self.image_topic, image_type, self.image_callback, queue_size=1)
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )

    def init_pubs(self):
        self.sem_image_pub = rospy.Publisher(self.output_topic_prefix + "_sem", Image, queue_size=1)
        self.sem_viz_pub = rospy.Publisher(self.output_topic_prefix + "_sem_viz", Image, queue_size=1)
        self.sem_camera_info_pub = rospy.Publisher(self.camera_info_topic + "_sem", CameraInfo, queue_size=1)

    def get_image_and_header(self):
        with self.image_lock:
            return copy.deepcopy(self.image), copy.deepcopy(self.header)

    def publish_images(self, sem_seg, sem_viz, header):
        with self.pub_lock:
            self.publish_image(self.sem_image_pub, sem_seg, header, encoding="passthrough", rotate_back=True)
            self.publish_image(self.sem_viz_pub, sem_viz, header, encoding="bgr8")
            with self.camera_info_lock:
                camera_info = copy.deepcopy(self.camera_info)
            if camera_info is not None:
                # Same stamp as the semantic image, so elevation_mapping_cupy's time synchronizer pairs them.
                camera_info.header = header
                self.sem_camera_info_pub.publish(camera_info)

    def publish_image(self, pub, image, header, encoding="passthrough", rotate_back=False):
        if rotate_back and self.rotate:
            image = rotate_image(image, -self.rotate)
        if encoding == "passthrough":
            image = np.ascontiguousarray(np.transpose(image, (1, 2, 0)).astype(np.float32))
        if image.shape[2] > 10:
            rospy.logwarn(
                "Image in shape of [%d x %d x %d] has too many channels to publish. Will cause great delay."
                % image.shape
            )
        image_msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        image_msg.header = header
        pub.publish(image_msg)

    def image_callback(self, data):
        try:
            if self.is_compressed:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "rgb8")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            image = self.process_image(PILImage.fromarray(cv_image))
            with self.image_lock:
                self.image = image
                self.header = data.header
        except Exception as e:
            rospy.logerr("Could not convert image message to cv image: %s", e)

    def camera_info_callback(self, data):
        try:
            camera_info_msg = self.process_camera_info(data)
            with self.camera_info_lock:
                self.camera_info = camera_info_msg
        except Exception as e:
            rospy.logerr("Could not process camera info: %s", e)

    def process_image(self, image):
        if self.rotate:
            image = rotate_image(image, self.rotate)
        image, self.resize_ratio = resize_image(image, self.max_size)
        # Size of the published semantic image, which is rotated back to the camera's orientation.
        width, height = image.size
        self.output_size = (height, width) if self.rotate % 180 == 90 else (width, height)
        return image

    def process_camera_info(self, camera_info_msg):
        if self.resize_ratio is None:
            return
        return self.resize_camera_info_matrices(camera_info_msg, self.resize_ratio)

    def resize_camera_info_matrices(self, camera_info_msg, resize_ratio):
        # Take the size from the resized image: int(ratio * size) can be one pixel off PIL's rounding.
        camera_info_msg.width, camera_info_msg.height = self.output_size
        K = np.array(camera_info_msg.K).reshape((3, 3))
        P = np.array(camera_info_msg.P).reshape((3, 4))
        K[:2, :3] *= resize_ratio
        P[:2, :3] *= resize_ratio
        camera_info_msg.K = K.flatten().tolist()
        camera_info_msg.P = P.flatten().tolist()
        return camera_info_msg


class VLMNavNode:
    def __init__(self):
        self.labels = None
        self.response = None

        vlm_nav_config_path = resolve_config_path(rospy.get_param("~vlm_nav_config_path", "config/prompts.yaml"))
        vlm_nav_config_key = rospy.get_param("~vlm_nav_config_key", "nature_hiking")
        seg_model = rospy.get_param("~seg_model", "SAN")
        vlm_model = rospy.get_param("~vlm_model", "gpt-6.1-sol")
        half_precision = rospy.get_param("~half_precision", True)
        rospy.loginfo("Using VLM nav config path: %s", vlm_nav_config_path)
        rospy.loginfo("Using VLM: %s", vlm_model)

        self.max_threads = rospy.get_param("~max_threads", 5)
        self.segment_duration = rospy.get_param("~segment_duration", 0.5)
        self.vlm_sensor_names = rospy.get_param("~vlm_sensor_names")
        self.vlm_duration = rospy.get_param("~vlm_duration", 10.0)
        self.label_mapping = rospy.get_param("~label_mapping", {})

        self.vlm_nav_agent = VLMNavAgent(
            vlm_nav_config_path,
            vlm_nav_config_key,
            segmentation_model=seg_model,
            vlm_model=vlm_model,
            half_precision=half_precision,
        )

        self.last_image_time = rospy.Time(0)
        self.last_thread_time = rospy.Time(0)
        self.last_segment_time = rospy.Time(0)
        self.lock = threading.Lock()
        self.seg_lock = threading.Lock()
        self.running_threads = 0

        self.enable_vlm = True
        self.register_sub_pub()
        self.segment_timers = [
            rospy.Timer(
                rospy.Duration(self.segment_duration),
                lambda event, name=sensor_name: self.run_segmentation(name),
            )
            for sensor_name in self.sensor_processors
        ]
        self.vlm_timer = rospy.Timer(rospy.Duration(0.1), self.ask_vlm)
        self.vlm_service = rospy.Service("~start_stop_vlm", SetBool, self.start_stop_vlm_service)

    def start_stop_vlm_service(self, request):
        self.enable_vlm = request.data
        return SetBoolResponse(success=True, message="VLM asking started" if request.data else "VLM asking stopped")

    def register_sub_pub(self):
        self.sensor_processors = {}
        for sensor_name, config in rospy.get_param("~sensor_configs").items():
            self.sensor_processors[sensor_name] = SensorProcessor(
                image_topic=config["image_topic"],
                camera_info_topic=config["camera_info_topic"],
                max_size=config.get("max_size", [640, 480]),
                rotate=config.get("rotate", 0),
                output_topic_prefix=config["output_topic_prefix"],
            )

        self.response_pub = rospy.Publisher("~response", String, queue_size=1)
        self.label_pub = rospy.Publisher("~labels", String, queue_size=1)
        self.duration_pub = rospy.Publisher("~duration", Float64, queue_size=1)

    def run_segmentation(self, sensor_name):
        image, header = self.sensor_processors[sensor_name].get_image_and_header()
        with self.lock:
            labels = self.labels

        if (rospy.Time.now() - self.last_segment_time).to_sec() > 10.0:
            if labels is None and image is None:
                rospy.logwarn_throttle(5, "Both image and labels are missing for %s" % sensor_name)
            elif labels is None:
                rospy.logwarn_throttle(5, "Segmentation labels are missing for %s" % sensor_name)
            elif image is None:
                rospy.logwarn_throttle(5, "Image is missing for %s" % sensor_name)
        if labels is None or image is None:
            return

        bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with self.seg_lock:
            seg_return = self.vlm_nav_agent.segment(copy.deepcopy(np.array(image)), labels)
        if seg_return is None:
            return
        seg_image, _ = seg_return
        self.last_segment_time = rospy.Time.now()

        if isinstance(seg_image, torch.Tensor):
            seg_image = seg_image.cpu().numpy()
        sem_viz_np = np.concatenate([bgr_image, colorize_labels(seg_image)], 1)
        self.sensor_processors[sensor_name].publish_images(seg_image, sem_viz_np, header)

    def get_image_for_vlm(self):
        images, headers = [], []
        for sensor_name in self.vlm_sensor_names:
            if sensor_name not in self.sensor_processors:
                rospy.logwarn("Sensor %s is not registered.", sensor_name)
                continue
            image, header = self.sensor_processors[sensor_name].get_image_and_header()
            if image is None:
                rospy.logwarn_throttle(5, "Image from sensor %s is missing." % sensor_name)
                continue
            images.append(image)
            headers.append(header)
        return images, headers

    def ask_vlm(self, event):
        now = rospy.Time.now()
        if self.running_threads >= self.max_threads:
            rospy.logwarn_throttle(5, "Maximum number of threads (%d) already running." % self.max_threads)
        elif (now - self.last_thread_time).to_sec() > self.vlm_duration:
            self.last_thread_time = now
            threading.Thread(target=self._ask_vlm_thread, daemon=True).start()

    def _ask_vlm_thread(self):
        if not self.enable_vlm:
            rospy.logwarn_throttle(5, "VLM asking is disabled.")
            return
        if (rospy.Time.now() - self.last_image_time).to_sec() <= self.vlm_duration:
            return
        start_time = rospy.Time.now()
        images, headers = self.get_image_for_vlm()
        if not images:
            return
        if self.running_threads == 0:
            rospy.loginfo("Asking VLM to analyze the image...")
        with self.lock:
            self.running_threads += 1
        try:
            labels, response = self.vlm_nav_agent.analyze(images, return_response=True)
        finally:
            with self.lock:
                self.running_threads -= 1
        if labels is None:
            rospy.logwarn("No response from VLM")
            return

        # Copy the values of a label to the labels it is mapped to.
        for expanded_label, additional_labels in self.label_mapping.items():
            if expanded_label in labels:
                if isinstance(additional_labels, str):
                    additional_labels = [additional_labels]
                for additional_label in additional_labels:
                    labels[additional_label] = labels[expanded_label]

        with self.lock:
            self.labels = labels
            self.response = response
        duration = (rospy.Time.now() - start_time).to_sec()
        rospy.loginfo("Got VLM response in %.6f sec. There's %d running threads", duration, self.running_threads)
        self.last_image_time = headers[0].stamp
        self.publish_response(duration)

    def publish_response(self, duration):
        with self.lock:
            response, labels = self.response, self.labels
        if response is None or labels is None:
            return
        response_str, label_table_str = format_vlm_response_string(response, labels)
        self.response_pub.publish(String(data=response_str))
        self.label_pub.publish(String(data=label_table_str))
        self.duration_pub.publish(Float64(data=duration))


def main():
    rospy.init_node("vlm_nav_node")
    VLMNavNode()
    rospy.spin()


if __name__ == "__main__":
    main()
