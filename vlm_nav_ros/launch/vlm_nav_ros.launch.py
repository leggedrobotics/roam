from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package share directory
    pkg = get_package_share_directory("vlm_nav_ros")
    # Default config file
    params = os.path.join(pkg, "config", "vlm_nav_cfg.yaml")

    return LaunchDescription([
        Node(
            package="vlm_nav_ros",
            executable="vlm_nav_node",
            output="screen",
            parameters=[params],
        ),
    ])
