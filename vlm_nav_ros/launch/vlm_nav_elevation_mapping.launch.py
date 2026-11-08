"""vlm_nav_node plus elevation_mapping_cupy fusing its risk, steppable and preferred-area images."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    vlm_nav_share = get_package_share_directory("vlm_nav_ros")
    elevation_share = get_package_share_directory("elevation_mapping_cupy")

    vlm_nav_config = LaunchConfiguration("vlm_nav_config")
    elevation_config = LaunchConfiguration("elevation_mapping_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument("vlm_nav_config", default_value=os.path.join(vlm_nav_share, "config", "vlm_nav_cfg.yaml")),
        DeclareLaunchArgument(
            "elevation_mapping_config",
            default_value=os.path.join(vlm_nav_share, "config", "elevation_mapping_vlm_nav.yaml"),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="vlm_nav_ros",
            executable="vlm_nav_node",
            output="screen",
            parameters=[vlm_nav_config, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="elevation_mapping_cupy",
            executable="elevation_mapping_node.py",
            name="elevation_mapping_node",
            output="screen",
            parameters=[
                os.path.join(elevation_share, "config", "core", "core_param.yaml"),
                elevation_config,
                {"use_sim_time": use_sim_time},
            ],
        ),
    ])
