import os
from glob import glob

from setuptools import find_packages, setup

package_name = "vlm_nav_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),  # assumes your code is in vlm_nav_ros/
    include_package_data=True,
    data_files=[
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "resource"), glob("resource/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Manthan Patel",
    maintainer_email="manthanp@leggedrobotics.com",
    description="ROS2 wrapper nodes for vlm_nav",
    license="MIT",
    entry_points={
        "console_scripts": [
            "vlm_nav_node = vlm_nav_ros.vlm_nav_node:main",
        ],
    },
)
