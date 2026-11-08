# ROAM + elevation_mapping_cupy

ROAM's segmentation gives risk, steppable and preferred-area values per image pixel. On the robot,
[elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) projects those images
onto its 2.5D elevation map, so the planner sees them as map layers next to the geometry.

This folder replays that offline on a 30 s clip from the Seealpsee hike (25 June 2024):

```bash
cd examples/elevation_mapping
python3 semantic_elevation_map.py                    # replay ROAM's on-board semantic images (no API key)
python3 semantic_elevation_map.py --source segment   # re-run SAN on the camera frames with the recorded VLM labels
python3 semantic_elevation_map.py --source vlm       # ask the VLM again, then segment
```

The script feeds the lidar point clouds and the semantic images to elevation_mapping_cupy's GPU core in
time order (no ROS needed), then shows the fused elevation, risk, steppable and preferred-area layers
top-down. Click a map cell to plan a path from the robot with A* over the fused risk layer. Cells a camera has
seen outside the preferred area are blocked. Unseen cells count as unknown (risk 0.5) if the lidar measured
their elevation, and are blocked otherwise, except along the robot's own trajectory (the lidar's blind spot). Without a display,
use `--goal x,y --save map.png` (metres, relative to where the clip starts).

## Data

[`data/seealpsee_trail`](data/seealpsee_trail), cut from the robot's rosbag:

| File | Content |
|---|---|
| `clip.npz` | Lidar clouds (lidar frame, cm) with `odom → lidar` and `odom → base` poses; ROAM's semantic images as computed on the robot (risk, steppable, preferred_area; uint8, half resolution) with `odom → camera` poses; camera intrinsics (for the half-resolution images) and distortion. Positions are relative to the robot at the start of the clip. |
| `images/` | The camera frame matching each semantic image |
| `labels.json` | The VLM labels active during the clip, with the raw response |

## Installing elevation_mapping_cupy

You need a CUDA GPU, [CuPy](https://docs.cupy.dev/en/stable/install.html) and a CUDA build of PyTorch.
The example only imports elevation_mapping_cupy's Python core, so a source checkout on `PYTHONPATH` is
enough. Two versions work and give the same map:

- **`main` (ROS 1):** has semantic image layers. Its CUDA kernels use the `float16` type that CuPy 14
  dropped, so use CuPy 13 or older (on Noetic's Python 3.8: CuPy 12, NumPy < 1.24). With NumPy 2 it also
  needs [leggedrobotics/elevation_mapping_cupy#149](https://github.com/leggedrobotics/elevation_mapping_cupy/pull/149)
  (`cp.bool8`); the example works around that one itself.
- **ROS 2:** the `ros2` branch can't fuse images yet (`input_image()` fails because its image buffers are
  never allocated). `pascal-roth/b2w-wild-navigation-semantic-layers` fixes that. Works with CuPy 13 and 14.

```bash
pip3 install "cupy-cuda12x<14" simple_parsing shapely scipy ruamel.yaml
# main (ROS 1)
git clone https://github.com/leggedrobotics/elevation_mapping_cupy.git
export PYTHONPATH=$PWD/elevation_mapping_cupy/elevation_mapping_cupy/script:$PYTHONPATH
# or ROS 2 with semantic layers
git clone -b pascal-roth/b2w-wild-navigation-semantic-layers https://github.com/leggedrobotics/elevation_mapping_cupy.git
export PYTHONPATH=$PWD/elevation_mapping_cupy/elevation_mapping_cupy:$PYTHONPATH
```

## On the robot

Both ROS packages have a launch file that starts the ROAM node and elevation_mapping_cupy together, with a
config that subscribes to the node's `<prefix>_sem` image and `<camera_info>_sem` topics. Set the frames
and the point cloud topic for your robot.

| | Launch | elevation_mapping_cupy config |
|---|---|---|
| ROS 1 (`main` branch) | `roslaunch vlm_nav_ros1 vlm_nav_elevation_mapping.launch` | [`vlm_nav_ros1/config/elevation_mapping_vlm_nav.yaml`](../../vlm_nav_ros1/config/elevation_mapping_vlm_nav.yaml) |
| ROS 2 | `ros2 launch vlm_nav_ros vlm_nav_elevation_mapping.launch.py` | [`vlm_nav_ros/config/elevation_mapping_vlm_nav.yaml`](../../vlm_nav_ros/config/elevation_mapping_vlm_nav.yaml) |

Replaying this clip through either launch file gives the same semantic layers: on the cells both maps cover,
preferred_area differs by 0.04 on average. The ROS 1 map uses `main`'s defaults (8 m at 4 cm), the ROS 2 map
20 m at 10 cm.

**Known issue (Nov 2026):** elevation_mapping_cupy's ROS 2 node passes the camera-to-map transform to
`ElevationMap.input_image()`, which expects map-to-camera (the ROS 1 node gets this right). With that bug, the
replay puts ROAM's semantics on 347 map cells and 5% of the robot's path; with the fix,
[leggedrobotics/elevation_mapping_cupy#148](https://github.com/leggedrobotics/elevation_mapping_cupy/pull/148),
on 18,187 cells and 77% of the path. Until it is merged, apply it on top of the semantic-layers branch:

```bash
git fetch https://github.com/mktk1117/elevation_mapping_cupy.git fix/image-callback-tf-direction
git cherry-pick FETCH_HEAD
```
