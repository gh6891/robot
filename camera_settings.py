# from omni.isaac.kit import SimulationApp
from isaacsim import SimulationApp

config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.semantics import add_update_semantics


import numpy as np
import os
import cv2


my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

seg_labels = [0, 1, 2]
seg_colors = [np.array([0, 0, 0]), np.array([255, 0, 0]), np.array([0, 255, 0])]

def save_image(rgb, depth, seg, file_name):
    seg_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)

    min_depth, max_depth = depth.min(), depth.max()
    depth = (depth - min_depth) / (max_depth - min_depth) * 255
    depth = depth.astype(np.uint8)

    num_classes = 3
    #class_num은 1 과 2만 조사
    for class_num in range(1, num_classes):
        seg_rgb[seg == class_num] = seg_colors[class_num]
    # image 저장
    cv2.imwrite(file_name + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name + "_depth.png", cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(file_name + "_seg.png", cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR))




cube = my_world.scene.add(
    DynamicCuboid(
        prim_path = "/World/cube",
        name = "cube1",
        position = [0, 0, 1],
        scale = [0.6, 0.5, 0.2],
        size = 1.0,
        color = np.array([0.0, 1.0, 0.0]),
    )
)
my_camera = Camera(
prim_path="/World/Camera",
frequency=20,
resolution=(1920, 1080),
position=[0.48176, 0.13541, 0.71],
orientation=[0.5,-0.5,0.5,0.5]
)

my_camera.initialize()
my_camera.add_distance_to_camera_to_frame()
my_camera.set_focal_length(1.93)
my_camera.set_focus_distance(4)
my_camera.set_horizontal_aperture(2.65)
my_camera.set_vertical_aperture(1.48)
my_camera.set_clipping_range(0.01, 10000)
my_camera.add_distance_to_camera_to_frame()
my_camera.add_instance_segmentation_to_frame()



my_world.reset()
is_semantic_initialized = False
while not is_semantic_initialized:
    my_world.step(render=True)
    if my_camera.get_current_frame()["instance_segmentation"] is not None:
        is_semantic_initialized = True
my_world.reset()

total_episodes = 10
max_episode_steps = 100
save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.3images")
os.makedirs(save_root, exist_ok=True)

for i in range(10):
    print(f"Episode: {i}")
    add_update_semantics(prim=cube.prim, semantic_label=f"{i}")

for i in range(10):
    print(f"Episode: {i}")
    add_update_semantics(prim=cube.prim, semantic_label=f"{i}")

    for j in range(100):
        print(f"Step: {j}")
        file_name = f"episode_{i}_step_{j}"
        rgb_image = my_camera.get_rgba()
        current_frame = my_camera.get_current_frame()
        print(current_frame["instance_segmentation"].keys())
        distance_image = current_frame["distance_to_camera"]
        instance_segmentation_image = current_frame["instance_segmentation"]["data"]
        instance_segmentation_dict = current_frame["instance_segmentation"]["info"]["idToSemantics"]
        # print(instance_segmentation_dict)
        save_image(rgb_image, distance_image, instance_segmentation_image, os.path.join(save_root,
        file_name))
        my_world.step(render=True)
# SimulationApp Close
simulation_app.close()
print("Simulation is Closed")