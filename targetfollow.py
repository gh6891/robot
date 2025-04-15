# from isaacsim import SimulationApp
# config = {'width': 1920, 'height': 1080, 'headless': False}
# kit = SimulationApp(config)
# print(kit.DEFAULT_LAUNCHER_CONFIG)

# from isaacsim.core.api import World
# from isaacsim.core.api.objects import VisualCuboid
from isaacsim import SimulationApp
config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid


import sys, os
import random
import numpy as np

external_lib_path = "/home/gh6891/robot/utils"
sys.path.append(external_lib_path)
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.api.robots import Robot
from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController

my_world = World(stage_units_in_meters=1.0)

# Initialize the Scene
scene = my_world.scene
scene.add_default_ground_plane()

# Add cube
cube = VisualCuboid(
prim_path = "/World/cube",
name = "cube",
position = [0.5, 0.5, 0.1],
color = np.array([0, 0, 1]),
size = 0.04,
)
scene.add(cube)

# Add robot
robot_usd_path = os.path.join(external_lib_path, "utils/assets/ur5e_handeye_gripper.usd")
my_robot = UR5eHandeye(
    prim_path="/World/ur5e", # should be unique
    name="my_ur5e",
    # should be unique, used to access the object
    usd_path=robot_usd_path,
    activate_camera=False,
    )
scene.add(my_robot)

# stage = omni.usd.get_context().get_stage()
# Define the prim path
wrist_path = "/World/ur5e/robotiq_arg2f_base_link"
left_finger_path = "/World/ur5e/left_inner_finger_pad"
right_finger_path = "/World/ur5e/right_inner_finger_pad"

# # Get the prim using the path
# wrist = get_prim_at_path(wrist_path)
# left_finger = get_prim_at_path(left_finger_path)
# right_finger = get_prim_at_path(right_finger_path)

from pxr import UsdGeom
# Get the Xformable API to access transformation attributes

# wrist = UsdGeom.Xformable(wrist)
# left_finger = UsdGeom.Xformable(left_finger)
# right_finger = UsdGeom.Xformable(right_finger)

my_controller = RMPFlowController(
name="end_effector_controller_cspace_controller", robot_articulation=my_robot, attach_gripper=True
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_robot.get_articulation_controller()
# Simulation Loop
my_world.reset()
my_controller.reset()

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step
    my_world.step(render=True)
    ee_target_position, ee_target_orientation = cube.get_world_pose()
    wrist_position = get_world_pose(wrist_path)
    left_finger_position = get_world_pose(left_finger_path)
    right_finger_position = get_world_pose(right_finger_path)
    offset = wrist_position[0] - (left_finger_position[0]+right_finger_position[0])/2
    ee_target_position += offset
    # For Gipper offset
    if my_world.is_playing():
        actions = my_controller.forward(
            target_end_effector_position=ee_target_position,
            )
    # 컨트롤러 내부에서 계산된 타겟 joint position값을
    # articulation controller에 전달하여 action 수행
    articulation_controller.apply_action(actions)
# 시뮬레이션 종료
simulation_app.close()