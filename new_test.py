import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from scipy.spatial.transform import Rotation as R
# import omni
from isaacsim import SimulationApp
import numpy as np
import torch
import cv2
#test import
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # GUI 없이 작동하는 non-interactive 백엔드 사용
import matplotlib.pyplot as plt
config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
from abc import abstractmethod
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.prims import RigidPrim as RigidPrimView
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.api import World
from isaacsim.core.api.objects.sphere import DynamicSphere
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.cloner.grid_cloner import GridCloner
from isaacsim.storage.native import find_nucleus_server
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, euler_to_rot_matrix
from isaacsim.core.utils.rotations import quat_to_euler_angles, rot_matrix_to_quat
from scipy.spatial.transform import Rotation as R
from pxr import UsdPhysics, UsdLux, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema
path = "/home/gh6891/robot/pushing"
sys.path.append(path)
from terrain_utils import *
from terraincreation import TerrainCreation
#로봇path
absolute_path = "/home/gh6891/robot/utils/utils/assets"
sys.path.append(absolute_path)
relative_path = "/home/gh6891/robot/utils"
absolute_path = os.path.abspath(relative_path)
sys.path.append(absolute_path)
from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController
# ─────────────────────────────────────────────────────── #
# ─────────────────────────────────────────────────────── #
if __name__ == "__main__":
    depth_images = []
    camera_Transforms = []
    world= World(stage_units_in_meters=1.0)
    scene = world.scene
    scene.add_default_ground_plane()
    # scene.add_default_ground_plane()
    #setup robot
    robot_usd_path = os.path.join(absolute_path, "utils/assets/ur5e_handeye_gripper.usd")
    my_robot = UR5eHandeye(
        prim_path="/World/ur5e", # should be unique
        name="my_ur5e",
        usd_path=robot_usd_path,
        activate_camera=False,
        )
    scene.add(my_robot)
    # Add Controller
    my_controller = BasicManipulationController(
        name='basic_manipulation_controller',
        cspace_controller=RMPFlowController(
            name="end_effector_controller_cspace_controller", robot_articulation=my_robot, attach_gripper=True
            ),
        gripper=my_robot.gripper,
        # phase의 진행 속도 설정
        events_dt=[0.008],
    )
    # robot control(PD control)을 위한 instance 선언
    articulation_controller = my_robot.get_articulation_controller()
    my_controller.reset()
    # Reset the world = Scene Setup
    world.reset()
    first_target_position = np.array([0., 0.3, 0.5])
    # first_target_position = np.array([-0.4, 0.3, 0.4])
    second_target_position = np.array([0.0, 0.3, 0.3])
    third_target_position = np.array([0.4, 0.3, 0.4])
    offset_orientation = np.array([90.0, 90.0, 0.0]) # euler angles
    offset_orientation_matrix = R.from_euler('xyz', offset_orientation, degrees=True)
    print(f"==>> offset_orientation_matrix: {offset_orientation_matrix}")
    target_orientation_euler = np.array([0, 0.0, 0.0]) # euler angles
    target_orientation_maxtix = R.from_euler('xyz', target_orientation_euler, degrees = True)
    print(f"==>> target_orientation_maxtix: {target_orientation_maxtix}")
    target_orientation_matrix = offset_orientation_matrix @ target_orientation_maxtix
    print(f"==>> target_orientation_matrix: {target_orientation_matrix}")
    if False:
        target_orientation = rot_matrix_to_quat(target_orientation_matrix)
    else:
        target_orientation = euler_angles_to_quat(target_orientation_euler, degrees = True)
    print("목표 target_orientation_quat : ", target_orientation)
    print("다시 오일러 target_orientation_euler : ", quat_to_euler_angles(target_orientation, degrees = True))
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.forward(
                target_position=first_target_position,
                end_effector_offset = np.array([0, 0, 0.0]),
                end_effector_orientation=target_orientation,
                current_joint_positions=my_robot.get_joints_state().positions
                )
            articulation_controller.apply_action(actions)
                    # controller의 동작이 끝남 여부를 확인w
            if my_controller.is_done():
                for _ in range(100):
                    world.step(render=True)
                # print("done position control of end-effector")
                # print("동작이 끝나고 end_effector_orientation",my_robot._end_effector.get_world_poses()[1][0])
                # print("동작이 끝나고 end_effector_orientation_True",quat_to_euler_angles(my_robot._end_effector.get_world_poses()[1][0], degrees = True))
                # print("동작이 끝나고 end_effector_orientation_False",quat_to_euler_angles(my_robot._end_effector.get_world_poses()[1][0], degrees = True, extrinsic=False))
                my_controller.reset()
    simulation_app.close()