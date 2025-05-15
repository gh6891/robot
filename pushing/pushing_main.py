import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

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
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles



from pxr import UsdPhysics, UsdLux, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema

from terrain_utils import *
from terraincreation import TerrainCreation

#로봇path 
relative_path = "../utils/utils/assets"
absolute_path = os.path.abspath(relative_path)
sys.path.append(absolute_path)

relative_path = "../utils"
absolute_path = os.path.abspath(relative_path)
sys.path.append(absolute_path)

from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController

# ─────────────────────────────────────────────────────── #
# 이미지 저장 디렉토리 설정
save_dir = "/home/gh6891/robot/pushing/test_image"
os.makedirs(save_dir, exist_ok=True)
# ─────────────────────────────────────────────────────── #
def generate_height_map(depth_images):
    all_points = []
    camera_positions = [first_target_position, second_target_position, third_target_position]
    for i in range(3):
        depth_img = depth_images[i]

        H, W = depth_img.shape # H :720, W :1280
        fx = fy = 700  # 카메라 focal length 가정값
        cx, cy = W / 2, H / 2

        xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
        xmap = xmap.flatten()
        ymap = ymap.flatten()
        depth_flat = depth_img.flatten()

        valid = depth_flat > 0.01
        z = depth_flat[valid]
        x = (xmap[valid] - cx) * z / fx
        y = (ymap[valid] - cy) * z / fy
        points = np.stack([x, y, z], axis=1)

        # 로봇 포즈를 기준으로 변환
        base_position = camera_positions[i]
        transformed = points + base_position  # 월드 좌표로 이동
        all_points.append(transformed)
    
    # 모든 포인트 병합
    merged_points = np.concatenate(all_points, axis=0)

    # height map 생성 (x-y 평면 기준 z 저장)
    resolution = 0.01  # 1cm 간격
    x_vals = merged_points[:, 0]
    y_vals = merged_points[:, 1]
    z_vals = merged_points[:, 2]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    grid_x = np.arange(x_min, x_max, resolution)
    grid_y = np.arange(y_min, y_max, resolution)
    height_map = np.full((len(grid_y), len(grid_x)), fill_value=np.nan)

    for x, y, z in zip(x_vals, y_vals, z_vals):
        xi = int((x - x_min) / resolution)
        yi = int((y - y_min) / resolution)
        if 0 <= xi < len(grid_x) and 0 <= yi < len(grid_y):
            if np.isnan(height_map[yi, xi]) or z < height_map[yi, xi]:
                height_map[yi, xi] = z

    # NaN → 0
    height_map = np.nan_to_num(height_map, nan=0.0)

    # 시각화
    plt.imshow(height_map, cmap='viridis', origin='lower')
    plt.colorbar(label='Height (m)')
    plt.title("Merged Height Map from 3 Camera Views")
    plt.xlabel("X Grid")
    plt.ylabel("Y Grid")
    plt.savefig(os.path.join(save_dir, f"merged_height_map{idx/3-1}.png"))
    plt.close()

    return height_map
def setup_world():

if __name__ == "__main__":
    setup_world()
    depth_images = []
    # world = World(
    #     stage_units_in_meters=1.0, 
    #     rendering_dt=1.0/60.0,
    #     backend="torch", 
    #     device="cpu"
    # )
    world= World(stage_units_in_meters=1.0)
    scene = world.scene
    # scene.add_default_ground_plane()

    #setup robot
    robot_usd_path = os.path.join(absolute_path, "utils/assets/ur5e_handeye_gripper.usd")
    my_robot = UR5eHandeye(
        prim_path="/World/ur5e", # should be unique
        name="my_ur5e",
        # should be unique, used to access the object
        usd_path=robot_usd_path,
        activate_camera=True,
        )
    scene.add(my_robot)

    # Add cube
    cube = DynamicCuboid(
        prim_path = "/World/cube",
        name = "cube",
        position = [0.0, 0.5, 1.0],
        color = np.array([1, 0, 0]),
        size = 0.06,
        mass = 0.01,
        )
    
    scene.add(cube)
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

    #setup terrain
    num_envs = 100  
    num_per_row = 10
    env_spacing = 0.56*2

    terrain_creation_task = TerrainCreation(name="TerrainCreation", 
                                            num_envs=num_envs,
                                            num_per_row=num_per_row,
                                            env_spacing=env_spacing,
                                            )
                            
    world.add_task(terrain_creation_task)

    # robot control(PD control)을 위한 instance 선언
    articulation_controller = my_robot.get_articulation_controller()
    my_controller.reset()
    # Reset the world = Scene Setup
    world.reset()
    my_robot.rgb_cam.initialize()
    my_robot.depth_cam.initialize()
    my_robot.depth_cam.add_distance_to_image_plane_to_frame()
    # init target position을 robot end effector의 위치로 지정
    init_target_position = my_robot._end_effector.get_world_poses()[0][0]

    state = "APPROACH_1"
    first_target_position = np.array([-0.3, 0.3, 0.3])
    second_target_position = np.array([0.0, 0.3, 0.3])
    third_target_position = np.array([0.3, 0.3, 0.3])
    target_orientation_euler = np.array([180.0, -90.0, 0.0]) # euler angles
    target_orientation = euler_angles_to_quat(target_orientation_euler, degrees = True, extrinsic=False)
    print("목표 target_orientation_quat : ", target_orientation)
    print("다시 오일러 target_orientation_euler : ", quat_to_euler_angles(target_orientation, degrees = True))
    idx = 0 #사진 인덱스
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():
            if state == "APPROACH_1":

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
                    print("done position control of end-effector")
                    print("동작이 끝나고 end_effector_orientation",my_robot._end_effector.get_world_poses()[1][0])
                    print("동작이 끝나고 end_effector_orientation",quat_to_euler_angles(my_robot._end_effector.get_world_poses()[1][0], degrees = True))
                    my_controller.reset()
                    # APPROACH가 끝났을 경우 GRASP state 단계로 변경
                    state = "APPROACH_2"

                    # 안정화 후 이미지 캡처
                    for _ in range(10):
                        world.step(render=True)
                    rgb = my_robot.rgb_cam.get_rgba()
                    depth = my_robot.depth_cam.get_depth()

                    if rgb is not None and depth is not None:
                        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        depth_clean = np.where(np.isfinite(depth), depth, 0) # inf, nan > 0으로 변경
                        depth_images.append(depth_clean)
                        depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
                        depth_gray = depth_norm.astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, f"rgb_pos{idx}.png"), rgb_bgr)
                        cv2.imwrite(os.path.join(save_dir, f"depth_pos{idx}.png"), depth_gray)
                        print(f"[INFO] Saved images for position {idx}")
                        idx+=1

            # if state == "APPROACH_2":
            #     # cube 위치 얻어오기
            #     cur_cube_position = cube.get_world_pose()[0]
            #     # 선언한 my_controller를 사용하여 action 수행
            #     actions = my_controller.forward(
            #         target_position=second_target_position,
            #         end_effector_orientation=target_orientation,
            #         current_joint_positions=my_robot.get_joints_state().positions
            #         )
            #     articulation_controller.apply_action(actions)
            #             # controller의 동작이 끝남 여부를 확인
            #     if my_controller.is_done():
            #         print("done position control of end-effector")
            #         my_controller.reset()
            #         # APPROACH가 끝났을 경우 GRASP state 단계로 변경
            #         state = "APPROACH_3"

            #         # 안정화 후 이미지 캡처
            #         for _ in range(10):
            #             world.step(render=True)
            #         rgb = my_robot.rgb_cam.get_rgba()
            #         depth = my_robot.depth_cam.get_depth()

            #         if rgb is not None and depth is not None:
            #             rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            #             depth_clean = np.where(np.isfinite(depth), depth, 0)  # inf, nan → 0
            #             depth_images.append(depth_clean)
            #             depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
            #             depth_gray = depth_norm.astype(np.uint8)
            #             cv2.imwrite(os.path.join(save_dir, f"rgb_pos{idx}.png"), rgb_bgr)
            #             cv2.imwrite(os.path.join(save_dir, f"depth_pos{idx}.png"), depth_gray)
            #             print(f"[INFO] Saved images for position {idx}")
            #             idx+=1

            # if state == "APPROACH_3":
            #     # cube 위치 얻어오기
            #     cur_cube_position = cube.get_world_pose()[0]
            #     # 선언한 my_controller를 사용하여 action 수행
            #     actions = my_controller.forward(
            #         target_position=third_target_position,
            #         end_effector_orientation=target_orientation,
            #         current_joint_positions=my_robot.get_joints_state().positions
            #         )
            #     articulation_controller.apply_action(actions)
            #             # controller의 동작이 끝남 여부를 확인
            #     if my_controller.is_done():
            #         print("done position control of end-effector")
            #         my_controller.reset()
            #         # APPROACH가 끝났을 경우 GRASP state 단계로 변경
            #         state = "APPROACH_1"

            #         # 안정화 후 이미지 캡처
            #         for _ in range(10):
            #             world.step(render=True)
            #         rgb = my_robot.rgb_cam.get_rgba()
            #         depth = my_robot.depth_cam.get_depth()
            #         if rgb is not None and depth is not None:
            #             rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            #             depth_clean = np.where(np.isfinite(depth), depth, 0)
            #             depth_images.append(depth_clean)
            #             depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
            #             depth_gray = depth_norm.astype(np.uint8)
            #             cv2.imwrite(os.path.join(save_dir, f"rgb_pos{idx}.png"), rgb_bgr)
            #             cv2.imwrite(os.path.join(save_dir, f"depth_pos{idx}.png"), depth_gray)
            #             print(f"[INFO] Saved images for position {idx}")
            #             idx+=1
            #             generate_height_map(depth_images=depth_images)

                              

    simulation_app.close()