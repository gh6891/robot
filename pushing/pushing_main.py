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
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix



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
def generate_height_map(depth_images, camera_positions):
    all_points = []
    # camera_positions = [first_target_position, second_target_position, third_target_position]
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
    return world, my_controller, my_robot, articulation_controller


def is_moving(robot, robot_is_moving):
    print("test : ", robot_is_moving)
    if robot_is_moving is None:
        robot_is_moving = robot.is_moving()
        return robot_is_moving
    else:
        return robot_is_moving

def robot_approach(robot, target_position, target_orientation, is_moving):
    # 선언한 my_controller를 사용하여 action 수행
    actions = my_controller.forward(
        target_position=first_target_position,
        end_effector_offset = np.array([0, 0, 0.0]),
        end_effector_orientation=target_orientation,
        current_joint_positions=robot.get_joints_state().positions
        )
    
    articulation_controller.apply_action(actions)
    # controller의 동작이 끝남 여부를 확인
    
    if is_moving == False:
        print(target_position, " >>> end_effector_orientation",robot._end_effector.get_world_poses()[1][0])
        print(target_orientation, " >>> end_effector_position",robot._end_effector.get_world_poses()[0][0])
        my_controller.reset()
        
def get_camera_image(robot, idx):
    rgb = robot.rgb_cam.get_rgba()
    depth = robot.depth_cam.get_depth()
    depth_clean = []
    if rgb is not None and depth is not None:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_clean = np.where(np.isfinite(depth), depth, 0) # inf, nan > 0으로 변경
        depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX)
        depth_gray = depth_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"rgb_pos_{idx}.png"), rgb_bgr)
        cv2.imwrite(os.path.join(save_dir, f"depth_pos_{idx}.png"), depth_gray)
        print(f"[INFO] Saved images for position {idx}")
        return rgb, depth_clean
    else:
        print("[WARNING] Could not retrieve RGB or Depth image. Returning None, None.")
        return None, None

def scale_rgb_to_depth_fov(rgb_img, intrinsic_rgb, intrinsic_depth):
        """
        RGB 이미지를 Depth 시야각에 맞게 확장하고, 
        확장된 이미지를 Depth 해상도에 맞게 스케일링합니다.
        """
        # 시야각 계산 (가로 및 세로)
        fov_rgb_x = 2 * np.arctan(intrinsic_rgb['cx'] / intrinsic_rgb['fx'])
        fov_rgb_y = 2 * np.arctan(intrinsic_rgb['cy'] / intrinsic_rgb['fy'])

        fov_depth_x = 2 * np.arctan(intrinsic_depth['cx'] / intrinsic_depth['fx'])
        fov_depth_y = 2 * np.arctan(intrinsic_depth['cy'] / intrinsic_depth['fy'])

        rgb_W = rgb_img.shape[1]
        rgb_H = rgb_img.shape[0]

        # RGB 시야에 맞게 Depth 이미지 크기 조정 (가로, 세로)
        scale_x = np.tan(fov_depth_x / 2) / np.tan(fov_rgb_x / 2)
        scale_y = np.tan(fov_depth_y / 2) / np.tan(fov_rgb_y / 2)

        # RGB 이미지 크기 확장 (Depth 시야각에 맞추기 위해 비율 적용)
        expanded_width = int(rgb_W * scale_x)
        expanded_height = int(rgb_H * scale_y)

        # 확장된 이미지 크기 계산
        expanded_rgb_img = np.zeros((expanded_height, expanded_width, 3), dtype=np.uint8)

        # 원본 RGB 이미지를 확장된 중앙에 배치
        x_offset = (expanded_width - rgb_W) // 2
        y_offset = (expanded_height - rgb_H) // 2
        expanded_rgb_img[y_offset:y_offset+rgb_H, x_offset:x_offset+rgb_W] = rgb_img

        # Depth 해상도에 맞게 리사이즈
        depth_W = int(intrinsic_depth['cx'] * 2 + 1)
        depth_H = int(intrinsic_depth['cy'] * 2 + 1)

        resized_rgb_img = cv2.resize(expanded_rgb_img, (depth_W, depth_H), interpolation=cv2.INTER_NEAREST)

        # Depth 크롭 영역 저장 (옵션)
        if save_dir is not None:
            expanded_rgb_bgr = cv2.cvtColor(expanded_rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"expanded_rgb_img{idx}_{i}.png"), expanded_rgb_bgr)

        return resized_rgb_img

def estimate_cube_center_from_rgbd(
        rgb_image, depth_image, cam_position, cam_orientation,
        color_lower, color_upper,
        intrinsic_rgb, intrinsic_depth
        ):
        """
        RGB + Depth 이미지에서 특정 색상 범위를 마스킹하여 해당 객체의 중심 좌표(world 좌표계 기준)를 추정합니다.

        Args:
            rgb_image (np.ndarray): RGB 이미지 (H, W, 3)
            depth_image (np.ndarray): Depth 이미지 (H, W)
            cam_position (np.ndarray): 카메라 위치 (3,)
            cam_orientation (np.ndarray): 카메라 orientation (quaternion, 4,)
            color_lower (np.ndarray): BGR 하한 (예: np.array([0, 0, 120]))
            color_upper (np.ndarray): BGR 상한 (예: np.array([60, 60, 255]))

        Returns:
            np.ndarray or None: 추정된 cube 중심 (world 좌표계, shape=(3,)), 찾지 못하면 None
        """

        # 1. Depth 이미지 시야각 보정 및 리사이즈
        expanded_rgb = scale_rgb_to_depth_fov(rgb_image, intrinsic_rgb, intrinsic_depth)

        # RGB → BGR
        rgb_bgr = cv2.cvtColor(expanded_rgb, cv2.COLOR_RGB2BGR)
        mask = cv2.inRange(rgb_bgr, color_lower, color_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        # 가장 큰 contour 선택
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        # 마스크에서 얻은 RGB 기준 중심 좌표
        cx_rgb = int(M["m10"] / M["m00"])
        cy_rgb = int(M["m01"] / M["m00"])

        # RGB 픽셀 좌표를 Depth 해상도에 맞게 변환
        depth_H, depth_W = depth_image.shape[:2]

        cx_depth = int(round(cx_rgb))
        cy_depth = int(round(cy_rgb))

        # 유효성 검사
        if not (0 <= cx_depth < depth_W and 0 <= cy_depth < depth_H):
            print(f"[Warning] Scaled center ({cx_depth},{cy_depth}) out of bounds")
            return None

        z = depth_image[cy_depth, cx_depth]
        print(f"==>> z: {z}")

        if not np.isfinite(z) or z < 0.01:
            return None
        # Intrinsic
        fx = intrinsic_depth['fx']
        fy = intrinsic_depth['fy']
        cx_intr = intrinsic_depth['cx']
        cy_intr = intrinsic_depth['cy']
        # W, H = my_robot.depth_cam.get_resolution()
        # a_x = my_robot.depth_cam.get_horizontal_aperture()
        # a_y = my_robot.depth_cam.get_vertical_aperture()
        # f = my_robot.depth_cam.get_focal_length()
        # fx = (f / a_x) * W
        # fy = (f / a_y) * H
        # cx_intr, cy_intr = W / 2, H / 2

        x = -(cx_depth - cx_intr) * z / fx
        y = -(cy_depth - cy_intr) * z / fy
        point_cam = np.array([x, y, z])

        R_world_cam = quat_to_rot_matrix(cam_orientation)

        point_world = R_world_cam @ point_cam + cam_position

        return point_world, cx_depth, cy_depth




if __name__ == "__main__":
    depth_images = []
    camera_positions = []
    robot_is_moving = None
    world, my_controller, my_robot, articulation_controller = setup_world()
    
    state = "APPROACH_1"
    first_target_position = np.array([0.0, -0.5, 0.4])
    second_target_position = np.array([0.0, -0.5, 0.4])
    third_target_position = np.array([0.0, -0.5, 0.4])


    target_orientation_euler_1 = np.array([0.0, 0.0, 0.0]) # euler angles
    target_orientation_euler_2 = np.array([90.0, 0.0, 0.0]) # euler angles
    target_orientation_euler_3 = np.array([90.0, 90.0, 0.0]) # euler angles
    target_orientation_euler_4 = np.array([90.0, 90.0, 90.0]) # euler angles

    target_orientation_1 = euler_angles_to_quat(target_orientation_euler_1, degrees = True, extrinsic=False)
    target_orientation_2 = euler_angles_to_quat(target_orientation_euler_2, degrees = True, extrinsic=False)
    target_orientation_3 = euler_angles_to_quat(target_orientation_euler_3, degrees = True, extrinsic=False)
    target_orientation_4 = euler_angles_to_quat(target_orientation_euler_4, degrees = True, extrinsic=False)
    
    # RGB 카메라 intrinsic 예시
    intrinsic_rgb = {
        'fx': (my_robot.rgb_cam.get_focal_length() / my_robot.rgb_cam.get_horizontal_aperture()) * my_robot.rgb_cam.get_resolution()[0],
        'fy': (my_robot.rgb_cam.get_focal_length() / my_robot.rgb_cam.get_vertical_aperture()) * my_robot.rgb_cam.get_resolution()[1],
        'cx': (my_robot.rgb_cam.get_resolution()[0] - 1) / 2,
        'cy': (my_robot.rgb_cam.get_resolution()[1] - 1) / 2,
    }    

    # Depth 카메라 intrinsic 예시
    intrinsic_depth = {
        'fx': (my_robot.depth_cam.get_focal_length() / my_robot.depth_cam.get_horizontal_aperture()) * my_robot.depth_cam.get_resolution()[0],
        'fy': (my_robot.depth_cam.get_focal_length() / my_robot.depth_cam.get_vertical_aperture()) * my_robot.depth_cam.get_resolution()[1],
        'cx': (my_robot.depth_cam.get_resolution()[0] - 1) / 2,
        'cy': (my_robot.depth_cam.get_resolution()[1] - 1) / 2,
    }

    prev_robot_is_moving = False
    idx = 0 #사진 인덱스
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_playing():

            robot_is_moving = my_robot.is_moving()
            robot_just_stopped = prev_robot_is_moving and (robot_is_moving == False)

            if state == "APPROACH_1":
                robot_approach(my_robot, first_target_position, target_orientation_1, robot_is_moving)
                if robot_just_stopped:
                    rgb, depth = get_camera_image(my_robot, "APPROACH_1")
                    if rgb is not None and depth is not None:
                        # camera_pose = 
                        depth_images.append(depth)
                        # camera_positions.append(camera_pose)
                        state = "APPROACH_2"
            
            elif state == "APPROACH_2":
                robot_approach(my_robot, second_target_position, target_orientation_2, robot_is_moving)
                if robot_just_stopped:
                    rgb, depth = get_camera_image(my_robot, "APPROACH_1")
                    if rgb is not None and depth is not None:
                        # camera_pose = 
                        depth_images.append(depth)
                        # camera_positions.append(camera_pose)
                        state = "APPROACH_3"
                        

            elif state == "APPROACH_3":
                robot_approach(my_robot, third_target_position, target_orientation_3, robot_is_moving)
                if robot_just_stopped:
                    rgb, depth = get_camera_image(my_robot, "APPROACH_1")
                    if rgb is not None and depth is not None:
                        # camera_pose = 
                        depth_images.append(depth)
                        # camera_positions.append(camera_pose)
                        state = "APPROACH_1"
                        depth_images = []

            # elif state == "APPROACH_4":
            #     robot_approach(my_robot, second_target_position, target_orientation_4, robot_is_moving)
            #     if robot_just_stopped:
            #         rgb, depth = get_camera_image(my_robot, "APPROACH_1")
            #         if rgb is not None and depth is not None:
            #             depth_images.append(depth)
            #             # camera_positions.append()
            #             print("depth shape : ", depth.shape)
            #             print("rgb shape : ", rgb.shape)
            #             state = "APPROACH_1"
            #             print(f"==>> state: {state}")
            #             depth_images = []
            


            prev_robot_is_moving = robot_is_moving
            # generate_height_map(depth_images, camera_positions)

            idx += 1
                              

    simulation_app.close()