# Seongho Bak, Taewon Kim, 2023

from typing import Optional, List
import numpy as np
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
import carb
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera

class UR5eHandeye(Robot):
    """[summary]
        made by seongho bak.
        modified 'Franka' class
        modified from '~/.local/share/ov/pkg/isaac_sim-2022.2.0/exts/omni.isaac.franka/omni/isaac/franka/franka.py'

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "ur5e".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
            gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur5e",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        activate_camera: Optional[bool] = True,  ### BSH
        rgb_cam_prim_name: Optional[str] = None,    ### BSH
        depth_cam_prim_name: Optional[str] = None,  ### BSH
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        self._rgb_cam = rgb_cam_prim_name         ### BSH
        self._depth_cam = depth_cam_prim_name     ### BSH
        self._activate_camera = activate_camera
        self._previous_joint_positions = None
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/flange"
                # self._end_effector_prim_path = prim_path + "/robotiq_arg2f_base_link"
                # self._end_effector_prim_path = prim_path + "/right_inner_finger_pad"

            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.0, 0.0])
            if gripper_closed_position is None:
                gripper_closed_position = np.array([np.pi*2/9, -np.pi*2/9])
            if rgb_cam_prim_name is None:   ### BSH
                self._rgb_cam_prim_path = prim_path + "/realsense/RGB"
            if depth_cam_prim_name is None:   ### BSH
                self._depth_cam_prim_path = prim_path + "/realsense/Depth"
        else:
            if self._end_effector_prim_name is None:
                # self._end_effector_prim_path = prim_path + "/robotiq_arg2f_base_link"
                self._end_effector_prim_path = prim_path + "/right_inner_finger_pad"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.0, 0.0]) 
            if gripper_closed_position is None:
                gripper_closed_position = np.array([np.pi*2/9, -np.pi*2/9])
            if rgb_cam_prim_name is None:   ### BSH
                self._rgb_cam_prim_path = prim_path + "/realsense/RGB"
            if depth_cam_prim_name is None:   ### BSH
                self._depth_cam_prim_path = prim_path + "/realsense/Depth"
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([-np.pi*2/9, np.pi*2/9])
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
            )
        if self._activate_camera:
            self._rgb_cam = Camera(prim_path=self._rgb_cam_prim_path,
                                frequency=30, resolution=(1920, 1080)) ### BSH
            self._depth_cam = Camera(prim_path=self._depth_cam_prim_path,
                                    #  frequency=30, resolution=(1920, 1080)) ### BSH
                                    frequency=30, resolution=(1280, 720)) ### BSH
        else:
            self._rgb_cam = None
            self._depth_cam = None
        return

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    @property   ### BSH
    def rgb_cam(self) -> Camera:
        """[summary]

        Returns:
            Camera: [description]
        """
        return self._rgb_cam

    @property   ### BSH
    def depth_cam(self) -> Camera:
        """[summary]

        Returns:
            Camera: [description]
        """
        return self._depth_cam

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        if self._activate_camera:
            self._rgb_cam.initialize(physics_sim_view)  ### BSH
            self._depth_cam.initialize(physics_sim_view)  ### BSH
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        """[summary]
        """
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return
    
    def is_moving(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if self._previous_joint_positions is None:
            self._previous_joint_positions = self.get_joints_state().positions
            return True  # 처음 호출 시에는 움직이는 것으로 간주

        current_joint_positions = self.get_joints_state().positions
        if current_joint_positions is None:
            return False  # 위치 정보를 얻을 수 없으면 안움직이는 걸로 간주주

        moving_threshold = 0.001  # 움직임으로 간주할 최소 위치 변화량 (조정 필요)

        for i in range(min(6, len(current_joint_positions))):
            if abs(current_joint_positions[i] - self._previous_joint_positions[i]) > moving_threshold:
                self._previous_joint_positions = current_joint_positions.copy()
                return True  # 위치 변화량이 임계값보다 크면 움직이는 것으로 판단

        self._previous_joint_positions = current_joint_positions.copy()
        return False
    