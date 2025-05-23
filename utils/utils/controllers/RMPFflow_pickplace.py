# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from isaacsim.robot_motion import motion_generation as mg
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, UsdGeom
import numpy as np
import torch

class RMPFlowController(mg.MotionPolicyController):
    """[summary]

        Args:
            name (str): [description]
            robot_articulation (Articulation): [description]
            physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
            attach_gripper (bool, optional): [description]. Defaults to False.
        """

    def __init__(
        self, name: str, robot_articulation: SingleArticulation, physics_dt: float = 1.0 / 60.0, attach_gripper: bool = False
    ) -> None:

        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR5e", "RMPflow"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )

        # # NumPy array를 list로 변환
        # if isinstance(self._default_position, torch.Tensor):
        #     self._default_position = self._default_position.numpy()
        # if isinstance(self._default_orientation, torch.Tensor):
        #     self._default_orientation = self._default_orientation.numpy()

        # print(">>>>>>>>>>>>",self._default_position, self._default_orientation)
        # print(">>>>>>>>>>>>",type(self._default_position), type(self._default_orientation))


        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
