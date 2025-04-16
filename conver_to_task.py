# Initialize SimulationAPP
from isaacsim import SimulationApp
config = {
'width': 1920,
'height': 1080,
'headless': False,
}
simulation_app = SimulationApp(config)
# print(simulation_app.DEFAULT_LAUNCHER_CONFIG)
# Import isaacsim python API
from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
import omni.usd
from pxr import Gf, UsdGeom
# Import necessary libraries
import sys, os
import numpy as np
# Import Custom libraries
lecture_dir = "/home/gh6891/robot/utils"
sys.path.append(lecture_dir)
from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController


class CustomTask(BaseTask):
    def __init__(self, name):
        BaseTask.__init__(self, name)

        self._robot = None
        self._target_cube = None

        self._robot_usd_path = os.path.join(lecture_dir, "utils/assets/ur5e_handeye_gripper.usd")
        return
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        # Add cube
        self._target_cube = VisualCuboid(
            prim_path = "/World/cube",
            name = "cube",
            position = [0.5, 0.5, 0.1],
            color = np.array([0, 0, 1]),
            size = 0.04,
            )
        scene.add(self._target_cube)
        # Add robot
        self._robot = UR5eHandeye(
            prim_path="/World/ur5e", # should be unique
            name="my_ur5e",
            # should be unique, used to access the object
            usd_path=self._robot_usd_path,
            activate_camera=False,
            )
        scene.add(self._robot)
        stage = omni.usd.get_context().get_stage()

        # Define the prim path
        wrist_path = "/World/ur5e/robotiq_arg2f_base_link"
        left_finger_path = "/World/ur5e/left_inner_finger_pad"
        right_finger_path = "/World/ur5e/right_inner_finger_pad"
        # Get the prim using the path
        wrist = stage.GetPrimAtPath(wrist_path)
        left_finger = stage.GetPrimAtPath(left_finger_path)
        right_finger = stage.GetPrimAtPath(right_finger_path)
        # Get the Xformable API to access transformation attributes
        self.wrist = UsdGeom.Xformable(wrist)
        self.left_finger = UsdGeom.Xformable(left_finger)
        self.right_finger = UsdGeom.Xformable(right_finger)

    def get_observations(self) -> dict:
        target_position, target_orientation = self._target_cube.get_world_pose()
        # Get the local transformation matrix
        wrist_transform = self.wrist.ComputeLocalToWorldTransform(0)
        left_finger_transform = self.left_finger.ComputeLocalToWorldTransform(0)
        right_finger_transform = self.right_finger.ComputeLocalToWorldTransform(0)
        # Extract the translation (position) from the transformation matrix
        wrist_position = wrist_transform.ExtractTranslation()
        left_finger_position = left_finger_transform.ExtractTranslation()
        right_finger_position = right_finger_transform.ExtractTranslation()
        # For Gipper offset
        offset = wrist_position - (left_finger_position+right_finger_position)/2
        target_position += offset
        return {
        "target_position": target_position
        }
    
    # Not Done while Application is running
    def is_done(self) -> bool:
        return False
    
    # World 선언
my_world = World(stage_units_in_meters=1.0)
# task 선언
my_task = CustomTask(name="my_task")
my_world.add_task(my_task)
# Reset the world = Scene Setup
my_world.reset()
# robot controller 생성
my_ur5e = my_task._robot
my_controller = RMPFlowController(
name="end_effector_controller_cspace_controller",
robot_articulation=my_ur5e,
attach_gripper=True
)
# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step
    my_world.step(render=True)
    # world가 동작하는 동안 작업 수행
    if my_world.is_playing():
        observations = my_world.get_observations()
        # some logic using the observations
        target_position = observations["target_position"]
        # 선언한 my_controller를 사용하여 action 수행
        actions = my_controller.forward(
        target_end_effector_position=target_position,
        )
        articulation_controller.apply_action(actions)
        # task의 끝남 여부를 확인
        if my_task.is_done():
            break
simulation_app.close()