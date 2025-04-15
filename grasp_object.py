from isaacsim import SimulationApp
config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

import sys, os
import random
import numpy as np

lecture_dir = "/home/gh6891/robot/utils"
sys.path.append(lecture_dir)

from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController

#로봇의 기본적인 매니퓰레이션 동작을 위한 환경 설정#

# World 선언
my_world = World(stage_units_in_meters=1.0)

# Initialize the Scene
scene = my_world.scene
scene.add_default_ground_plane()

# Add cube
cube = DynamicCuboid(
prim_path = "/World/cube",
name = "cube",
position = [0.5, 0.5, 0.1],
color = np.array([0, 0, 1]),
size = 0.04,
mass = 0.01,
)
scene.add(cube)

# Add robot
robot_usd_path = os.path.join(lecture_dir, "utils/assets/ur5e_handeye_gripper.usd")
my_robot = UR5eHandeye(
prim_path="/World/ur5e", # should be unique
name="my_ur5e",
# should be unique, used to access the object
usd_path=robot_usd_path,
activate_camera=False,
)
scene.add(my_robot)

############################### Pick place controller 생성 ###############################
# Add Controller
my_controller = BasicManipulationController(
    # Controller의 이름 설정
    name='basic_manipulation_controller',
    # 로봇 모션 controller 설정
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_robot, attach_gripper=True
        ),
    # 로봇의 gripper 설정
    gripper=my_robot.gripper,
    # phase의 진행 속도 설정
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_robot.get_articulation_controller()
my_controller.reset()
# Reset the world = Scene Setup
my_world.reset()

# init target position을 robot end effector의 위치로 지정
init_target_position = my_robot._end_effector.get_world_poses()[0][0]
state = "APPROACH"
while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step
    my_world.step(render=True)
    print(f"==>> init_target_position: {init_target_position}")
    # world가 동작하는 동안 작업 수행
    if my_world.is_playing():# Get transformation
        # gripper에 대한 offset 선언
        end_effector_offset = np.array([0, 0, 0.24])

        # APPROACH 하는 state에 대한 action 수행
        if state == "APPROACH":
            # cube 위치 얻어오기
            cur_cube_position = cube.get_world_pose()[0]
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.forward(
                target_position=cur_cube_position,
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset
                )
            articulation_controller.apply_action(actions)
                    # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done position control of end-effector")
                my_controller.reset()
                # APPROACH가 끝났을 경우 GRASP state 단계로 변경
                state = "GRASP"

        # GRASP 하는 state에 대한 action 수행
        elif state == "GRASP":
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.close(
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset
                )
            articulation_controller.apply_action(actions)
            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done grasping")
                my_controller.reset()
                # GRASP가 끝났을 경우 LIFT state 단계로 변경
                state = "LIFT"

        # LIFT 하는 state에 대한 action 수행
        elif state == "LIFT":
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.forward(
                target_position=init_target_position,
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset
                )
            articulation_controller.apply_action(actions)
            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done lifting")
                my_controller.reset()
                # LIFT가 끝났을 경우 OPEN state 단계로 변경
                state = "OPEN"

        # OPEN 하는 state에 대한 action 수행
        elif state == "OPEN":
            # 선언한 my_controller를 사용하여 action 수행
            actions = my_controller.open(
                current_joint_positions=my_robot.get_joints_state().positions,
                end_effector_offset = end_effector_offset
                )
            articulation_controller.apply_action(actions)
            # controller의 동작이 끝남 여부를 확인
            if my_controller.is_done():
                print("done lifting")
                my_controller.reset()
                # LIFT가 끝났을 경우 APPROACH state 단계로 변경
                state = "APPROACH"