import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.stage import update_stage
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
# from isaacsim.cortex.framework.robot import CortexUr10
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_setup.assembler import create_fixed_joint

from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import get_all_matching_child_prims
import numpy as np

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
robot_asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
gripper_asset_path = assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"

add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/ur5e")
add_reference_to_stage(usd_path=gripper_asset_path, prim_path="/World/ur5e/Gripper/Robotiq_2F_85edit")

parent_path = "/World/ur5e/ur5e/wrist_3_link"  # UR5e의 wrist_3_link
child_path = "/World/ur5e/Gripper/Robotiq_2F_85_edit/Robotiq_2F_85/base_link"  # 그리퍼의 베이스

create_fixed_joint(
    prim_path="/World/ur5e/gripper_fixed_joint",  # 조인트 프림 이름 (중복 안 되게)
    parent_path=parent_path,
    child_path=child_path
)

all_prims = get_all_matching_child_prims("/World/ur5e/Gripper")
for prim in all_prims:
    print(prim.GetPath())

# gripper = ParallelGripper(
#     #We chose the following values while inspecting the articulation
#     end_effector_prim_path="/World/ur5e/Gripper/Robotiq_2F_85edit/Robotiq_2F_85/base_link",
#     joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
#     joint_opened_positions=np.array([0, 0]),
#     joint_closed_positions=np.array([0.628, -0.628]),
#     action_deltas=np.array([-0.628, 0.628]),
# )

# my_ur5 = my_world.scene.add(SingleManipulator(
#     prim_path="/World/ur5e",
#     name="ur5e_robbot",
#     end_effector_prim_name="Gripper/Robotiq_2F_85edit/Robotiq_2F_85/base_link",
#     # gripper=gripper
#     ))

# # 월드 초기화 및 업데이트
# joints_default_positions = np.zeros(12)

# my_ur5.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()

my_world.reset()

# 모든 child prim 경로 출력
print("\n--- All prims under /World ---")
all_prims = get_all_matching_child_prims("/World")
for prim in all_prims:
    print(prim.GetPath())

# 시뮬레이션 실행 (GUI 창이 떠서 확인 가능)
while simulation_app.is_running():
    my_world.step(render=True)

# 시뮬레이션 종료
simulation_app.close()