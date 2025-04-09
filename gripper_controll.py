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
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import get_all_matching_child_prims
import numpy as np

from isaacsim.core.articulation import Articulation_view
# from omni.isaac.core.articulations import ArticulationView

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
gripper_asset_path = assets_root_path + "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
gripper_prim_path = "/World/Gripper/Robotiq_2F_85edit"

add_reference_to_stage(usd_path=gripper_asset_path, prim_path=gripper_prim_path)

gripper = ParallelGripper(
    #We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/Gripper/Robotiq_2F_85edit/Robotiq_2F_85/base_link",
    joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0, 0]),
    joint_closed_positions=np.array([0.785, 0.785]),
    action_deltas=np.array([0.785, 0.785]),
)
gripper_view = Articulation_view(
    prim_path=gripper_prim_path,
    name="gripper_view",
)

my_world.scene.add_default_ground_plane()
my_world.reset()

gripper.initialize(
    gripper_prim_path,
    get_joint_positions_func=gripper_view.get_joint_positions,
    set_joint_positions_func=gripper_view.set_joint_positions,
    dof_names=gripper_view.dof_names
)
# 모든 child prim 경로 출력
print("\n--- All prims under /World ---")
all_prims = get_all_matching_child_prims("/World")
for prim in all_prims:
    print(prim.GetPath())



i = 0
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        i += 1
        gripper_positions = gripper.get_joint_positions()
        print(f"==>> gripper_positions: {gripper_positions}")
       
        if i // 500 % 2 == 0:
            print("close the gripper")
            #close the gripper slowly
            gripper.close()
        else :
            print("open the gripper")
            #open the gripper slowly
            gripper.open()

simulation_app.close()