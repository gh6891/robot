from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})



from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.prims import get_all_matching_child_prims

import numpy as np

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
# use Isaac Sim provided asset
asset_path = assets_root_path + "/Robots/UniversalRobots/ur10e/ur10.usd"

add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur10e")

all_prims = get_all_matching_child_prims(prim_path="/World/ur10e")
print("All Prims in /World/ur10e:")
for prim in all_prims:
    print(prim.GetPath())
print("UR10: ", is_prim_path_valid("/World/ur10e"))
print("Gripper: ", is_prim_path_valid("/World/ur10e/Gripper"))  # True/False
print("gripper: ", is_prim_path_valid("/World/ur10e/gripper"))  # True/False
#define the gripper
gripper = ParallelGripper(
    #We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/ur10e/gripper",
    joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0, 0]),
    joint_closed_positions=np.array([0.628, -0.628]),
    action_deltas=np.array([-0.628, 0.628]),
)

#define the manipulator
my_ur10 = my_world.scene.add(SingleManipulator(
    prim_path="/World/ur10e",
    name="UR10e",
    end_effector_prim_name="gripper",
    gripper=gripper,
))
#set the default positions of the other gripper joints to be opened so
#that its out of the way of the joints we want to control when gripping an object for instance.
joints_default_positions = np.zeros(12)
joints_default_positions[7] = 0.628
joints_default_positions[8] = 0.628
my_ur10.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()
my_world.reset()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_ur10.gripper.get_joint_positions()
        if i < 500:
            #close the gripper slowly
            my_ur10.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.1, gripper_positions[1] - 0.1]))
        if i > 500:
            #open the gripper slowly
            my_ur10.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.1, gripper_positions[1] + 0.1]))
        if i == 1000:
            i = 0

simulation_app.close()