from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api import World
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_all_matching_child_prims

import numpy as np

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
asset_path = assets_root_path + "/Isaac/Robots/Denso/cobotta_pro_900.usd"

#TODO: change this to your own path if you downloaded the asset
# asset_path = "/home/user_name/cobotta_pro_900/cobotta_pro_900/cobotta_pro_900.usd"

add_reference_to_stage(usd_path=asset_path, prim_path="/World/cobotta")


all_prims = get_all_matching_child_prims("/World")
for prim in all_prims:
    print(prim.GetPath())

#define the gripper
gripper = ParallelGripper(
    #We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/cobotta/onrobot_rg6_base_link",
    joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0, 0]),
    joint_closed_positions=np.array([0.628, -0.628]),
    action_deltas=np.array([-0.628, 0.628]),
)
#define the manipulator
my_denso = my_world.scene.add(SingleManipulator(
    prim_path="/World/cobotta",
    name="cobotta_robot",
    end_effector_prim_name="onrobot_rg6_base_link",
    gripper=gripper
    ))
#set the default positions of the other gripper joints to be opened so
#that its out of the way of the joints we want to control when gripping an object for instance.
joints_default_positions = np.zeros(12)
joints_default_positions[7] = 0.628
joints_default_positions[8] = 0.628
my_denso.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()
my_world.reset()

articulation_controller = my_denso.get_articulation_controller()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        joints_default_positions[2] = np.sin(i/100) * 0.628
        articulation_controller.apply_action(
            ArticulationAction(joint_positions=joints_default_positions))
        gripper_positions = my_denso.gripper.get_joint_positions()
        print("full: ", my_denso.get_joint_positions())
        print(f"==>> gripper_positions: {gripper_positions}")
        if i < 500:
            #close the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.3, gripper_positions[1] - 0.3]))
        if i > 500:
            #open the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.3, gripper_positions[1] + 0.3]))
        if i == 1000:
            i = 0

simulation_app.close()