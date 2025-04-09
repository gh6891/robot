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

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()

# use Isaac Sim provided asset
robot_asset_path = assets_root_path + "/Isaac/Robots/UR10/ur10e_robotiq2f-140.usd"

add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/ur10e_robotiq2f_140")

all_prims = get_all_matching_child_prims("/World")
for prim in all_prims:
    print(prim.GetPath())


# define the gripper
gripper = ParallelGripper(
    #We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/ur10e_robotiq2f_140/Robotiq_2F_140_config/robotiq_base_link",
    joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0.0, 0.0]),
    joint_closed_positions=np.array([0.785, 0.785]),
    action_deltas=np.array([0.785, 0.785])
)

# raise SystemExit(0)


#define the manipulator
my_ur10 = my_world.scene.add(SingleManipulator(
    prim_path="/World/ur10e_robotiq2f_140",
    name="UR10e",
    end_effector_prim_name = "Robotiq_2F_140_config/robotiq_base_link",
    gripper=gripper,
))

#set the default positions of the other gripper joints to be opened so
#that its out of the way of the joints we want to control when gripping an object for instance.
# joints_default_positions = np.zeros(14)
joints_default_positions = my_ur10.get_joint_positions()
# joints_default_positions[1] = -np.pi/2
# joints_default_positions[0] = 0.628
print("================: ", joints_default_positions)

# my_ur10.set_joints_default_state(positions=joints_default_positions)

my_world.scene.add_default_ground_plane()
my_world.reset()

joints_default_positions = my_ur10.get_joint_positions()
print("================: ", joints_default_positions)

arm_joint_indices = np.array([0, 1, 2, 3, 4, 5])

articulation_controller = my_ur10.get_articulation_controller()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_ur10.gripper.get_joint_positions()
        print(f"==>> gripper_positions: {gripper_positions}")
        joints_default_positions[1] = -np.pi/2
        joints_default_positions[2] = np.sin(i/100) * 0.628
        arm_values = joints_default_positions[arm_joint_indices]
        print(f"==>> joints_default_positions: {joints_default_positions}")
        print(f"==>> arm_values: {arm_values}")
        actions = ArticulationAction(joint_positions=arm_values, joint_indices=arm_joint_indices)
        print(f"==>> actions: {actions}")
        articulation_controller.apply_action(actions)
        print("full: ", my_ur10.get_joint_positions())
        if i // 500 % 2 == 0:
            print("close the gripper")
            #close the gripper slowly
            my_ur10.gripper.close()
            my_ur10.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.1, gripper_positions[1] - 0.1]))
        else :
            print("open the gripper")
            #open the gripper slowly
            my_ur10.gripper.open()
            # gripper_target[0] = max(0.0, gripper_target[0] - 0.02)
            # gripper_target[1] = min(0.0, gripper_target[1] - 0.02)
            my_ur10.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.1, gripper_positions[1] + 0.1]))
simulation_app.close()