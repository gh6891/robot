from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
import numpy as np

def setup_world_and_robot():
    # 1. 월드 생성
    world = World()
    world.scene.add_default_ground_plane()
    
    # 3. Franka asset 로딩
    franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka"))

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/random_cube",
            name="fancy_cube",
            position=np.array([0.3, 0.3, 0.3]),
            scale=np.array([0.0515, 0.0515, 0.0515]),
            color=np.array([0, 0, 1.0]),
        )
    )

    # 5. 물리 초기화 (physics handles 생성)
    world.reset()
    #------------------------------------------------setup post load()
    controller = PickPlaceController(
        name = "pick_place_controller",
        gripper = franka.gripper,
        robot_articulation = franka
    )
    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    #------------------------------------------------setup post reset()
    controller.reset()
    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

    return world, franka, cube, controller

def physics_step(franka, cube, controller, world):
    cube_position, _ = cube.get_world_pose()
    goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
    current_joint_positions = franka.get_joint_positions()
    actions = controller.forward(
        picking_position=cube_position,
        placing_position=goal_position,
        current_joint_positions=current_joint_positions,
    )
    franka.apply_action(actions)
    # Only for the pick and place controller, indicating if the state
    # machine reached the final state.
    if controller.is_done():
        world.pause()

def main():
    world, franka, cube, controller = setup_world_and_robot()

    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)

    for i in range(2000):
        physics_step(franka, cube, controller, world)
        world.step(render = True)
        if not world.is_playing():
            break

    simulation_app.close()
if __name__ =="__main__":
    main()