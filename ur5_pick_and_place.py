from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.universal_robots.controllers import PickPlaceController
import numpy as np
import carb

def setup_world_and_robot():
    # 1. 월드 생성
    world = World()
    world.scene.add_default_ground_plane()

    asset_path = get_assets_root_path() + "/Robots/UniversalRobots/ur10e/ur10.usd"
    add_reference_to_stage(asset_path, "/World/UR10")
    ur10 = world.scene.add(Robot(prim_path="/World/UR10", name="ur10"))

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
    return world, ur10, cube

def physics_step(ur10):
    joint_targets = np.array([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
    robot.apply_action(ArticulationAction(joint_positions=joint_targets))
    
def main():
    world, ur10, cube = setup_world_and_robot()
    controller = ur10.get_articulation_controller()

    for i in range(300):
        if i < 100:
            controller.apply_action(ArticulationAction(joint_positions=np.array([0, -1.2, 1.5, 0, 1.2, 0])))
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()