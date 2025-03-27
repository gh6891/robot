from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

import carb
import numpy as np

def setup_world_and_robot():
    # 1. 월드 생성
    world = World()
    world.scene.add_default_ground_plane()

    # 2. Nucleus asset 경로 설정
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find nucleus server with /Isaac folder")

    # 3. Jetbot asset 로딩
    asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")

    # 4. Robot 객체 등록
    jetbot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))

    # 5. 물리 초기화 (physics handles 생성)
    world.reset()

    # 6. 컨트롤러 얻기
    controller = jetbot.get_articulation_controller()

    return world, controller

def send_robot_actions(controller):
    # 두 개의 바퀴에 랜덤 속도 명령
    velocities = np.random.rand(2,) * 5.0
    controller.apply_action(ArticulationAction(joint_velocities=velocities))

def main():
    world, controller = setup_world_and_robot()

    for i in range(2000):
        send_robot_actions(controller)
        world.step(render=True)

    simulation_app.close()

if __name__ == "__main__":
    main()
