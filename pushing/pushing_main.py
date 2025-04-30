import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# import omni
from isaacsim import SimulationApp
import numpy as np
import torch

config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)

from abc import abstractmethod
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.prims import RigidPrim as RigidPrimView
from isaacsim.core.prims import SingleRigidPrim as RigidPrim
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.api import World
from isaacsim.core.api.objects.sphere import DynamicSphere
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.cloner.grid_cloner import GridCloner
from isaacsim.storage.native import find_nucleus_server
from isaacsim.core.api.objects import DynamicCuboid




from pxr import UsdPhysics, UsdLux, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema

from terrain_utils import *
from terraincreation import TerrainCreation

#로봇path 
relative_path = "../utils/utils/assets"
absolute_path = os.path.abspath(relative_path)
sys.path.append(absolute_path)

relative_path = "../utils"
absolute_path = os.path.abspath(relative_path)
sys.path.append(absolute_path)

from utils.robots.ur5e_handeye import UR5eHandeye
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController







if __name__ == "__main__":
    # world = World(
    #     stage_units_in_meters=1.0, 
    #     rendering_dt=1.0/60.0,
    #     backend="torch", 
    #     device="cpu"
    # )
    world= World(stage_units_in_meters=1.0)
    scene = world.scene

    #setup robot
    robot_usd_path = os.path.join(absolute_path, "utils/assets/ur5e_handeye_gripper.usd")
    my_robot = UR5eHandeye(
        prim_path="/World/ur5e", # should be unique
        name="my_ur5e",
        # should be unique, used to access the object
        usd_path=robot_usd_path,
        activate_camera=False,
        )
    scene.add(my_robot)

    # Add cube
    cube = DynamicCuboid(
        prim_path = "/World/cube",
        name = "cube",
        position = [0.0, 0.5, 1.0],
        color = np.array([1, 0, 0]),
        size = 0.06,
        mass = 0.01,
        )
    
    scene.add(cube)
    # Add Controller
    my_controller = BasicManipulationController(
        name='basic_manipulation_controller',
        cspace_controller=RMPFlowController(
            name="end_effector_controller_cspace_controller", robot_articulation=my_robot, attach_gripper=True
            ),
        gripper=my_robot.gripper,
        # phase의 진행 속도 설정
        events_dt=[0.008],
    )

    #setup terrain
    num_envs = 100  
    num_per_row = 10
    env_spacing = 0.56*2

    terrain_creation_task = TerrainCreation(name="TerrainCreation", 
                                            num_envs=num_envs,
                                            num_per_row=num_per_row,
                                            env_spacing=env_spacing,
                                            )
                            
    world.add_task(terrain_creation_task)

    # robot control(PD control)을 위한 instance 선언
    articulation_controller = my_robot.get_articulation_controller()
    my_controller.reset()
    # Reset the world = Scene Setup
    world.reset()
    # init target position을 robot end effector의 위치로 지정
    init_target_position = my_robot._end_effector.get_world_poses()[0][0]

    while simulation_app.is_running():
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset(soft=True)
            world.step(render=True)
        else:
            world.step(render=True)

    simulation_app.close()