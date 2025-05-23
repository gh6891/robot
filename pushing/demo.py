import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# import omni
from isaacsim import SimulationApp
import numpy as np
import torch

simulation_app = SimulationApp({"headless": False})

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

from pxr import UsdPhysics, UsdLux, UsdShade, Sdf, Gf, UsdGeom, PhysxSchema

from terrain_utils import *


class TerrainCreation(BaseTask):
    def __init__(self, name, num_envs, num_per_row, env_spacing, config=None, offset=None,) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._num_terrain = 8
        self._num_envs = num_envs
        self._num_per_row = num_per_row
        self._env_spacing = env_spacing
        self._device = "cpu"
        self._cloner = GridCloner(self._env_spacing, self._num_per_row)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

    @property
    def default_base_env_path(self):
        return "/World/envs"
    
    @property
    def default_zero_env_path(self):
        return f"{self.default_base_env_path}/env_0"

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(2000)

        self.get_terrain()
        self.get_ball()

        super().set_up_scene(scene)
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        print(f"cloning {self._num_envs} environments...")
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", 
            prim_paths=prim_paths
        )
        return
    
    def get_terrain(self):
        # create all available terrain types
        num_terains = self._num_terrain
        terrain_width = 12.
        terrain_length = 12.
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale) #48
        num_cols = int(terrain_length/horizontal_scale) #48
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16) # (384, 48)

        def new_sub_terrain(): 
            return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.3, max_height=0.3, step=0.01, downsampled_scale=0.5).height_field_raw
        heightfield[num_rows:2*num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
        heightfield[2*num_rows:3*num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
        heightfield[3*num_rows:4*num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5, min_size=1., max_size=5., num_rects=20).height_field_raw
        heightfield[4*num_rows:5*num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2., amplitude=1.).height_field_raw
        heightfield[5*num_rows:6*num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
        heightfield[6*num_rows:7*num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75, step_height=-0.5).height_field_raw
        heightfield[7*num_rows:8*num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
                                                                        stone_distance=1., max_height=0.5, platform_size=0.).height_field_raw
        
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)

        position = np.array([-6.0, 48.0, 0])
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])
        add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position, orientation=orientation)

    def get_ball(self):
        ball = DynamicSphere(prim_path=self.default_zero_env_path + "/ball",
                             name="ball",
                             translation=np.array([0.0, 0.0, 1.0]),
                             mass=0.5,
                             radius=0.2,)
    
    def post_reset(self):
        for i in range(self._num_envs):
            ball_prim = self._stage.GetPrimAtPath(f"{self.default_base_env_path}/env_{i}/ball")
            color = 0.5 + 0.5 * np.random.random(3)
            visual_material = PreviewSurface(prim_path=f"{self.default_base_env_path}/env_{i}/ball/Looks/visual_material", color=color)
            binding_api = UsdShade.MaterialBindingAPI(ball_prim)
            binding_api.Bind(visual_material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def get_observations(self):
        pass

    def calculate_metrics(self) -> None:
        pass

    def is_done(self) -> None:
        pass
    
    
if __name__ == "__main__":
    world = World(
        stage_units_in_meters=1.0, 
        rendering_dt=1.0/60.0,
        backend="torch", 
        device="cpu"
    )

    num_envs = 800
    num_per_row = 80
    env_spacing = 0.56*2

    terrain_creation_task = TerrainCreation(name="TerrainCreation", 
                                            num_envs=num_envs,
                                            num_per_row=num_per_row,
                                            env_spacing=env_spacing,
                                            )
                            
    world.add_task(terrain_creation_task)
    world.reset()

    while simulation_app.is_running():
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset(soft=True)
            world.step(render=True)
        else:
            world.step(render=True)

    simulation_app.close()