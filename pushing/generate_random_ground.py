from isaacsim import SimulationApp
config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

import sys, os
import random
import numpy as np
from pxr import UsdGeom, Usd, Gf, Sdf
import omni.usd

def generate_random_gaussian_map(size=(100, 100), num_gaussians=5, sigma_range=(5, 15), amplitude_range=(0.5, 1.5)):
    height_map = np.zeros(size)
    x = np.arange(0, size[0])
    y = np.arange(0, size[1])
    X, Y = np.meshgrid(x, y)

    for _ in range(num_gaussians):
        x0 = np.random.uniform(0, size[0])
        y0 = np.random.uniform(0, size[1])
        sigma = np.random.uniform(*sigma_range)
        amplitude = np.random.uniform(*amplitude_range)

        gaussian = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        height_map += gaussian

    height_map -= height_map.min()
    height_map /= height_map.max()
    return height_map