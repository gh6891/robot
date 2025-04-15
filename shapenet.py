import argparse
import os
from isaacsim import SimulationApp
import carb

if "SHAPENET_LOCAL_DIR" not in os.environ:
    carb.log_error("SHAPENET_LOCAL_DIR not defined:")
    carb.log_error("Please specify the SHAPENET_LOCAL_DIR environment variable to the location of your local shapenet database, exiting")
exit()
kit = SimulationApp()

from isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.shapenet")

