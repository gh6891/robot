import numpy as np
import os

save_dir = "/home/gh6891/robot/pushing/test_image"
os.makedirs(save_dir, exist_ok=True)

load_path = os.path.join(save_dir, "heightmap_array.npy")
height_map = np.load(load_path)
print(height_map.max())
print(height_map.min())