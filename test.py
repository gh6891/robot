import numpy as np
from scipy.spatial.transform import Rotation as R

base_to_flange_euler = np.array([0, 0, 0])
flange_to_tool_euler = np.array([90.0, 0, 90.0])

# 각 오일러 각도를 Rotation 객체로 변환 (URDF의 rpy는 일반적으로 외재적 zyx 순서)
r_base_to_flange = R.from_euler('zyx', base_to_flange_euler, degrees=True)
r_flange_to_tool = R.from_euler('zyx', flange_to_tool_euler, degrees=True)

r_base_to_tool = r_base_to_flange * r_flange_to_tool

# 여기서는 XYZ 순서와 ZYX 순서로 변환하여 비교해봅니다.
base_to_tool_euler_xyz = r_base_to_tool.as_euler('xyz', degrees=True)
print(f"==>> base_to_tool_euler_xyz: {base_to_tool_euler_xyz}")
base_to_tool_euler_zyx = r_base_to_tool.as_euler('zyx', degrees=True)
print(f"==>> base_to_tool_euler_zyx: {base_to_tool_euler_zyx}")


