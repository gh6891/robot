import os
import numpy as np

import matplotlib.pyplot as plt
import cv2
import warp as wp



# from omni.isaac.kit import SimulationApp
from isaacsim import SimulationApp

config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

import isaacsim.replicator as rep

RESOLUTION = (1024, 1024)

OBJ_LOC_MIN = (-50, 5, -50)
OBJ_LOC_MAX = (50, 5, 50)

CAM_LOC_MIN = (100, 0, -100)
CAM_LOC_MAX = (100, 100, 100)

SCALE_MIN = 5
SCALE_MAX = 15

from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import set_stage_up_axis
from isaacsim.core.utils.render_product import set_camera_prim_path, get_camera_prim_path
set_stage_up_axis("y")

# 배경 세팅 (방: Sphere, 바닥: Cylinder)
create_prim("/World/Room", "Sphere", attributes={"radius": 1e3, "primvars:displayColor": [(1.0, 1.0, 1.0)]})
create_prim(
    "/World/Ground",
    "Cylinder",
    position=np.array([0.0, -0.5, 0.0]),
    orientation=euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True),
    attributes={"height": 1, "radius": 1e4, "primvars:displayColor": [(1.0, 1.0, 1.0)]},
)
create_prim("/World/Asset", "Xform")

# 카메라 설정
# camera = rep.create.camera()
set_camera_prim_path("/World/Camera")
camera = get_camera_prim_path("/World/Camera")

render_product = rep.create.render_product(camera, RESOLUTION)
# Annotator 설정 (본 예제에서는 rgb만 설정 / bounding box, segmentation, 등 다양한 annotator 설정가능)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(render_product)
kit.update()

# 두 개의 sphere light 생성
light1 = rep.create.light(light_type="sphere", position=(-450, 350, 350), scale=100, intensity=30000.0)
light2 = rep.create.light(light_type="sphere", position=(450, 350, 350), scale=100, intensity=30000.0)

# Replicator randomizer graph 설정
with rep.new_layer():
# Replicator on_frame trigger 세팅, on_frame 안에서 랜덤화된 새로운 프레임 생성
    with rep.trigger.on_frame():
    # Light 색상 랜덤화
        with rep.create.group([light1, light2]):
            rep.modify.attribute("color", rep.distribution.uniform((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)))
        # 카메라 위치 랜덤화
        with camera:
            rep.modify.pose(
            position=rep.distribution.uniform(CAM_LOC_MIN, CAM_LOC_MAX), look_at=(0, 0, 0)
            )
        # 3개의 큐브를 랜덤한 위치와 색상으로 설정
        for idx in range(3):
            cube = rep.create.cube(
                position=rep.distribution.uniform(OBJ_LOC_MIN, OBJ_LOC_MAX),
                rotation=rep.distribution.uniform((0, -180, 0), (0, 180, 0)),
                scale=rep.distribution.uniform(SCALE_MIN, SCALE_MAX),
                material=rep.create.material_omnipbr(diffuse=rep.distribution.uniform((0.1, 0.1, 0.1), (1, 1, 1))),
                )
            
# 예시 이미지 저장할 디렉토리 생성
out_dir = os.path.join(os.getcwd(), "_out_gen_imgs_cube", "")
os.makedirs(out_dir, exist_ok=True)
# Orchestrator: Replicator graph의 실행을 관리하는 클래스
# Orchestrator를 사용하여 triggering 없이 한 번 실행
rep.orchestrator.preview()
# 데이터셋 생성 변수 설정
num_test_images = 10

# 데이터셋 생성
for i in range(num_test_images):
    # Step - trigger a randomization and a render
    rep.orchestrator.step(rt_subframes=4)
    # RGB 수집 (추후에 다른 GT 추가 예정)
    gt = {
    "rgb": rgb.get_data(device="cuda"),
    }
    # RGB 이미지를 torch tensor로 변환 및 alpha channel 제거
    image = wp.to_torch(gt["rgb"])[..., :3]
    # 이미지 출력
    np_image = image.cpu().numpy()
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", np_image)
    cv2.waitKey(1)
    # 이미지 저장
    cv2.imwrite(os.path.join(out_dir, f"image_{i}.png"), np_image)

rep.orchestrator.stop()
kit.close()