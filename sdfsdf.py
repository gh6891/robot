from omni.isaac.kit import SimulationApp

# SimulationApp 초기화 (headless 모드로 실행)
simulation_app = SimulationApp({"headless": True})

# 확장 활성화 확인
from omni.kit.app import get_app
ext_manager = get_app().get_extension_manager()
ext_manager.set_extension_enabled("isaacsim.robot_setup.assembler", True)

# 필요한 모듈 import
from isaacsim.robot_setup import assembler

# 테스트: assembler 모듈의 내용 출력
print(dir(assembler))

# SimulationApp 종료
simulation_app.close()