from isaacsim import SimulationApp
config = {
'width': 1920,
'height': 1080,
'headless': False
}
simulation_app = SimulationApp(config)
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)