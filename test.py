import numpy as np




min_height, max_height = -0.2, 0.2
step = 0.2
vertical_scale = 0.001
width = 48
length = 48
horizontal_scale = 0.25
downsampled_scale = 0.5

if downsampled_scale is None:
    downsampled_scale = horizontal_scale #0.25

# switch parameters to discrete units
min_height = int(min_height / vertical_scale)  #-0.2/ 0.005 = -40
print(f"==>> min_height: {min_height}")
max_height = int(max_height / vertical_scale) #0.2/ 0.005 = 40
print(f"==>> max_height: {max_height}")
step = int(step / vertical_scale) #1/ 0.005 = 200
print(f"==>> step: {step}")

heights_range = np.arange(min_height, max_height + step, step)
print(f"==>> heights_range: {heights_range}")
height_field_downsampled = np.random.choice(
    heights_range,
    (
        int(width * horizontal_scale / downsampled_scale), #48 * 0.25 / 0.5 = 24
        int(length * horizontal_scale / downsampled_scale), #48 * 0.25 / 0.5 = 24
    ),
)
# print(f"==>> height_field_downsampled: {height_field_downsampled}")