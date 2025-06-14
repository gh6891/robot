import os, sys
# ───────────────────────────── #
# 디렉토리 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

save_dir = os.path.join(SCRIPT_DIR, "test_image")
os.makedirs(save_dir, exist_ok=True)
# ───────────────────────────── #

import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import gaussian_filter1d

pixel_size_mm = 25  # 기본 설정

def pixel_to_mm(y_px, x_px, height_map, pixel_size_mm):
    height_px, width_px = height_map.shape
    x_mm = (x_px - width_px / 2 + 0.5) * pixel_size_mm
    y_mm = (height_px - y_px - 0.5) * pixel_size_mm
    return x_mm, y_mm

def save_original_heightmap(height_map, pixel_size_mm):
    height_px, width_px = height_map.shape
    map_width_mm = width_px * pixel_size_mm
    map_height_mm = height_px * pixel_size_mm
    extent = [-map_width_mm / 2, map_width_mm / 2, 0, map_height_mm]
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='terrain', extent=extent, origin='upper', vmin=-30, vmax=30)
    plt.title("Height Map")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar(label="Height")
    plt.savefig(os.path.join(save_dir, "height_map_only.png"))
    plt.close()

def add_random_start_goal_and_save(height_map, pixel_size_mm):
    height_px, width_px = height_map.shape
    map_width_mm = width_px * pixel_size_mm
    map_height_mm = height_px * pixel_size_mm
    extent = [-map_width_mm / 2, map_width_mm / 2, 0, map_height_mm]
    start_px = (np.random.randint(height_px), np.random.randint(width_px))
    goal_px = (np.random.randint(height_px), np.random.randint(width_px))
    start_mm = pixel_to_mm(*start_px, height_map, pixel_size_mm)
    goal_mm = pixel_to_mm(*goal_px, height_map, pixel_size_mm)

    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='terrain', extent=extent, origin='upper', vmin=-30, vmax=30)
    plt.plot(start_mm[0], start_mm[1], 'o', markersize=4, color='blue', label='Start')
    plt.plot(goal_mm[0], goal_mm[1], 'x', markersize=4, color='red', label='Goal')
    plt.title("Height Map with Random Start & Goal")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar(label="Height")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "height_map_with_start_goal.png"))
    plt.close()

    return start_px, goal_px, start_mm, goal_mm

def astar(height_map, pixel_size_mm, start, goal, alpha=1.0):
    """
    A* 알고리즘 기반의 경로 탐색
    - start, goal: (y, x) 형태의 픽셀 단위 좌표
    - alpha: 고도 변화에 따른 비용 반영 계수
    """
    height, width = height_map.shape
    map_width_mm = width * pixel_size_mm
    map_height_mm = height * pixel_size_mm
    extent = [-map_width_mm / 2, map_width_mm / 2, 0, map_height_mm]
    came_from = {}

    # 비용 배열 (g: 시작 → 현재까지의 비용, f: g + 휴리스틱)
    g_score = np.full((height, width), np.inf)
    f_score = np.full((height, width), np.inf)

    g_score[start] = 0
    f_score[start] = np.linalg.norm(np.array(start) - np.array(goal))

    # 우선순위 큐: (f_score, y, x)
    open_set = [(f_score[start], start)]
    open_set_hash = {start}  # 중복 방지용 집합

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)

        if current == goal:
            # 경로 재구성
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path = path[::-1]

            if path is not None:
                """
                A* 경로 시각화 (픽셀 경로를 mm 단위로 변환하여 시각화)
                """
                plt.figure(figsize=(10, 8))
                plt.imshow(height_map, cmap='terrain', extent=extent, origin='upper', vmin=-30, vmax=30)

                path_mm = [pixel_to_mm(y, x, height_map, pixel_size_mm) for y, x in path]
                xs, ys = zip(*path_mm)

                plt.plot(xs, ys, 'w-', linewidth=1.5, label='A* Path')
                plt.plot(xs[0], ys[0], 'o', markersize=4, color='blue', label='Start')
                plt.plot(xs[-1], ys[-1], 'x', markersize=4, color='red', label='Goal')
                plt.title("A* Path on Height Map")
                plt.xlabel("X (mm)")
                plt.ylabel("Y (mm)")
                plt.colorbar(label="Height")
                plt.legend()
                plt.savefig(os.path.join(save_dir, "height_map_path.png"))
                plt.close()
            else:
                print("경로를 찾지 못했습니다.")

            return path

        cy, cx = current

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    tentative_g = g_score[current] + np.linalg.norm([dy, dx])
                    height_cost = alpha * abs(height_map[ny, nx] - height_map[cy, cx]) / np.linalg.norm([dy, dx])
                    total_g = tentative_g + height_cost

                    if total_g < g_score[ny, nx]:
                        came_from[(ny, nx)] = current
                        g_score[ny, nx] = total_g
                        f = total_g + np.linalg.norm(np.array((ny, nx)) - np.array(goal))
                        f_score[ny, nx] = f
                        heapq.heappush(open_set, (f_score[ny, nx], (ny, nx)))
                        if (ny, nx) not in open_set_hash:
                            heapq.heappush(open_set, (f, (ny, nx)))
                            open_set_hash.add((ny, nx))

    return None  # 경로 없음

def save_path_comparison(height_map, pixel_size_mm, path):
    """
    원본 경로와 부드러운 경로를 함께 시각화하고 저장하는 함수
    - path: A* 경로 결과 (pixel 좌표 리스트)
    - pixel_to_mm_fn: pixel_to_mm(y_px, x_px, height_map, pixel_size_mm) 함수
    """
    height, width = height_map.shape
    map_width_mm = width * pixel_size_mm
    map_height_mm = height * pixel_size_mm
    extent = [-map_width_mm / 2, map_width_mm / 2, 0, map_height_mm]
    # 원본 경로 mm 단위로 변환
    path_mm = np.array([pixel_to_mm(y, x, height_map, pixel_size_mm) for y, x in path])
    xs, ys = path_mm[:, 0], path_mm[:, 1]

    # 부드럽게 만들기 (가우시안 필터)
    xs_smooth = gaussian_filter1d(xs, sigma=2, mode='nearest')
    ys_smooth = gaussian_filter1d(ys, sigma=2, mode='nearest')

    # 시작점, 목표점은 원래 경로로 덮기
    xs_smooth[0], ys_smooth[0] = xs[0], ys[0]
    xs_smooth[-1], ys_smooth[-1] = xs[-1], ys[-1]

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='terrain', extent=extent, origin='upper', vmin=-30, vmax=30)
    plt.plot(xs, ys, 'b-', linewidth=1.5, label='Original Path')
    plt.plot(xs_smooth, ys_smooth, 'r-', linewidth=1.5, label='Smoothed Path')
    plt.plot(xs[0], ys[0], 'o', markersize=4, color='blue', label='Start')
    plt.plot(xs[-1], ys[-1], 'x', markersize=4, color='red', label='Goal')

    plt.title("A* vs Smoothed Path")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar(label="Height")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "path_gaussian_smoothing_comparison.png"))
    plt.close()

def catmull_rom_spline_2d(points, n_points=100):
    """
    Catmull-Rom Spline for 2D points
    points: (N, 2) array
    n_points: total number of points to generate
    """
    def interpolate(p0, p1, p2, p3, t):
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t**3
        )

    P = np.array(points)
    result = []

    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, n_points // (len(P) - 3)):
            result.append(interpolate(P[i-1], P[i], P[i+1], P[i+2], t))
    
    # 첫 점과 마지막 점은 원래대로 삽입
    result.insert(0, P[0])
    result.append(P[-1])
    return np.array(result)

def save_path_spline_comparison(height_map, pixel_size_mm, path):
    """
    Catmull-Rom Spline으로 보간한 부드러운 경로 시각화
    """
    height, width = height_map.shape
    map_width_mm = width * pixel_size_mm
    map_height_mm = height * pixel_size_mm
    extent = [-map_width_mm / 2, map_width_mm / 2, 0, map_height_mm]

    # A* path를 mm 단위로 변환
    path_mm = np.array([pixel_to_mm(y, x, height_map, pixel_size_mm) for y, x in path])

    # 제어점이 너무 적으면 spline 불가
    if len(path_mm) < 4:
        print("[WARNING] Too few points for spline. Skipping smoothing.")
        smooth_path = path_mm
    else:
        smooth_path = catmull_rom_spline_2d(path_mm, n_points=200)

    xs, ys = path_mm[:, 0], path_mm[:, 1]
    xs_smooth, ys_smooth = smooth_path[:, 0], smooth_path[:, 1]

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='terrain', extent=extent, origin='upper', vmin=-30, vmax=30)
    plt.plot(xs, ys, 'b-', linewidth=1.5, label='Original A* Path')
    plt.plot(xs_smooth, ys_smooth, 'r-', linewidth=1.5, label='Spline Smoothed Path')
    plt.plot(xs[0], ys[0], 'o', markersize=4, color='blue', label='Start')
    plt.plot(xs[-1], ys[-1], 'x', markersize=4, color='red', label='Goal')

    plt.legend()
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.colorbar(label="Height")
    plt.title("A* vs Catmull-Rom Spline Path")
    plt.savefig(os.path.join(save_dir, "path_spline_comparison.png"))
    plt.close()

