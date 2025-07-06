import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD
import random
from collections import deque

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree

BASE_DIR = os.path.dirname(__file__)  # 當前檔案所在的資料夾
MODEL_AIM_PATH = os.path.join(BASE_DIR, "R_model_aim.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "R_model_chase.zip")

COMMAND_AIM = [
    ["NONE"],
    [AIM_LEFT_CMD],
    [AIM_RIGHT_CMD],
    [SHOOT],   
]

COMMAND_CHASE = [
    ["NONE"],
    [FORWARD_CMD],
    [BACKWARD_CMD],
    [TURN_LEFT_CMD],
    [TURN_RIGHT_CMD],
]

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"
        self._last_position = deque()
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)
        
        self.target_x, self.target_y = None, None

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        self._scene_info = scene_info
        self._set_nearest_enemy()  
        used_frame = scene_info["used_frame"]


        if self.target_x is None or self.target_y is None:
            return "RESET"

        # #停滯太久
        # if self._is_stuck(used_frame):
        #     print(f"[Frame {used_frame}] +++++++++++++++++++")
        #     return ["TURN_LEFT", "FORWARD"] 
        
        # 判斷是否要去補子彈（如何知道是哪個power？？)
        if scene_info["power"] == 0:
            return self._move_to_nearest_station("oil_stations_info")

        # 根據距離決定aim或chase
        distance = math.dist((self.target_x, self.target_y), (scene_info["x"], scene_info["y"]))
        if scene_info["power"] > 0 and distance <= 300:
            obs = self._get_obs_aim()  # 取得 `model_aim` 的觀測資料
            action, _ = self.model_aim.predict(obs, deterministic=True)
            return COMMAND_AIM[action]
        
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            return COMMAND_CHASE[action]
# # 停滯的定義
#     def _is_stuck(self, used_frame):
#         current_x, current_y = self._scene_info['x'], self._scene_info['y']

#         # 確保 _last_position 存在，且初始化為當前座標與 frame
#         if not hasattr(self, "_last_position") or not self._last_position:
#             self._last_position = (current_x, current_y, used_frame)
#             return False

#         last_x, last_y, last_frame = self._last_position

#         # 超過 30 frame 後更新 last_position
#         if used_frame - last_frame >= 30:
#             self._last_position = (current_x, current_y, used_frame)

        
#         return (current_x, current_y) == (last_x, last_y)
    
# 最近的敵人定義
    def _set_nearest_enemy(self):
        
        my_x, my_y = self._scene_info["x"], self._scene_info["y"]
        min_dist = float("inf")
        nearest_enemy = None

        for enemy in self._scene_info.get("competitor_info", []):
            enemy_x, enemy_y = enemy["x"], enemy["y"]
            dist = math.dist((my_x, my_y), (enemy_x, enemy_y))
            if dist < min_dist:
                min_dist = dist
                nearest_enemy = enemy

        if nearest_enemy:
            self.target_x, self.target_y = nearest_enemy["x"], nearest_enemy["y"]

#攻擊目標
    def _attack_target(self):
        scene_info = self._scene_info
        gun_angle_index = self._angle_to_index(scene_info["gun_angle"])
        target_angle_index = self._angle_to_index(math.degrees(math.atan2(
            self.target_y - scene_info["y"], 
            self.target_x - scene_info["x"]
        )))
        angle_diff = (target_angle_index - gun_angle_index) % 8

        if angle_diff == 0:
            return [SHOOT]
        elif angle_diff in [1, 2, 3]:
            return [AIM_LEFT_CMD]
        elif angle_diff in [5, 6, 7]:
            return [AIM_RIGHT_CMD]
        return ["NONE"]

    
#移動到最近的敵人附近
    def _move_to_nearest_station(self, station_type):
        my_x, my_y = self._scene_info["x"], self._scene_info["y"]
        min_dist = float("inf")
        nearest_station = None

        for station in self._scene_info.get(station_type, []):
            station_x, station_y = station["x"], station["y"]
            dist = math.dist((my_x, my_y), (station_x, station_y))
            if dist < min_dist:
                min_dist = dist
                nearest_station = station

        if nearest_station:
            self.target_x, self.target_y = nearest_station["x"], nearest_station["y"]
            return [FORWARD_CMD]
        return [TURN_LEFT_CMD]

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index: int = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        print("Chase obs: " + str(obs))
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print("Aim angle: " + str(angle_to_target))
        obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
        return obs

    def _get_obs_chase(self) -> np.ndarray:
        return self.get_obs_chase(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _get_obs_aim(self) -> np.ndarray:
        return self.get_obs_aim(
            self.player,
            self.target_x,
            self.target_y,
            self._scene_info,
        )

    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))

    def reset(self):
        print(f"Resetting Game {self.side}")