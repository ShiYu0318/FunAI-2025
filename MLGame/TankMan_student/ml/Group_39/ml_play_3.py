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

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
MODEL_DIR = os.path.join(BASE_DIR, "Group_39")
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "model_3a")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "model_1b")

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
        """
        Constructor

        @param side A string like "1P" or "2P" indicates which player the `MLPlay` is for.
        """
        self.side = ai_name
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "3P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = None
        self.target_y = None


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        competitors = scene_info.get("competitor_info", [])

        # 玩家座標
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        # 玩家彈藥量
        power = scene_info.get("power", 0)
        # 玩家油量
        oil = scene_info.get("oil", 0)

        # 預設目標為 None
        self.target_x = None
        self.target_y = None

        # 如果目前的目標消失或無效，重新找最近的敵方玩家
        if self.target_x is None or self.target_y is None:
            if competitors:
                closest_distance = float('inf')  # 用來儲存最短距離
                for competitor in competitors:
                    # 計算敵方玩家與玩家的距離
                    competitor_x = competitor["x"]
                    competitor_y = competitor["y"] * -1  # y 座標可能需要轉換為正值

                    # 計算玩家和敵方玩家的歐氏距離
                    distance = (competitor_x - player_x) ** 2 + (competitor_y - player_y) ** 2

                    # 更新最短距離並設定目標
                    if distance < closest_distance:
                        closest_distance = distance
                        self.target_x = competitor_x + 15
                        self.target_y = competitor_y# - 2

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        # 根據距離選擇模型
        if closest_distance < 300:
            # 使用 model_aim 進行推論 (當目標距離小於 300)
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            # 使用 model_chase 進行推論 (當目標距離大於或等於 300)
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        print(f"Target is: ({self.target_x}, {self.target_y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command



    def find_closest_station(self, player_x, player_y, stations):
        """
        找到最近的站點，並返回其座標
        """
        import math
        min_distance = float('inf')
        target_x, target_y = None, None

        for station in stations:
            distance = math.hypot(station['x'] - player_x, -station['y'] - player_y)
            if distance < min_distance:
                min_distance = distance
                target_x = station['x']
                target_y = -station['y']

        return target_x, target_y



    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

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