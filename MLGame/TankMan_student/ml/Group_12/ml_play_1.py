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

BASE_DIR = os.path.dirname(__file__)
MODEL_AIM_PATH = os.path.join(BASE_DIR, "aim_rl.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "test_model.zip")

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
        self.player: str = "1P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = None
        self.target_y = None

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        
        self.player = scene_info["id"]
        
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        # 獲取最近敵人座標
        closest_enemy = self.get_closest_enemy(scene_info, self.player)
        if closest_enemy is None:
            print("No valid enemy available.")
            return "RESET"

        self.target_x, self.target_y = closest_enemy

        # 計算與目標的歐幾里得距離
        player_info = next(player for player in scene_info["teammate_info"] if player["id"] == self.player)
        player_x, player_y = player_info["x"], -player_info["y"]
        distance_to_target = math.sqrt((self.target_x - player_x) ** 2 + (self.target_y - player_y) ** 2)

        # 判斷是否在同一水平線上
        coordinate_tolerance = 15  # 允許的誤差範圍
        is_on_same_horizontal_or_vertical_line = (
            abs(player_x - self.target_x) <= coordinate_tolerance or
            abs(player_y - self.target_y) <= coordinate_tolerance
        )

        # print(f"Distance to target: {distance_to_target}")
        print(f"Player position: ({player_x}, {player_y})")
        print(f"Target position: ({self.target_x}, {self.target_y})")
        print(f"On same horizontal/vertical line: {is_on_same_horizontal_or_vertical_line}")

        # 根據距離和水平/垂直對齊判斷模型切換
        if distance_to_target <= 250 and is_on_same_horizontal_or_vertical_line:
            # 切換到 aim 模型
            print("Switching to aim model")
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            # 使用 chase 模型
            print("Using chase model")
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        print(f"Predicted action: {command}")
        self.time += 1
        return command

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

    def get_closest_enemy(self, scene_info: dict, player_id: str) -> tuple[int, int]:
        player_info = next(player for player in scene_info["teammate_info"] if player["id"] == player_id)
        player_x, player_y = player_info["x"], -player_info["y"]

        # 計算與每個敵人的距離，並選擇最近的
        closest_enemy = None
        min_distance = float("inf")
        for enemy in scene_info["competitor_info"]:
            enemy_x, enemy_y = enemy["x"], -enemy["y"]
            distance = math.sqrt((enemy_x - player_x) ** 2 + (enemy_y - player_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_enemy = (enemy_x, enemy_y)

                print("enemy id = " + str(enemy["id"]))
                # print("min_distance = " + str(min_distance))

        return closest_enemy