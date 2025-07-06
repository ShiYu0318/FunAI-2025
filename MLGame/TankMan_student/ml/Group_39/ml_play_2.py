import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD

WIDTH = 1000  # pixel
HEIGHT = 600  # pixel
TANK_SPEED = 8  # pixel
CELL_PIXEL_SIZE = 50  # pixel
DEGREES_PER_SEGMENT = 45  # degree

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
MODEL_DIR = os.path.join(BASE_DIR, "Group_39")
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "model_3a")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "best_model")

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
        """
        self.side = ai_name
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "2P"

        # Load trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH, device='cpu')
        self.model_chase = PPO.load(MODEL_CHASE_PATH, device='cpu')

        self.target_x = None
        self.target_y = None

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        # Find the nearest enemy
        nearest_enemy = self._find_nearest_enemy(scene_info["competitor_info"])
        if nearest_enemy is None:
            print("No valid target available.")
            return "RESET"

        self.target_x = nearest_enemy["x"] + 100 if random.choice([True, False]) else nearest_enemy["y"] + 100
        self.target_y = nearest_enemy["y"] 

        # Move to the target position
        if abs(scene_info["x"] - self.target_x) > TANK_SPEED:
            if scene_info["x"] < self.target_x:
                command = [BACKWARD_CMD]
            else:
                command = [FORWARD_CMD]
        elif abs(scene_info["y"] - self.target_y) > TANK_SPEED:
            if scene_info["angle"] % 180 != 90:
                command = [TURN_LEFT_CMD] if scene_info["angle"] < 90 or scene_info["angle"] > 270 else [TURN_RIGHT_CMD]
            else:
                if scene_info["y"] < self.target_y:
                    command = [FORWARD_CMD]
                else:
                    command = [BACKWARD_CMD]
        else:
            # Use aim model to attack
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]

        # **如果對準敵人，強制開火**
        if self._is_aimed_at_target():
            command = [SHOOT]

        print(f"Target is : ({self.target_x}, {self.target_y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command

    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

    def _is_aimed_at_target(self) -> bool:
        """
        判斷砲塔是否對準敵人
        """
        if self.target_x is None or self.target_y is None:
            return False

        player_x = self._scene_info["x"]
        player_y = self._scene_info["y"]
        gun_angle = (self._scene_info["gun_angle"] + self._scene_info["angle"]) % 360

        dx = self.target_x - player_x
        dy = self.target_y - player_y
        angle_to_target = math.degrees(math.atan2(dy, dx)) % 360

        angle_diff = abs(gun_angle - angle_to_target)
        angle_diff = min(angle_diff, 360 - angle_diff)  # 確保使用最短旋轉角度

        return angle_diff <= 10  # 允許 10° 的誤差

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
        segment_center = (angle + DEGREES_PER_SEGMENT / 2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))

    def _find_nearest_enemy(self, competitors: list) -> Optional[dict]:
        """
        Find the nearest enemy based on the current position
        """
        min_distance = float('inf')
        nearest_enemy = None
        for enemy in competitors:
            distance = math.sqrt((enemy["x"] - self._scene_info["x"]) ** 2 + (enemy["y"] - self._scene_info["y"]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_enemy = enemy
        return nearest_enemy
