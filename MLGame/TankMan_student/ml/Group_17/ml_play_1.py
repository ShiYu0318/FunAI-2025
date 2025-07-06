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
MODEL_DIR = os.path.join(BASE_DIR, "Group_17", "model")
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim.zip")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase.zip")

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
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)
        self.target_x = None
        self.target_y = None
        self.last_x = None
        self.last_y = None
        self.stuck_counter = 0

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        self.target_x, self.target_y = self._find_nearest_enemy(scene_info)

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        self._check_stuck(scene_info)

        if random.choice([True, False]):
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
        
        if self.stuck_counter > 5:
            command = [random.choice([FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD])]
        
        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command

    def _check_stuck(self, scene_info):
        if self.last_x is not None and self.last_y is not None:
            if abs(self.last_x - scene_info["x"]) < 5 and abs(self.last_y - scene_info["y"]) < 5:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        self.last_x = scene_info["x"]
        self.last_y = scene_info["y"]

    def _find_nearest_enemy(self, scene_info: dict):
        enemies = scene_info.get("enemy", [])
        if not enemies:
            return WIDTH // 2, HEIGHT // 2
        player_x, player_y = scene_info["x"], scene_info["y"]
        nearest_enemy = min(enemies, key=lambda e: math.dist((player_x, player_y), (e["x"], e["y"])))
        return nearest_enemy["x"], nearest_enemy["y"]
    
    def reset(self):
        print(f"Resetting Game {self.side}")

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index = self._angle_to_index(angle_to_target)
        return np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        gun_angle_index = self._angle_to_index(gun_angle)
        dx = target_x - player_x
        dy = target_y - player_y 
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index = self._angle_to_index(angle_to_target)
        return np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)

    def _get_obs_chase(self) -> np.ndarray:
        return self.get_obs_chase(self.player, self.target_x, self.target_y, self._scene_info)

    def _get_obs_aim(self) -> np.ndarray:
        return self.get_obs_aim(self.player, self.target_x, self.target_y, self._scene_info)

    def _angle_to_index(self, angle: float) -> int:
        angle = (angle + 360) % 360
        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))
