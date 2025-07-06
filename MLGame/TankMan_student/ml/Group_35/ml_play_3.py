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

WIDTH = 1000  # pixel
HEIGHT = 600  # pixel
TANK_SPEED = 8  # pixel
CELL_PIXEL_SIZE = 50  # pixel
DEGREES_PER_SEGMENT = 45  # degree

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim_3_steps.zip")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase_3.zip")

COMMAND_AIM = [
    ["NONE"],
    [AIM_LEFT_CMD],
    [AIM_RIGHT_CMD],
    [SHOOT],   
    [FORWARD_CMD],
    [BACKWARD_CMD],
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
        self.closest_competitor = None

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

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):

        """
        Generate the command according to the received scene information
        """
        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index: int = self._angle_to_index(tank_angle)
        if scene_info["status"] != "GAME_ALIVE":
            self.closest_competitor = None
            return "RESET"
        self._scene_info = scene_info

        if not self.closest_competitor or self.closest_competitor not in scene_info["competitor_info"]:
            if scene_info["competitor_info"]:  # 確保有敵人
                self.closest_competitor = min(
                    scene_info["competitor_info"],
                    key=lambda x: ((scene_info["x"] - x["x"]) ** 2 + (scene_info["y"] - x["y"]) ** 2) ** 0.5
                )
        
        if self.closest_competitor:
            self.target_x = self.closest_competitor["x"]
            self.target_y = -self.closest_competitor["y"]
            print(f"Target is : ({self.target_x}, {self.target_y})")

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        # 計算 dx, dy
        dx = self.target_x - scene_info["x"]
        dy = self.target_y - scene_info["y"]

        # 計算角度
        angle_to_target = math.degrees(math.atan2(dy, dx))
        angle_to_target_index = self._angle_to_index(angle_to_target)

        # 計算距離
        dis = ((scene_info["x"] - self.target_x) ** 2 + (scene_info["y"] + self.target_y) ** 2) ** 0.5

        tank_angle = scene_info.get("angle", 0) + 180
        tank_angle_index = self._angle_to_index(tank_angle) 
        if dis <=30  :
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
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
