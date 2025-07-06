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
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "model_1a")
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

        @param side A string like "1P" or "2P" indicates which player the `MLPlay` is for.
        """
        self.side = ai_name
        print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"

        # Load the trained models
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

        # Prioritize attacking the first enemy in competitor_info
        if scene_info["competitor_info"]:
            nearest_enemy = scene_info["competitor_info"][0]
        else:
            print("No valid target available.")
            return "RESET"

        self.target_x = nearest_enemy["x"]
        if nearest_enemy["y"] - 100 < 0:
            self.target_y = nearest_enemy["y"] + 100
        else:
            self.target_y = nearest_enemy["y"] - 100

        # Move to the target x position
        if abs(scene_info["x"] - self.target_x) > TANK_SPEED:
            if scene_info["x"] < self.target_x:
                command = ["BACKWARD"]
            else:
                command = ["FORWARD"]
        elif abs(scene_info["y"] - self.target_y) > TANK_SPEED:
            # Turn 90 degrees to be perpendicular to the enemy
            if scene_info["angle"] % 180 != 90:
                command = ["TURN_LEFT"] if scene_info["angle"] < 90 or scene_info["angle"] > 270 else ["TURN_RIGHT"]
            else:
                # Move to the target y position
                if scene_info["y"] < self.target_y:
                    command = ["FORWARD"]
                else:
                    command = ["BACKWARD"]
        else:
            # Aim the gun towards the enemy
            angle_to_enemy = math.degrees(math.atan2(nearest_enemy["y"] - scene_info["y"], nearest_enemy["x"] - scene_info["x"]))
            angle_diff = (scene_info["gun_angle"] - angle_to_enemy + 360) % 360
            if angle_diff > 180:
                angle_diff -= 360

            if abs(angle_diff) > DEGREES_PER_SEGMENT:
                if angle_diff > 0:
                    command = ["AIM_RIGHT"]
                else:
                    command = ["AIM_LEFT"]
            else:
                # Use aim model to attack
                obs = self._get_obs_aim()
                action, _ = self.model_aim.predict(obs, deterministic=True)
                command = COMMAND_AIM[action]

        print(f"Target is : ({self.target_x, self.target_y})")
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