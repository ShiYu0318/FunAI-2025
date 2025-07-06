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

BASE_DIR = os.path.dirname(__file__)  # 當前檔案所在的資料夾
MODEL_AIM_PATH = os.path.join(BASE_DIR, "aim3.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "chase3.zip")

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
        
        # avoid wall steps
        self.wall_right_count = 0
        self.wall_left_count = 0
        self.wall_forward_count = 0
        self.wall_back_count = 0
        
        self.basic_size = 50
        self.margin_of_error = 20
        
        
        
    def get_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    def get_target(self, player_x, player_y, scene_info: dict) -> dict:
        competitors = scene_info['competitor_info']
        if not competitors:
            return None
        
        target = {"x": 0, "y": 0, "id": "None", "distance": 100000, "lives": 0}
        
        for competitor in competitors:
            distance = self.get_distance(player_x, player_y, competitor['x'], competitor['y'])
            if(distance < target["distance"]):
                target["distance"] = distance
                target["x"] = competitor['x'] - self.basic_size/2
                target["y"] = competitor['y'] - self.basic_size/2
                target["id"] = competitor['id']
                target["lives"] = competitor['lives']
        return target
    
    def is_near_wall(self, player_x, player_y, wall):
        wall_x = wall['x'] - self.basic_size/2
        wall_y = wall['y'] - self.basic_size/2
        distance = np.sqrt((player_x - wall_x) ** 2 + (player_y - wall_y) ** 2)
        return distance < 100 

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        self.player = scene_info["id"]
        
        player_x = scene_info["x"] - self.basic_size/2
        player_y = scene_info["y"] - self.basic_size/2
        
        print(f"Player at ({player_x}, {player_y})")
        
        # shoot control
        player_power = scene_info["power"]
        gun_angle = scene_info['gun_angle']
        
        target = self.get_target(player_x, player_y, scene_info)
        if target:
            self.target_x = target['x']
            self.target_y = target['y']
            print(f"Target is : {target['id']} ({self.target_x}, {self.target_y})")
            print(f"Distance to target: {target['distance']}")
        else:
            print("No valid target available.")
            return "RESET"


        #TODO : Implement the model prediction
        # Randomly switch between model_aim and model_chase
        if target["distance"] < 260 and (abs(player_x - target["x"]) < self.margin_of_error or abs(player_y - target["y"]) < self.margin_of_error):
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
        
        # avoid wall steps
        if self.wall_right_count > 0:
            self.wall_right_count -= 1
            # print(f"Avoid wall right{self.wall_right_count}")
            command = [TURN_RIGHT_CMD]
        elif self.wall_left_count > 0:
            self.wall_left_count -= 1
            # print(f"Avoid wall left{self.wall_left_count}")
            command = [TURN_LEFT_CMD]
        elif self.wall_forward_count > 0:
            self.wall_forward_count -= 1
            # print(f"Avoid wall forwar{self.wall_forward_count}")
            command = [FORWARD_CMD]
        elif self.wall_back_count > 0:
            self.wall_back_count -= 1
            # print(f"Avoid wall back{self.wall_back_count}")
            command = [BACKWARD_CMD]
        else:
            for wall in scene_info['walls_info']:
                if self.is_near_wall(player_x, player_y, wall):
                    self.wall_right_count = random.randint(1, 2)
                    self.wall_forward_count = 4
        
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
        # print("Chase obs: " + str(obs))
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
        # print("Aim angle: " + str(angle_to_target))
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
