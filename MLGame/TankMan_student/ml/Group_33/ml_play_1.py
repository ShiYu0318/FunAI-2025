import random
from typing import Optional
import pygame
import os
import numpy as np
from stable_baselines3 import PPO
from mlgame.utils.enum import get_ai_name
import math
from src.env import BACKWARD_CMD, FORWARD_CMD, TURN_LEFT_CMD, TURN_RIGHT_CMD, SHOOT, AIM_LEFT_CMD, AIM_RIGHT_CMD

WIDTH = 1000 # pixel
HEIGHT = 600 # pixel
TANK_SPEED = 8 # pixel
CELL_PIXEL_SIZE = 50 # pixel
DEGREES_PER_SEGMENT = 45 # degree

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
# MODEL_DIR = os.path.join(BASE_DIR, "")
# MODEL_AIM_PATH = os.path.join(MODEL_DIR, ".\Group_33\model_1a.zip")
# MODEL_CHASE_PATH = os.path.join(MODEL_DIR, ".\Group_33\model_1a.zip")

BASE_DIR = os.path.dirname(__file__)  # 當前檔案所在的資料夾
MODEL_AIM_PATH = os.path.join(BASE_DIR, "model_1a.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "model_1b.zip")

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
    def dis(self,x1,y1,x2,y2):
        return((x1-x2)**2+(y1-y2)**2)**0.5

    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        self.player = scene_info["id"]
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        self.target_x = 1e9
        self.target_y = 1e9

        for target in scene_info['competitor_info']:
            if self.dis(scene_info['x'],scene_info['y'],self.target_x,self.target_y) > self.dis(scene_info['x'],scene_info['y'],target['x'],target['y']):
                self.target_x=target['x']
                self.target_y=target['y']

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"
        
        # Randomly switch between model_aim and model_chase
        if self.dis(scene_info['x'],scene_info['y'],self.target_x,self.target_y) <250:
            obs = self._get_obs_aim() 
            action, _= self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
            print('Now is' +'\033[34m' + 'aim'+'\033[0m')
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
            print(self.target_x,self.target_y)

        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command

        r"""
    import random

class ResupplyEnv:
    def __init__(self):
        # 初始化其他屬性
        self.model_aim = ...  # 初始化 model_aim
        self.model_chase = ...  # 初始化 model_chase

    def _get_obs_aim(self):
        # 獲取觀察值的實現
        pass

    def _get_obs_chase(self):
        # 獲取觀察值的實現
        pass

    def _should_use_aim_model(self):
        # 根據某些條件決定是否使用 aim model
        return random.choice([True, False])

    def some_method(self):
        if self._should_use_aim_model():
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
        # 其他程式碼
        """
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
