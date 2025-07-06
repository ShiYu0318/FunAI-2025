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
MODEL_AIM_PATH = os.path.join(BASE_DIR, "model_2a.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "model_2b.zip")

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
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        choice = False

        yourteam = "competitor_info"
        myteam = "teammate_info"
        if scene_info["id"] == '1P' or scene_info["id"] == '2P' or scene_info["id"] == '3P':
            myteam = "teammate_info"
            yourteam = "competitor_info"
        # 隊友與我的距離 / 面向
        teammate_distance = []
        teammate_angle = [0, 0, 0, 0, 0, 0, 0, 0]
        for target in scene_info[myteam]:
            if target["x"] != scene_info["x"] and target["y"] != scene_info["y"]:
                t_angle_num = int((((target["angle"] + 22.5)) % 360) // 45)
                teammate_angle[t_angle_num] += 1
                teammate_distance.append((self.get_distance(target["x"], scene_info["x"], target["y"], scene_info["y"])))

        # 敵人與我的距離 / 面相
        competitor_distance = 0
        for target in scene_info[yourteam]:
            competitor_distance = (self.get_distance(target["x"], scene_info["x"], target["y"], scene_info["y"]))
            c_angle_num = int((((target["angle"] + 22.5)) % 360) // 45)
            
            # 300 pixel 內的敵人隨機找一人攻擊
            
            if competitor_distance <= 332 and teammate_angle[c_angle_num] == 0:  # 距離332內有敵人 且 面相沒有對友
                choice = True
                com_id = (int(target["id"][0])-1) % 3


            elif competitor_distance <= 332 and teammate_angle[c_angle_num] != 0: # # 距離332內有敵人 且 面相有隊友
                for i in range(2):
                    if teammate_distance[i] > competitor_distance: # 隊友在對手後面 就離開迴圈
                        break 
                    choice = True
                    com_id = (int(target["id"][0])-1) % 3


            if choice == True: # 找到就離開迴圈
                break
                
                
        # 彈藥庫與我的距離 
        bullet_distance = []
        for target in scene_info["bullet_stations_info"]:
            bullet_distance.append(self.get_distance(target["x"], scene_info["x"], target["y"], scene_info["y"]))

        min_bu_id = bullet_distance.index(min(bullet_distance))

        if choice == True:
            self.target_x = scene_info[yourteam][com_id]["x"]
            self.target_y = scene_info[yourteam][com_id]["y"] * -1
        else:
            self.target_x = scene_info["bullet_stations_info"][min_bu_id]["x"]
            self.target_y = scene_info["bullet_stations_info"][min_bu_id]["y"] * -1

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        # Randomly switch between model_aim and model_chase
        if choice == True:
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

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

        segment_center = (angle + DEGREES_PER_SEGMENT/2) // DEGREES_PER_SEGMENT
        return int(segment_center % (360 // DEGREES_PER_SEGMENT))
    
    def get_distance(self, x1, y1, x2, y2):

        return ((x1-x2)**2 + (y1-y2)**2)**0.5