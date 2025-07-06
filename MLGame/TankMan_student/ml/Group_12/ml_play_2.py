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
MODEL_AIM_PATH = os.path.join(BASE_DIR,"model_2a.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR,"model_2b.zip")

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
        self.player: str = "2P"

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
        
        self.player = scene_info["id"]

        self._scene_info = scene_info
        def find_nearest_competitor(player, competitors):
            px, py = player['x'], player['y']
            nearest = None
            min_distance = float('inf')
            
            for competitor in competitors:
                cx, cy = competitor['x'], competitor['y']
                distance = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest = competitor
            
            return nearest,min_distance
        
        nearest_target, md = find_nearest_competitor(scene_info, scene_info['competitor_info'])
        print(nearest_target,md)
        if nearest_target:
            self.target_x = nearest_target['x']
            self.target_y = -nearest_target['y']
        else:
            self.target_x = None
            self.target_y = None
            print("No valid target available.")
            return "RESET"
        
        def detect_wall_collision(player, walls, obs):
            px, py = player['x']+25, player['y']+25
            for wall in walls:
                wx, wy = wall['x']+25, wall['y']+25
                if math.sqrt((px - wx) ** 2 + (py - wy) ** 2) <= 60*math.sqrt(2):
                    if wx-px < 0 and obs[0] == 0:
                        return COMMAND_CHASE[TURN_RIGHT_CMD]
                    elif wx - px > 0 and obs[0] == 4:
                        return COMMAND_CHASE[TURN_RIGHT_CMD]
                    elif wy - py > 0 and obs[0] == 1 or obs[0] == 2 or obs[0] == 3:
                        return COMMAND_CHASE[TURN_LEFT_CMD]
                    elif wy - py > 0 and obs[0] == 5 or obs[0] == 6 or obs[0] == 7:
                        return COMMAND_CHASE[TURN_LEFT_CMD]
                else:
                    return False
                    
        obs = self._get_obs_chase()
        result = detect_wall_collision(scene_info, scene_info['walls_info'],obs)            
        if  result == False:
            if md <= 50:
                obs = self._get_obs_aim()
                action, _ = self.model_aim.predict(obs, deterministic=True)
                command = COMMAND_AIM[action]
            else:
                action, _ = self.model_chase.predict(obs, deterministic=True)
                command = COMMAND_CHASE[action]
        
        else:

            return result
        '''
        test = 1 #testing
        if test == 0:
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
            print(command)
            #return [FORWARD_CMD]
        '''
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