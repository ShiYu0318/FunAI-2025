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
MODEL_AIM_PATH = os.path.join(BASE_DIR, "realbam1.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "realbam1.zip")

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
        # print(scene_info)

        self.x = scene_info.get("x",0)
        self.y = scene_info.get("y",0)

        mode = 0 # 0 = navigate, 1 = attack
        self.target_x = 0
        self.target_y = 0
        oil = scene_info.get("oil",0)
        power = scene_info.get("power",0)
        nearest = np.inf
        if oil < 30:
            mode = 0
            for i in scene_info.get("oil_stations_info"):
                ix = i.get("x",0)
                iy = i.get("y",0)
                d = self.get_distance(self.x,self.y,ix,iy)
                if d < nearest:
                    self.target_x = ix
                    self.target_y = iy
                    nearest = d
        elif power < 4:
            mode = 0
            for i in scene_info.get("bullet_stations_info"):
                ix = i.get("x",0)
                iy = i.get("y",0)
                d = self.get_distance(self.x,self.y,ix,iy)
                if d < nearest:
                    self.target_x = ix
                    self.target_y = iy
                    nearest = d
        else:
            mode = 1
            for i in scene_info.get("competitor_info"):
                ix = i.get("x",0)
                iy = i.get("y",0)
                d = self.get_distance(self.x,self.y,ix,iy)
                if d < nearest:
                    self.target_x = ix
                    self.target_y = iy
                    nearest = d
                    # print(i["id"])

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"
        # mode=0
        # rule
        # print(mode,nearest)
        if scene_info.get("cooldown",0) == 0:
            obs = self._get_obs_aim()
            if mode == 1 and (abs(obs[3]) < 18 or abs(obs[4]) < 18) and obs[2] < 250: # TODO: is 250 the right distance?
                print("in")
                # obs = self._get_obs_aim()
                if obs[0] == obs[1]:
                    action, _ = self.model_aim.predict(obs, deterministic=True)
                else:
                    if obs[0] - obs[1] < obs[0]+8 - obs[1]:
                        action = 1
                    else:
                        action = 2
                command = COMMAND_AIM[action]
            elif obs[2] < 300:
                print("in 2")
                if obs[3] > 0: # on right
                    obs = self._get_obs_chase()
                    if obs[0] != 4:
                        action = 3
                    else:
                        action = 1
                elif obs[3] < 0:
                    obs = self._get_obs_chase()
                    if obs[0] != 0:
                        action = 3
                    else:
                        action = 1
                else:
                    action = 1
                command = COMMAND_CHASE[action]
            else:
                obs = self._get_obs_chase()
                # action, _ = self.model_chase.predict(obs, deterministic=True)
                if obs[2] == 0:
                    if obs[0] in [(obs[1] + x) % 8 for x in [5, 6, 7]]:
                        action = 3
                    elif obs[0] in [(obs[1] + x) % 8 for x in [1, 2, 3]]:
                        action = 4
                    elif obs[0] == obs[1]:
                        action = 1
                    elif (obs[0] == (obs[1] + 4) % 8):
                        action = 2
                else:
                    if obs[3] not in [(obs[0]+x)%8 for x in [7,0,1]]:
                        action = 3
                    else:
                        action = 1
                command = COMMAND_CHASE[action]
        else:
            command = COMMAND_AIM[0]

        print(f"Target is : ({self.target_x, self.target_y}) Now on : ({self.x,self.y})")
        print(f"Predicted action: {command}")
        self.time += 1
        return command


    def reset(self):
        """
        Reset the status
        """
        print(f"Resetting Game {self.side}")

    def get_distance(self,x, y, x1, y1) -> float:
        return ((x-x1)**2 + (y-y1)**2)**0.5

    def get_obs_chase(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        # print("chase")
        player_x = scene_info.get("x", 0)
        player_y = scene_info.get("y", 0)
        tank_angle = scene_info.get("angle", 0)
        # print(scene_info.get("angle", 0),scene_info.get("angle", 0)+180)
        tank_angle_index: int = self._angle_to_index(tank_angle)
        dx = target_x - player_x
        dy = target_y - player_y  # Adjust for screen coordinate system
        angle_to_target = 180-math.degrees(math.atan2(dy, dx))
        angle_to_target_index: int = self._angle_to_index(angle_to_target)
        wall = 0
        wallDir = 0
        nearestWall = np.inf
        for i in scene_info.get("walls_info",[]):
            wdx = i.get("x",0)
            wdy = i.get("y",0)
            d = self.get_distance(player_x,player_y,wdx,wdy)
            if d < nearestWall:
                d = nearestWall
                if d < 100:
                    wall = 1
                    wallDir = self._angle_to_index(math.degrees(math.atan2((player_y-wdy),(player_x-wdx))))

        # Return gun_angle and normalized angle_to_target
        obs = np.array([
            float(tank_angle_index),
            float(angle_to_target_index),
            float(wall),
            float(wallDir)
        ], dtype=np.float32)
        # player_x = scene_info.get("x", 0)
        # player_y = -scene_info.get("y", 0)
        # tank_angle = scene_info.get("angle", 0) + 180 #? +180 for what?
        # tank_angle_index: int = self._angle_to_index(tank_angle)
        # dx = target_x - player_x
        # dy = target_y - player_y
        # angle_to_target = math.degrees(math.atan2(dy, dx))
        # angle_to_target_index: int = self._angle_to_index(angle_to_target)
        # obs = np.array([float(tank_angle_index), float(angle_to_target_index)], dtype=np.float32)
        print("Target angle",angle_to_target)
        print("Chase obs: " , obs)
        return obs

    def get_obs_aim(self, player: str, target_x: int, target_y: int, scene_info: dict) -> np.ndarray:
        # print(scene_info)
        player_x = scene_info.get("x", 0)
        player_y = -scene_info.get("y", 0)
        gun_angle = scene_info.get("gun_angle", 0)# + scene_info.get("angle", 0) + 180
        gun_angle_index: int = self._angle_to_index(gun_angle)
        # Calculate angle to the target
        dx = target_x - player_x
        dy = target_y - player_y  # WTF? Adjust for screen coordinate system
        angle_to_target = 180 - math.degrees(math.atan2(dy, dx))
        print(angle_to_target)
        angle_to_target_index: int = self._angle_to_index(angle_to_target)

        distance = self.get_distance(player_x, -player_y, target_x, target_y)
        power = scene_info.get("power", 0)
        print("xy",player_x,player_y)

        # Return gun_angle and normalized angle_to_target
        obs = np.array([
            float(gun_angle_index),
            float(angle_to_target_index),
            float(distance),
            float(dx),
            float(dy),
            float(power),
        ], dtype=np.float32)
        # player_x = scene_info.get("x", 0)
        # player_y = -scene_info.get("y", 0)
        # gun_angle = scene_info.get("gun_angle", 0) + scene_info.get("angle", 0) + 180
        # gun_angle_index: int = self._angle_to_index(gun_angle)
        # dx = target_x - player_x
        # dy = target_y - player_y 
        # angle_to_target = math.degrees(math.atan2(dy, dx))
        # angle_to_target_index: int = self._angle_to_index(angle_to_target)
        print("Aim angle: " + str(angle_to_target))
        print("OBS:",obs)
        # obs = np.array([float(gun_angle_index), float(angle_to_target_index)], dtype=np.float32)
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
