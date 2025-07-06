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
MODEL_AIM_PATH = os.path.join(BASE_DIR, "model_3a.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "model_3b.zip")

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
        
        self.right = 0
        self.left = 0
        self.forward = 0

        self.last_command = None


    def get_dist(self, x1, x2, y1, y2):
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def get_dist2(self, x1, x2, y1, y2):
        return abs(x1 - x2) + abs(y1 - y2)


    def teammate_same_line_dist(self, scene_info):
        
        for teammate in scene_info["teammate_info"]:
            if teammate["id"] == scene_info["id"]:
                continue

            tx, ty = teammate["x"], -teammate["y"]

            obs =  self.get_obs_aim(
                self.player,
                tx,
                ty,
                scene_info,
            )

            if (8 + (obs[0] - obs[1])) % 8 in [0, 1, 7]:
                return self.get_dist(self.player_x, tx, self.player_y, ty)
        return 2001



    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        self.player = scene_info["id"]
        self.player_x = scene_info["x"]
        self.player_y = -scene_info["y"]
        self.oil = scene_info["oil"]
        self.power = scene_info["power"]

        if self.oil < 30: # 油量低, 找最近的 oil station
            enemy_dist = 2001
            for oil_station in scene_info["oil_stations_info"]:
                oil_station_x = oil_station["x"]
                oil_station_y = -oil_station["y"]
                curr_dist = self.get_dist(oil_station_x, self.player_x, oil_station_y, self.player_y)
                if curr_dist < enemy_dist and abs(enemy_dist - curr_dist) > 20:
                    enemy_dist = curr_dist
                    self.target = oil_station["id"]
                    self.target_x = oil_station_x
                    self.target_y = oil_station_y
        elif self.power == 0: # 沒子彈, 找最近的 bullet station
            enemy_dist = 2001
            for oil_station in scene_info["bullet_stations_info"]:
                competitor_x = oil_station["x"]
                competitor_y = -oil_station["y"]
                curr_dist = self.get_dist(competitor_x, self.player_x, competitor_y, self.player_y)
                if curr_dist < enemy_dist and abs(enemy_dist - curr_dist) > 20:
                    enemy_dist = curr_dist
                    self.target = oil_station["id"]
                    self.target_x = competitor_x
                    self.target_y = competitor_y
        else:
            # 敵人


            # for i in range(3):
            #     if scene_info["competitor_info"][i]["lives"] > 0:
            #         self.target = scene_info["competitor_info"][i]["id"]
            #         self.target_x = scene_info["competitor_info"][i]["x"]
            #         self.target_y = -scene_info["competitor_info"][i]["y"]


            enemy_dist = 2001
            for competitor in scene_info["competitor_info"]:
                competitor_x = competitor["x"]
                competitor_y = -competitor["y"]
                curr_dist = self.get_dist(competitor_x, self.player_x, competitor_y, self.player_y)
                if curr_dist < enemy_dist and abs(enemy_dist - curr_dist) > 20:
                    enemy_dist = curr_dist
                    self.target = competitor["id"]
                    self.target_x = competitor_x
                    self.target_y = competitor_y

        # TODO 牆
        # TODO 邊界


        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        enemy_dist = self.get_dist(self.player_x, self.target_x, self.player_y, self.target_y)


        if enemy_dist < 120 and self.target != "oil" and self.target != "bullets" and random.random() < 0.8:
            
            if random.random() < 0.5:
                command = [SHOOT]
            else:
                obs = self._get_obs_aim()
                action, _ = self.model_aim.predict(obs, deterministic=True)
                command = COMMAND_AIM[action]




            print(f"TEAMMATE = {self.teammate_same_line_dist(scene_info)}")
            if command == [SHOOT] and self.teammate_same_line_dist(scene_info) < enemy_dist:
                obs = self._get_obs_chase()
                action, _ = self.model_chase.predict(obs, deterministic=True)
                command = COMMAND_CHASE[action]
            else:
                command = [SHOOT]


        else:
            if random.random() >0.5:
                command = [FORWARD_CMD]
            else:
                obs = self._get_obs_chase()
                action, _ = self.model_chase.predict(obs, deterministic=True)
                command = COMMAND_CHASE[action]
            # obs = self._get_obs_chase()
            # if (8 + (obs[0] - obs[1])) % 8 in [1, 2, 3]:
            #     command = [TURN_RIGHT_CMD]
            #     self.right += 1
            # elif (8 + (obs[0] - obs[1])) % 8 in [5, 6, 7]:
            #     command = [TURN_LEFT_CMD]
            #     self.left += 1
            # else:
            #     command = [FORWARD_CMD]
            #     self.forward += 1

        # print(f"({self.left}, {self.right}, {self.forward})")


        # if self.last_command == [TURN_RIGHT_CMD] and command == [TURN_LEFT_CMD]:
        #     command = random.choice([[FORWARD_CMD], [BACKWARD_CMD]])
        # self.last_command = command

        print(f"Target is : {self.target}")
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