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



wall_centers = [
    [125, -100],
    [150, -100],
    [175, -100],

    [150, -450],
    [175, -450],
    [150, -475],
    [175, -475],

    [800, -100],
    [825, -100],
    [850, -100],
    [825, -75],
    [825, -125],

    [800, -475],
    [825, -475],
    [850, -475],
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
        self.target = None

        self.competitor_dist = 1500 # 不可能超過


    def get_dist(self, x1, x2, y1, y2):
        '''
        兩點距離
        '''
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


    def select_target(self, scene_info):
        self.target = None
        self.target_x = None
        self.target_y = None

        if scene_info["oil"] < 15:
            # 沒油了, 先去找最近的 oil station

            oil_dist = 1500 # 不可能超過, 要先歸零

            for oil_station in scene_info["oil_stations_info"]:
                ox = oil_station["x"]
                oy = -oil_station["y"]
                curr_dist = self.get_dist(
                    self.player_x, ox,
                    self.player_y, oy
                )
                if curr_dist < oil_dist:
                    oil_dist = curr_dist
                    self.target_x = ox
                    self.target_y = oy
                    self.target = oil_station["id"]


        elif scene_info["power"] == 0:
            # 沒子彈了, 先去找最近的 bullet station

            bullet_dist = 1500 # 不可能超過, 要先歸零

            for bullet_station in scene_info["bullet_stations_info"]:
                bx = bullet_station["x"]
                by = -bullet_station["y"]
                curr_dist = self.get_dist(
                    self.player_x, bx,
                    self.player_y, by
                )
                # 因為 bullet 很多，很可能會循環選擇打架, 所以設定一個門檻
                if curr_dist < bullet_dist and abs(curr_dist - bullet_dist) > 200:
                    bullet_dist = curr_dist
                    self.target_x = bx
                    self.target_y = by
                    self.target = bullet_station["id"]

        else:

            # 目標先選擇最近的敵人
            self.competitor_dist = 1500 # 不可能超過, 要先歸零

            for competitor in scene_info["competitor_info"]:
                if competitor["lives"] <= 0: # 已經死了
                    continue

                cx = competitor["x"]
                cy = -competitor["y"]
                curr_dist = self.get_dist(
                    self.player_x, cx,
                    self.player_y, cy
                )
                if curr_dist < self.competitor_dist and abs(curr_dist - self.competitor_dist) > 200:
                    self.competitor_dist = curr_dist
                    self.target_x = cx
                    self.target_y = cy
                    self.target = competitor["id"]


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        self.player = scene_info["id"]


        self.player_x = scene_info["x"]
        self.player_y = -scene_info["y"]

        # print(f"player location = ({self.player_x}, {self.player_y})")


        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info

        # 目標選擇 (先歸零)
        self.select_target(scene_info)

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"
        
        # print(f"TARGET = {self.target}")
        # print(f"DIST = {self.competitor_dist}")


        if self.competitor_dist < 150 and self.target != "bullets" and self.target != "oil" and random.random() < 0.95: # 開始射擊距離

            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]

            if random.random() < 0.2:
                if command == [SHOOT]:
                    command = random.choice([[AIM_LEFT_CMD], [AIM_RIGHT_CMD]])
                else:
                    command = [SHOOT]

            # # ==== rule-based ====
            # obs = self._get_obs_aim()
            # if (8 + (obs[0] - obs[1])) % 8 == 0:
            #     command = [SHOOT]

            # elif (8 + (obs[0] - obs[1])) % 8 in [5, 6, 7]:
            #     command = [AIM_LEFT_CMD]


            # elif (8 + (obs[0] - obs[1])) % 8 in [1, 2, 3]:
            #     command = [AIM_RIGHT_CMD]

            # else:
            #     command = random.choice([[AIM_RIGHT_CMD], [AIM_LEFT_CMD]])

        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

            # # ==== rule-based ====
            # obs = self._get_obs_chase()
            # if (8 + (obs[0] - obs[1])) % 8 == 0: # same dir
            #     command = [FORWARD_CMD]

            # # 要往左 -> 往前
            # elif (8 + (obs[0] - obs[1])) % 8 == 7: 
            #     command = [TURN_LEFT_CMD]

            # # 要往右 -> 往前
            # elif (8 + (obs[0] - obs[1])) % 8 == 1: 
            #     command = [TURN_RIGHT_CMD]

            # # 要往左 -> 往後
            # elif (8 + (obs[0] - obs[1])) % 8 == 3: 
            #     command = [TURN_LEFT_CMD]

            # # 要往右 -> 往後
            # elif (8 + (obs[0] - obs[1])) % 8 == 5: 
            #     command = [TURN_RIGHT_CMD]
            
            # # 都可以
            # elif (8 + (obs[0] - obs[1])) % 8 == 2: 
            #     command = random.choice([ [TURN_RIGHT_CMD],  [TURN_LEFT_CMD]])

            # # 都可以
            # elif (8 + (obs[0] - obs[1])) % 8 == 4: 
            #     command = random.choice([ [TURN_RIGHT_CMD],  [TURN_LEFT_CMD]])

            # else: # back dir
            #     command = [BACKWARD_CMD]





        # print(f"Player is : ({self.player_x, self.player_y})")
        # print(f"Target is : ({self.target_x, self.target_y})")
        # print(f"Predicted action: {command}")
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
        gun_angle = scene_info.get("gun_angle", 0) + 180
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