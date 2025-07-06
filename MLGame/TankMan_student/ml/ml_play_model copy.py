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
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim_bad.zip")
MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "best_model.zip")

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

        self.birth_positions = 0
        

    def get_closest_enemy(self, scene_info: dict) -> Optional[dict]:
        """
        遍歷 scene_info["competitor_info"]，選出與我方坦克距離最短的敵人。
        如果全部敵人已死亡或列表為空，返回 None。
        """
        competitors = scene_info.get("competitor_info", [])
        closest_enemy = None
        min_distance = float("inf")

        """
        player_info = scene_info.get(self.player, {})
        player_x = player_info.get("x", 0)
        player_y = player_info.get("y", 0)
        """
        player_x = scene_info['x']
        player_y = scene_info['y']

        for enemy in competitors:
            if(enemy["lives"] == 0): continue
            # 如果需要，可根據 enemy 狀態判斷是否存活，這裡假設敵方若存在數值就是存活
            enemy_x = enemy.get("x", 0)
            enemy_y = enemy.get("y", 0)
            distance = (((enemy_x - player_x)**2 + (enemy_y - player_y)**2)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_enemy = enemy
        print(competitors)
        return closest_enemy


    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        self._scene_info = scene_info
        # 比賽會對應
        self.player = scene_info["id"]

        player_x = scene_info['x']
        player_y = scene_info['y']

        if scene_info['used_frame'] == 0:
            self.birth_positions = player_x
            print(f"id = {self.player},初始位置{self.birth_positions}")
            # print("Recording birth positions")
            # 記錄所有隊伍的出生點
            # for teammate in scene_info['teammate_info']:
            #     tank_id = teammate['id']
            #     self.birth_positions[tank_id] = (teammate['x'], teammate['y'])

            # # 記錄敵方坦克的出生點
            # for enemy in scene_info['competitor_info']:
            #     tank_id = enemy['id']
            #     self.birth_positions[tank_id] = (enemy['x'], enemy['y'])

            # print(f"Recorded birth positions: {self.birth_positions}")



        # 改成對手座標
        # self.target_x = 500
        # self.target_y = 300
        # self.target_x = scene_info["competitor_info"][0]['x'] + 200
        # self.target_y = -scene_info["competitor_info"][0]['y']
        
        closest_enemy = self.get_closest_enemy(scene_info)
        if closest_enemy is not None:
            self.target_x = closest_enemy.get("x", 0)
            self.target_y = -closest_enemy.get("y", 0)
        else:
            # 如果找不到有效的敵人，則使用預設目標（例如保持原位置或其他策略）
            print("找不到存活的敵人，採取預設策略")
            self.target_x = scene_info[self.player].get("x", 0)
            self.target_y = -scene_info[self.player].get("y", 0)
        


        if self.birth_positions >= 500:
            self.target_x += 200
        else:
            self.target_x -= 200

        distance = ((self.target_x - player_x)**2 + (self.target_y - player_y)**2) ** 0.5

        if self.target_x is None or self.target_y is None:
            print("No valid target available.")
            return "RESET"

        # Randomly switch between model_aim and model_chase
        
        # 修改使用的 Model
        # if random.choice([True, False]):
        #     obs = self._get_obs_aim()
        #     action, _ = self.model_aim.predict(obs, deterministic=True)
        #     command = COMMAND_AIM[action]
        # else:
        if distance >= 70 and abs(-self.target_y - player_y) > 2:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]
        else:
            command = [SHOOT]

        print(f"dis:{distance}")
        print(f"差距：{abs(-self.target_y - player_y)}")
        print(f"Target is : ({self.target_x, self.target_y})")
        print(f"Predicted action: {command}\n")
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
