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
MODEL_AIM_PATH = os.path.join(BASE_DIR, "aim_bad.zip")
MODEL_CHASE_PATH = os.path.join(BASE_DIR, "chase_best.zip")
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 上層資料夾
# MODEL_DIR = os.path.join(BASE_DIR, "model")
# MODEL_AIM_PATH = os.path.join(MODEL_DIR, "aim_bad.zip")
# MODEL_CHASE_PATH = os.path.join(MODEL_DIR, "chase_best.zip")

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
        # print(f"Initial Game {ai_name} ML script")
        self.time = 0
        self.player: str = "1P"

        # Load the trained models
        self.model_aim = PPO.load(MODEL_AIM_PATH)
        self.model_chase = PPO.load(MODEL_CHASE_PATH)

        self.target_x = None
        self.target_y = None
    #new def********************************************************************************
    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)   
    def _find_nearest_enemy(self, competitors: list) -> Optional[dict]:
        """
        從 competitors 中找到最近的敵人。
        """
        if not competitors:
            # print("No competitors available.")
            return None

        # 玩家自己的位置
        player_x = self._scene_info["x"]
        player_y = self._scene_info["y"]

        # 使用 min 函數找到距離最近的敵人
        nearest_enemy = min(
            competitors,
            key=lambda comp: math.sqrt((comp["x"] - player_x) ** 2 + (comp["y"] - player_y) ** 2)
        )

        return nearest_enemy

    #***************************************************************************************
    def update(self, scene_info: dict, keyboard=[], *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        self._scene_info = scene_info
        self.player = scene_info["id"]

        # self.target_x = random.randint(0, WIDTH)
        # self.target_y = random.randint(0, HEIGHT)
        if scene_info["competitor_info"]:
            nearest_enemy = self._find_nearest_enemy(scene_info["competitor_info"])
            self.target_x = nearest_enemy["x"]
            self.target_y = nearest_enemy["y"]
        else:
            # print("No enemy found.")
            return "RESET"

        if self.target_x is None or self.target_y is None:
            # print("No valid target available.")
            return "RESET"

        # # Randomly switch between model_aim and model_chase
        # if random.choice([True, False]):
        #     obs = self._get_obs_aim()
        #     action, _ = self.model_aim.predict(obs, deterministic=True)
        #     command = COMMAND_AIM[action]
        # else:
        #     obs = self._get_obs_chase()
        #     action, _ = self.model_chase.predict(obs, deterministic=True)
        #     command = COMMAND_CHASE[action]
    
        player_x = self._scene_info["x"]
        player_y =self._scene_info["y"]
        player_angle = scene_info["angle"]
        gun_angle = scene_info["gun_angle"]

        # 計算與敵人的距離和角度
        dx = self.target_x - player_x
        dy = self.target_y - player_y
        distance = self._calculate_distance(player_x, player_y, self.target_x, self.target_y)
        target_angle = math.degrees(math.atan2(dy, dx))
        target_angle = (target_angle + 360) % 360  # 確保角度在 [0, 360)
        # print(distance)

        # 判斷敵人方向是否在夾角區域
        angle_diff = (target_angle - gun_angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # 如果敵人位於夾角區域，先調整位置
        if abs(angle_diff) > 22.5:  # 夾角區域閾值（調整到最近的方向）
            # 判斷應該向哪個主要方向移動
            if abs(dx) > abs(dy):  # 水平方向
                if dx > 0:
                    move_direction = [FORWARD_CMD]  # 向右移動
                else:
                    move_direction = [BACKWARD_CMD]  # 向左移動
            else:  # 垂直方向
                if dy > 0:
                    move_direction = [FORWARD_CMD]  # 向下移動
                else:
                    move_direction = [BACKWARD_CMD]  # 向上移動

            # 返回移動指令
            # print(f"Adjusting position: {move_direction}")
            return move_direction

        # 如果在射程內且方向對齊，則射擊
        if distance <= 300 and abs(angle_diff) <= 22.5:
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        else:
            # 否則追擊敵人
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        # 如果敵人在射程內，瞄準並發射
        if distance <= 300:
            obs = self._get_obs_aim()
            action, _ = self.model_aim.predict(obs, deterministic=True)
            command = COMMAND_AIM[action]
        # 如果敵人不在射程內，追擊敵人
        else:
            obs = self._get_obs_chase()
            action, _ = self.model_chase.predict(obs, deterministic=True)
            command = COMMAND_CHASE[action]

        # print(f"Target is : ({self.target_x, self.target_y})")
        # print(f"Predicted action: {command}")
        self.time += 1
        return command


    def reset(self):
        """
        Reset the status
        """
        # print(f"Resetting Game {self.side}")

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